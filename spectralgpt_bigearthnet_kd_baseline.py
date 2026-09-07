"""Resumable baseline/CD-I KD: fine-tuned SpectralGPT+ -> MobileNet on BigEarthNet-S2."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_IN_DOMAIN_LABELS = (
    "Mixed forest",
    "Coniferous forest",
    "Arable land",
    "Transitional woodland/shrub",
    "Broad-leaved forest",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
)
parser = argparse.ArgumentParser()
parser.add_argument("--spectralgpt-root", type=Path, default=Path.home() / "spectralgpt")
parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "artifacts/spectralgpt-bigearthnet-teacher/checkpoint-149.pth")
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/spectralgpt-bigearthnet-kd")
parser.add_argument("--train-split", type=Path, help="Defaults to the fixed 10%-per-label stratified train list.")
parser.add_argument("--method", choices=("baseline", "cd_i"), default="baseline")
parser.add_argument("--alpha", type=float, default=0.0, help="CD-I binary label smoothing for out-of-domain labels.")
parser.add_argument("--in-domain-labels", nargs="*", default=DEFAULT_IN_DOMAIN_LABELS, help="CD-I labels trained against teacher logits; default is the six most frequent training labels.")
parser.add_argument("--epochs", type=int, default=50, help="Total epochs; resume with a larger value to extend.")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--student-model", default="mobilenetv3_large_075")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--eval-batch-size", type=int, default=512)
parser.add_argument("--teacher-batch-size", type=int, default=32)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()

spectralgpt_root = args.spectralgpt_root.expanduser().resolve()
if not spectralgpt_root.is_dir():
    raise SystemExit(f"Missing SpectralGPT checkout: {spectralgpt_root}")
sys.path[:0] = [str(spectralgpt_root / "vendor"), str(spectralgpt_root)]
os.chdir(spectralgpt_root)  # Official loader resolves BE_cor relative to data/.

import timm
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import models_vit_tensor
from util.datasets import BigEarthNetFinetuneDataset, NEW_LABELS

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
output = args.output.expanduser().resolve()
train_split = (args.train_split.expanduser().resolve() if args.train_split else spectralgpt_root / "txt_file/bigearthnet_train_10pct_labelset_stratified_seed0.txt")
val_split = spectralgpt_root / "txt_file/bigearthnet_val.txt"
train_split_sha256 = hashlib.sha256(train_split.read_bytes()).hexdigest()
for path in (teacher_checkpoint, train_split, val_split, spectralgpt_root / "data/BE_cor"):
    if not path.exists():
        raise SystemExit(f"Missing {path}")

DEVICE = "cuda"
LABELS = len(NEW_LABELS)
if not 0 <= args.alpha <= 1:
    raise SystemExit("--alpha must be in [0, 1]")
unknown_labels = set(args.in_domain_labels) - set(NEW_LABELS)
if unknown_labels:
    raise SystemExit(f"Unknown in-domain labels: {sorted(unknown_labels)}")
in_domain = torch.tensor([NEW_LABELS.index(label) for label in args.in_domain_labels], device=DEVICE)
out_of_domain = torch.ones(LABELS, dtype=torch.bool, device=DEVICE)
out_of_domain[in_domain] = False


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, labels = self.dataset[index]
        return image, labels, index


def dataset(split, train):
    transform = BigEarthNetFinetuneDataset.build_transform(
        train, 128, BigEarthNetFinetuneDataset.mean, BigEarthNetFinetuneDataset.std
    )
    return IndexedDataset(BigEarthNetFinetuneDataset(split, transform))


def loader(data, batch_size, shuffle=False):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def cache_logits(data_loader, teacher, path):
    if path.exists():
        logits = torch.load(path, map_location="cpu", weights_only=True)
        if logits.shape == (len(data_loader.dataset), LABELS):
            print(f"using {path}")
            return logits
        raise SystemExit(f"Invalid cache shape in {path}; remove it and retry")
    logits = []
    with torch.inference_mode():
        for batch, (images, _, _) in enumerate(data_loader, 1):
            logits.append(teacher(images.to(DEVICE, non_blocking=True)).cpu())
            if batch % 100 == 0 or batch == len(data_loader):
                print(f"{path.name}: {batch}/{len(data_loader)} batches")
    logits = torch.cat(logits)
    torch.save(logits, path)
    return logits


def targets_for_batch(teacher_logits, labels):
    targets = torch.sigmoid(teacher_logits)
    if args.method == "cd_i":
        targets[:, out_of_domain] = labels[:, out_of_domain] * (1 - args.alpha) + args.alpha / 2
    return targets


def evaluate(student, data_loader):
    student.eval()
    targets, scores = [], []
    with torch.inference_mode():
        for images, labels, _ in data_loader:
            targets.append(labels.numpy())
            scores.append(torch.sigmoid(student(images.to(DEVICE, non_blocking=True))).cpu().numpy())
    targets, scores = np.concatenate(targets), np.concatenate(scores)
    metrics = {
        "validation_macro_mAP": float(average_precision_score(targets, scores, average="macro")),
        "validation_micro_mAP": float(average_precision_score(targets, scores, average="micro")),
    }
    if args.method == "cd_i":
        metrics.update(
            validation_in_domain_macro_mAP=float(average_precision_score(targets[:, in_domain.cpu()], scores[:, in_domain.cpu()], average="macro")),
            validation_in_domain_micro_mAP=float(average_precision_score(targets[:, in_domain.cpu()], scores[:, in_domain.cpu()], average="micro")),
        )
    return metrics


def save_checkpoint(path, epoch, student, optimizer, scheduler, history, best_micro_map):
    torch.save(
        {
            "epoch": epoch,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduled_epochs": args.epochs,
            "history": history,
            "best_micro_mAP": best_micro_map,
            "student_model": args.student_model,
            "train_split_sha256": train_split_sha256,
            "method": args.method,
            "alpha": args.alpha,
            "in_domain_labels": args.in_domain_labels,
        },
        path,
    )


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
output.mkdir(parents=True, exist_ok=True)

teacher_train_data = dataset(train_split, train=False)
student_train_data = dataset(train_split, train=True)
student_val_data = dataset(val_split, train=False)
teacher = models_vit_tensor.vit_base_patch8_128(drop_path_rate=0.2, num_classes=LABELS)
teacher.load_state_dict(load_checkpoint(teacher_checkpoint)["model"], strict=True)
teacher = teacher.to(DEVICE).eval()
student = timm.create_model(args.student_model, pretrained=False, in_chans=12, num_classes=LABELS).to(DEVICE)

if args.smoke_test:
    teacher_images, labels, _ = next(iter(loader(teacher_train_data, min(args.teacher_batch_size, 2))))
    student_images, student_labels, _ = next(iter(loader(student_train_data, min(args.batch_size, 2))))
    assert torch.equal(labels, student_labels)
    with torch.inference_mode():
        teacher_logits = teacher(teacher_images.to(DEVICE))
        student_logits = student(student_images.to(DEVICE))
    assert teacher_logits.shape == student_logits.shape == (len(labels), LABELS)
    targets = targets_for_batch(teacher_logits, labels.to(DEVICE))
    assert torch.isfinite(F.binary_cross_entropy_with_logits(student_logits, targets))
    if args.method == "cd_i" and args.alpha == 0:
        assert torch.equal(targets[:, out_of_domain], labels.to(DEVICE)[:, out_of_domain])
    print(f"smoke test passed: 12 bands, 19 labels, {args.method}, alpha={args.alpha}")
    raise SystemExit

teacher_train = cache_logits(
    loader(teacher_train_data, args.teacher_batch_size), teacher, output / f"teacher_train_logits_{train_split.stem}.pt"
)
assert len(teacher_train) == len(student_train_data)

optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
)
start_epoch = 0
history = []
best_micro_map = 0.0
last_checkpoint = output / "checkpoint_last.pt"
if args.resume:
    if not last_checkpoint.exists():
        raise SystemExit(f"Missing {last_checkpoint}; cannot resume")
    state = load_checkpoint(last_checkpoint)
    if (state["student_model"] != args.student_model or state.get("train_split_sha256") != train_split_sha256 or state.get("method", "baseline") != args.method or state.get("alpha", 0.0) != args.alpha or state.get("in_domain_labels", list(DEFAULT_IN_DOMAIN_LABELS)) != args.in_domain_labels):
        raise SystemExit("Resume checkpoint uses different model, split, or CD-I settings")
    student.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"]
    history = state["history"]
    best_micro_map = state["best_micro_mAP"]
    if state["scheduled_epochs"] == args.epochs:
        scheduler.load_state_dict(state["scheduler"])
    else:
        scheduler.step(start_epoch)
    print(f"resuming at epoch {start_epoch}/{args.epochs}")

if start_epoch >= args.epochs:
    raise SystemExit(f"Already completed {start_epoch} epochs; increase --epochs to continue")

student_train_loader = loader(student_train_data, args.batch_size, shuffle=True)
student_val_loader = loader(student_val_data, args.eval_batch_size)
for epoch in range(start_epoch, args.epochs):
    torch.manual_seed(args.seed + epoch)
    student.train()
    total_loss = 0.0
    for images, labels, indices in student_train_loader:
        targets = targets_for_batch(teacher_train[indices].to(DEVICE), labels.to(DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(student(images.to(DEVICE, non_blocking=True)), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(indices)
    metrics = evaluate(student, student_val_loader)
    scheduler.step()
    row = {
        "epoch": epoch + 1,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": total_loss / len(student_train_data),
        **metrics,
    }
    history.append(row)
    best_micro_map = max(best_micro_map, metrics["validation_micro_mAP"])
    save_checkpoint(last_checkpoint, epoch + 1, student, optimizer, scheduler, history, best_micro_map)
    if metrics["validation_micro_mAP"] >= best_micro_map:
        save_checkpoint(output / "checkpoint_best_micro_map.pt", epoch + 1, student, optimizer, scheduler, history, best_micro_map)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(row), f"best_micro_mAP={best_micro_map:.4%}")

print(json.dumps({"best_validation_micro_mAP": best_micro_map, "output": str(output)}, indent=2))
