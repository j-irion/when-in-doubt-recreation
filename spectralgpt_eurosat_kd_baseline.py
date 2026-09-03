"""Resumable baseline KD: SpectralGPT+ teacher -> MobileNet student on EuroSAT-MS."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--spectralgpt-root", type=Path, default=Path("/home/irion/spectralgpt"))
parser.add_argument("--teacher-checkpoint", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--epochs", type=int, default=50, help="Total epochs; resume with a larger value to extend.")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--student-bands", type=int, choices=(12, 13), default=12)
parser.add_argument("--student-model", default="mobilenetv3_large_075")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--teacher-batch-size", type=int, default=32)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smoke-test", action="store_true", help="Check one teacher/student batch without caching or training.")
args = parser.parse_args()

root = args.spectralgpt_root.expanduser().resolve()
if not root.is_dir():
    raise SystemExit(f"Missing SpectralGPT checkout: {root}")
sys.path[:0] = [str(root / "vendor"), str(root)]
os.chdir(root)  # The official EuroSat loader reads TIFFs relative to data/.

import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import models_vit_tensor
from util.datasets import EuroSat

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

teacher_checkpoint = (
    args.teacher_checkpoint
    or root / "experiments/finetune/eurosat/checkpoint-149.pth"
).expanduser().resolve()
output = (
    args.output
    or root / f"experiments/kd/{args.student_model}-{args.student_bands}band-baseline"
).expanduser().resolve()
train_split = root / "txt_file/train_euro_result.txt"
val_split = root / "txt_file/val_euro_result.txt"
for path in (teacher_checkpoint, train_split, val_split, root / "data/tif"):
    if not path.exists():
        raise SystemExit(f"Missing {path}")

DEVICE = "cuda"
CLASSES = 10
TEACHER_DROPPED_BANDS = [10]  # Matches the completed SpectralGPT+ fine-tuning run.
student_dropped_bands = TEACHER_DROPPED_BANDS if args.student_bands == 12 else None


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def dataset(split, train, dropped_bands):
    transform = EuroSat.build_transform(train, 128, EuroSat.mean, EuroSat.std)
    return IndexedDataset(EuroSat(split, transform, dropped_bands=dropped_bands))


def loader(data, batch_size, shuffle=False):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


def checkpoint_load(path):
    # The official checkpoint contains argparse.Namespace metadata; it is trusted and MD5-verified.
    return torch.load(path, map_location="cpu", weights_only=False)


def cache_logits(data_loader, teacher, path):
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached.shape == (len(data_loader.dataset), CLASSES):
            print(f"using {path}")
            return cached
        raise SystemExit(f"Invalid cache shape in {path}; remove it and retry")

    values = []
    with torch.inference_mode():
        for batch, (images, _, _) in enumerate(data_loader, 1):
            with torch.autocast("cuda"):
                values.append(teacher(images.to(DEVICE, non_blocking=True)).cpu())
            if batch % 100 == 0 or batch == len(data_loader):
                print(f"{path.name}: {batch}/{len(data_loader)} batches")
    values = torch.cat(values)
    torch.save(values, path)
    return values


def accuracy(model, data_loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels, _ in data_loader:
            prediction = model(images.to(DEVICE, non_blocking=True)).argmax(1).cpu()
            correct += (prediction == labels).sum().item()
            total += len(labels)
    return correct / total


def save_checkpoint(path, epoch, model, optimizer, scheduler, history, best_accuracy):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduled_epochs": args.epochs,
            "history": history,
            "best_accuracy": best_accuracy,
            "student_bands": args.student_bands,
            "student_model": args.student_model,
        },
        path,
    )


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
output.mkdir(parents=True, exist_ok=True)

teacher_train_data = dataset(train_split, train=False, dropped_bands=TEACHER_DROPPED_BANDS)
teacher_val_data = dataset(val_split, train=False, dropped_bands=TEACHER_DROPPED_BANDS)
student_train_data = dataset(train_split, train=True, dropped_bands=student_dropped_bands)
student_val_data = dataset(val_split, train=False, dropped_bands=student_dropped_bands)
teacher = models_vit_tensor.vit_base_patch8_128(drop_path_rate=0.2, num_classes=CLASSES)
teacher.load_state_dict(checkpoint_load(teacher_checkpoint)["model"], strict=True)
teacher = teacher.to(DEVICE).eval()
student = timm.create_model(
    args.student_model,
    pretrained=False,
    in_chans=args.student_bands,
    num_classes=CLASSES,
).to(DEVICE)

if args.smoke_test:
    teacher_images, labels, _ = next(iter(loader(teacher_val_data, min(2, args.batch_size))))
    student_images, student_labels, _ = next(iter(loader(student_val_data, min(2, args.batch_size))))
    assert torch.equal(labels, student_labels)
    with torch.inference_mode(), torch.autocast("cuda"):
        teacher_logits = teacher(teacher_images.to(DEVICE))
        student_logits = student(student_images.to(DEVICE))
    assert teacher_logits.shape == student_logits.shape == (len(labels), CLASSES)
    assert torch.isfinite(-(torch.softmax(teacher_logits, 1) * F.log_softmax(student_logits, 1)).sum(1)).all()
    print(f"smoke test passed: teacher=12 bands, student={args.student_bands} bands")
    raise SystemExit

teacher_train = cache_logits(
    loader(teacher_train_data, args.teacher_batch_size), teacher, output / "teacher_train_12band.pt"
)
teacher_val = cache_logits(
    loader(teacher_val_data, args.teacher_batch_size), teacher, output / "teacher_val_12band.pt"
)
# Keep the validation cache tied to the same ordered split used for student accuracy.
assert len(teacher_train) == len(student_train_data)
assert len(teacher_val) == len(student_val_data)

optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
)
start_epoch = 0
history = []
best_accuracy = 0.0
last_checkpoint = output / "checkpoint_last.pt"
if args.resume:
    if not last_checkpoint.exists():
        raise SystemExit(f"Missing {last_checkpoint}; cannot resume")
    state = checkpoint_load(last_checkpoint)
    if (state["student_bands"], state["student_model"]) != (args.student_bands, args.student_model):
        raise SystemExit("Resume settings do not match saved student")
    student.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_epoch = state["epoch"]
    history = state["history"]
    best_accuracy = state["best_accuracy"]
    if state["scheduled_epochs"] == args.epochs:
        scheduler.load_state_dict(state["scheduler"])
    else:
        scheduler.step(start_epoch)  # Extend the cosine horizon to the requested total epoch count.
    print(f"resuming at epoch {start_epoch}/{args.epochs}")

if start_epoch >= args.epochs:
    raise SystemExit(f"Already completed {start_epoch} epochs; increase --epochs to continue")

student_train_loader = loader(student_train_data, args.batch_size, shuffle=True)
student_val_loader = loader(student_val_data, args.batch_size)
for epoch in range(start_epoch, args.epochs):
    torch.manual_seed(args.seed + epoch)
    student.train()
    total_loss = 0.0
    for images, _, indices in student_train_loader:
        targets = torch.softmax(teacher_train[indices].to(DEVICE), dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss = -(targets * F.log_softmax(student(images.to(DEVICE, non_blocking=True)), dim=1)).sum(1).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(indices)
    validation_top1 = accuracy(student, student_val_loader)
    scheduler.step()
    row = {
        "epoch": epoch + 1,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": total_loss / len(student_train_data),
        "validation_student_top1_accuracy": validation_top1,
    }
    history.append(row)
    best_accuracy = max(best_accuracy, validation_top1)
    save_checkpoint(last_checkpoint, epoch + 1, student, optimizer, scheduler, history, best_accuracy)
    if validation_top1 >= best_accuracy:
        save_checkpoint(output / "checkpoint_best.pt", epoch + 1, student, optimizer, scheduler, history, best_accuracy)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(row), f"best={best_accuracy:.4%}")

print(json.dumps({"best_validation_top1_accuracy": best_accuracy, "output": str(output)}, indent=2))
