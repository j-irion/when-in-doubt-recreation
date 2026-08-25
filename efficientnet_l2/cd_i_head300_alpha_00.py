import json
import time
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = ROOT / "artifacts/imagenet1k-cdi-head300-alpha00"
METHOD = "class"  # "baseline", "class", "margin"

# reported by paper
ALPHA = 0.0
STUDENT_WIDTH = 0.75
IN_DOMAIN = None  # derived from the 300 largest train folders
MARGIN_IN_DOMAIN = 0.4

# not reported by paper
RHO_TRAIN = 0.8
TEACHER_BATCH_SIZE = 32
STUDENT_BATCH_SIZE = 256
WORKERS = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = "cuda"


class Images(Dataset):
    def __init__(self, directory: Path, train: bool, teacher: bool):
        folder = ImageFolder(directory)
        self.images = folder.samples
        self.classes = folder.classes
        if teacher:
            self.transform = Compose(
                [
                    Resize(475),
                    CenterCrop(475),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = Compose(
                [
                    RandomResizedCrop(224) if train else Resize(256),
                    RandomHorizontalFlip() if train else CenterCrop(224),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, label = self.images[index]
        return self.transform(default_loader(path)), label, index


def loader(images, batch_size, shuffle=False):
    return DataLoader(
        images,
        batch_size,
        shuffle=shuffle,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=WORKERS > 0,
    )


if not (DATA / "train").is_dir() or not (DATA / "val").is_dir():
    raise SystemExit(f"Expected {DATA}/train and {DATA}/val")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

torch.backends.cudnn.benchmark = True
OUTPUT.mkdir(parents=True, exist_ok=True)
teacher_train_loader = loader(
    Images(DATA / "train", train=False, teacher=True), TEACHER_BATCH_SIZE
)
teacher_val_loader = loader(
    Images(DATA / "val", train=False, teacher=True), TEACHER_BATCH_SIZE
)
student_train_data = Images(DATA / "train", train=True, teacher=False)
student_val_data = Images(DATA / "val", train=False, teacher=False)
if student_train_data.classes != student_val_data.classes:
    raise SystemExit("train and validation class mappings differ")
class_counts = [0] * len(student_train_data.classes)
for _, label in student_train_data.images:
    class_counts[label] += 1
IN_DOMAIN = set(
    sorted(
        range(len(class_counts)),
        key=lambda class_id: (-class_counts[class_id], student_train_data.classes[class_id]),
    )[:300]
)
print(
    "head-300 train images per class:",
    min(class_counts[class_id] for class_id in IN_DOMAIN),
    "to",
    max(class_counts[class_id] for class_id in IN_DOMAIN),
)
student_train_loader = loader(student_train_data, STUDENT_BATCH_SIZE, shuffle=True)
student_val_loader = loader(student_val_data, STUDENT_BATCH_SIZE)

teacher = (
    timm.create_model("tf_efficientnet_l2.ns_jft_in1k_475", pretrained=True)
    .to(DEVICE)
    .eval()
)
student_name = f"mobilenetv3_large_{int(STUDENT_WIDTH * 100):03d}"
student = timm.create_model(student_name, pretrained=False, num_classes=1000).to(DEVICE)


def cached_logits(data_loader, name):
    path = OUTPUT / name
    if path.exists():
        print(f"using {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    scores = []
    with torch.inference_mode():
        for batch, (images, _, _) in enumerate(data_loader, 1):
            scores.append(teacher(images.to(DEVICE)).cpu())
            if batch % 100 == 0 or batch == len(data_loader):
                print(f"{name}: {batch}/{len(data_loader)} batches")
    scores = torch.cat(scores)
    torch.save(scores, path)
    return scores


teacher_train = cached_logits(teacher_train_loader, "teacher_train.pt")
teacher_val = cached_logits(teacher_val_loader, "teacher_val.pt")
optimizer = torch.optim.AdamW(
    student.parameters(), lr=LEARNING_RATE
)  # not specified in the paper
in_domain = torch.tensor(sorted(IN_DOMAIN), device=DEVICE)

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0.0
    for batch, (images, labels, indices) in enumerate(student_train_loader, 1):
        labels = labels.to(DEVICE)
        teacher_scores = teacher_train[indices].to(DEVICE)
        teacher_probs = torch.softmax(teacher_scores, dim=1)

        if METHOD == "baseline":
            targets = teacher_probs
        elif METHOD == "class":
            targets = teacher_probs.clone()
            hard = ~torch.isin(labels, in_domain)
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(
                labels[hard], 1000
            ) + ALPHA / 1000
        elif METHOD == "margin":
            top_two = teacher_probs.topk(2, dim=1).values
            hard = top_two[:, 0] - top_two[:, 1] <= RHO_TRAIN
            targets = teacher_probs.clone()
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(
                labels[hard], 1000
            ) + ALPHA / 1000
        else:
            raise ValueError("METHOD must be baseline, class, or margin")

        optimizer.zero_grad()
        loss = (
            -(targets * torch.log_softmax(student(images.to(DEVICE)), dim=1))
            .sum(dim=1)
            .mean()
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        if batch % 100 == 0 or batch == len(student_train_loader):
            print(
                f"epoch {epoch + 1}/{EPOCHS}: {batch}/{len(student_train_loader)} batches"
            )
    print(
        f"epoch {epoch + 1}/{EPOCHS}: loss={total_loss / len(student_train_data):.4f}"
    )

torch.save(student.state_dict(), OUTPUT / "student.pt")
student.eval()
student_scores, labels = [], []
with torch.inference_mode():
    for images, batch_labels, _ in student_val_loader:
        student_scores.append(student(images.to(DEVICE)).cpu())
        labels.append(batch_labels)
student_scores = torch.cat(student_scores)
labels = torch.cat(labels)
teacher_prediction = teacher_val.argmax(1)
student_prediction = student_scores.argmax(1)
student_probs = torch.softmax(student_scores, dim=1)
top_two = student_probs.topk(2, dim=1).values
margin = top_two[:, 0] - top_two[:, 1]
teacher_probs = torch.softmax(teacher_val, dim=1)
teacher_top_two = teacher_probs.topk(2, dim=1).values
teacher_margin = teacher_top_two[:, 0] - teacher_top_two[:, 1]
in_domain_rows = (
    teacher_margin >= MARGIN_IN_DOMAIN
    if METHOD == "margin"
    else torch.isin(labels, in_domain.cpu())
)
in_domain_definition = (
    f"teacher margin >= {MARGIN_IN_DOMAIN}"
    if METHOD == "margin"
    else "true label in configured class IDs"
)


def metric(prediction, keep_student):
    rows = in_domain_rows
    return {
        "overall_accuracy": (prediction == labels).float().mean().item(),
        "overall_student_fraction": keep_student.float().mean().item(),
        "in_domain_accuracy": (prediction[rows] == labels[rows]).float().mean().item(),
        "in_domain_student_fraction": keep_student[rows].float().mean().item(),
    }


def latency_ms(model, size):
    images = torch.randn(1, 3, size, size, device=DEVICE)
    with torch.inference_mode():
        for _ in range(10):
            model(images)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            model(images)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 50


student_ms = latency_ms(student, 224)
teacher_ms = latency_ms(teacher, 475)
metrics = {
    "run": {
        "gpu": torch.cuda.get_device_name(0),
        "student_width": STUDENT_WIDTH,
        "method": METHOD,
        "alpha": ALPHA,
        "in_domain_selection": "300 largest train folders; synset breaks ties",
        "rho_train": RHO_TRAIN if METHOD == "margin" else None,
        "in_domain_definition": in_domain_definition,
        "in_domain_class_ids": sorted(IN_DOMAIN) if METHOD != "margin" else None,
        "margin_in_domain_threshold": MARGIN_IN_DOMAIN if METHOD == "margin" else None,
        "student_forward_ms_batch_1": student_ms,
        "teacher_forward_ms_batch_1": teacher_ms,
        "latency_note": "Warm GPU forward-pass time only; excludes image loading, preprocessing, routing, and remote-teacher transport.",
    },
    "teacher_only": metric(
        teacher_prediction, torch.zeros_like(labels, dtype=torch.bool)
    ),
    "student_only": metric(
        student_prediction, torch.ones_like(labels, dtype=torch.bool)
    ),
    "margin_delegation": [],
}
metrics["teacher_only"]["expected_cascade_forward_ms"] = teacher_ms
metrics["student_only"]["expected_cascade_forward_ms"] = student_ms

if METHOD != "margin":
    class_keep = torch.isin(student_prediction, in_domain.cpu())
    class_prediction = torch.where(class_keep, student_prediction, teacher_prediction)
    metrics["class_delegation"] = metric(class_prediction, class_keep)
    metrics["class_delegation"]["expected_cascade_forward_ms"] = (
        student_ms
        + (1 - metrics["class_delegation"]["overall_student_fraction"]) * teacher_ms
    )

per_class = []
for class_id, synset in enumerate(student_val_data.classes):
    rows = labels == class_id
    row = {
        "class_id": class_id,
        "synset": synset,
        "samples": rows.sum().item(),
        "teacher_accuracy": (teacher_prediction[rows] == labels[rows])
        .float()
        .mean()
        .item(),
        "student_accuracy": (student_prediction[rows] == labels[rows])
        .float()
        .mean()
        .item(),
    }
    if METHOD != "margin":
        row.update(
            {
                "in_domain": class_id in IN_DOMAIN,
                "class_delegation_accuracy": (class_prediction[rows] == labels[rows])
                .float()
                .mean()
                .item(),
                "class_delegation_student_fraction": class_keep[rows]
                .float()
                .mean()
                .item(),
            }
        )
    per_class.append(row)


def min_max(rows, key):
    low = min(rows, key=lambda row: row[key])
    high = max(rows, key=lambda row: row[key])
    return {
        "min": {
            "class_id": low["class_id"],
            "synset": low["synset"],
            "accuracy": low[key],
        },
        "max": {
            "class_id": high["class_id"],
            "synset": high["synset"],
            "accuracy": high[key],
        },
    }


metrics["per_class"] = per_class
metrics["per_class_min_max"] = {
    "teacher": min_max(per_class, "teacher_accuracy"),
    "student": min_max(per_class, "student_accuracy"),
}
if METHOD != "margin":
    in_domain_classes = [row for row in per_class if row["in_domain"]]
    metrics["per_class_min_max"].update(
        {
            "class_delegation": min_max(per_class, "class_delegation_accuracy"),
            "in_domain_student": min_max(in_domain_classes, "student_accuracy"),
            "in_domain_class_delegation": min_max(
                in_domain_classes, "class_delegation_accuracy"
            ),
        }
    )
for rho in range(101):
    keep_student = margin >= rho / 100
    prediction = torch.where(keep_student, student_prediction, teacher_prediction)
    row = {"rho": rho / 100, **metric(prediction, keep_student)}
    row["expected_cascade_forward_ms"] = (
        student_ms + (1 - row["overall_student_fraction"]) * teacher_ms
    )
    metrics["margin_delegation"].append(row)

with (OUTPUT / "metrics.json").open("w") as file:
    json.dump(metrics, file, indent=2)
print(json.dumps(metrics["per_class_min_max"], indent=2))
print(
    json.dumps(
        {
            name: value
            for name, value in metrics.items()
            if name not in {"margin_delegation", "per_class"}
        },
        indent=2,
    )
)
print(f"wrote {OUTPUT / 'metrics.json'}")
