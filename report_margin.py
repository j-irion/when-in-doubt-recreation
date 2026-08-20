"""Rewrite margin-run metrics using Figure 7's student-margin in-domain mask."""

import json
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = Path("artifacts/imagenet1k-margin")
STUDENT_WIDTH = 0.75
STUDENT_BATCH_SIZE = 256
WORKERS = 8
MARGIN_IN_DOMAIN = 0.4  # Figure 7 caption
DEVICE = "cuda"


class Images(Dataset):
    def __init__(self, directory):
        self.images = ImageFolder(directory).samples
        self.transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, label = self.images[index]
        return self.transform(default_loader(path)), label


if not (DATA / "val").is_dir() or not torch.cuda.is_available():
    raise SystemExit("ImageNet validation data and CUDA GPU required")
if not (OUTPUT / "student.pt").exists() or not (OUTPUT / "teacher_val.pt").exists():
    raise SystemExit(f"Expected student.pt and teacher_val.pt in {OUTPUT}")

loader = DataLoader(
    Images(DATA / "val"),
    STUDENT_BATCH_SIZE,
    num_workers=WORKERS,
    pin_memory=True,
    persistent_workers=WORKERS > 0,
)
student = timm.create_model(
    f"mobilenetv3_large_{int(STUDENT_WIDTH * 100):03d}", pretrained=False, num_classes=1000
).to(DEVICE)
student.load_state_dict(torch.load(OUTPUT / "student.pt", map_location=DEVICE, weights_only=True))
student.eval()

student_scores, labels = [], []
with torch.inference_mode():
    for images, batch_labels in loader:
        student_scores.append(student(images.to(DEVICE)).cpu())
        labels.append(batch_labels)
student_scores = torch.cat(student_scores)
labels = torch.cat(labels)
teacher_scores = torch.load(OUTPUT / "teacher_val.pt", map_location="cpu", weights_only=True)
if len(labels) != len(teacher_scores):
    raise SystemExit("student and cached teacher validation scores have different lengths")

student_prediction = student_scores.argmax(1)
teacher_prediction = teacher_scores.argmax(1)
student_probs = torch.softmax(student_scores, dim=1)
top_two = student_probs.topk(2, dim=1).values
margin = top_two[:, 0] - top_two[:, 1]
in_domain_rows = margin >= MARGIN_IN_DOMAIN


def metric(prediction, keep_student):
    return {
        "overall_accuracy": (prediction == labels).float().mean().item(),
        "overall_student_fraction": keep_student.float().mean().item(),
        "in_domain_accuracy": (prediction[in_domain_rows] == labels[in_domain_rows]).float().mean().item(),
        "in_domain_student_fraction": keep_student[in_domain_rows].float().mean().item(),
    }


with (OUTPUT / "metrics.json").open() as file:
    metrics = json.load(file)
run = metrics["run"]
run.update(
    {
        "method": "margin",
        "in_domain_definition": f"student margin >= {MARGIN_IN_DOMAIN}",
        "in_domain_class_ids": None,
        "margin_in_domain_threshold": MARGIN_IN_DOMAIN,
    }
)
student_ms = run["student_forward_ms_batch_1"]
teacher_ms = run["teacher_forward_ms_batch_1"]
metrics["teacher_only"] = metric(teacher_prediction, torch.zeros_like(labels, dtype=torch.bool))
metrics["student_only"] = metric(student_prediction, torch.ones_like(labels, dtype=torch.bool))
metrics["teacher_only"]["expected_cascade_forward_ms"] = teacher_ms
metrics["student_only"]["expected_cascade_forward_ms"] = student_ms

metrics.pop("class_delegation", None)
metrics["margin_delegation"] = []
for rho in range(101):
    keep_student = margin >= rho / 100
    prediction = torch.where(keep_student, student_prediction, teacher_prediction)
    row = {"rho": rho / 100, **metric(prediction, keep_student)}
    row["expected_cascade_forward_ms"] = student_ms + (1 - row["overall_student_fraction"]) * teacher_ms
    metrics["margin_delegation"].append(row)

for row in metrics.get("per_class", []):
    for key in ("in_domain", "class_delegation_accuracy", "class_delegation_student_fraction"):
        row.pop(key, None)
for key in ("class_delegation", "in_domain_student", "in_domain_class_delegation"):
    metrics.get("per_class_min_max", {}).pop(key, None)

with (OUTPUT / "metrics.json").open("w") as file:
    json.dump(metrics, file, indent=2)
print(json.dumps({name: value for name, value in metrics.items() if name not in {"margin_delegation", "per_class"}}, indent=2))
print(f"wrote {OUTPUT / 'metrics.json'}")
