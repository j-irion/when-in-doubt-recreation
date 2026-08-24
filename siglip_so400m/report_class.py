"""Write class-run metrics from an already-trained SigLIP student."""

import json
import time
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = ROOT / "artifacts/imagenet1k-siglip-class"
TEACHER_NAME = "vit_so400m_patch14_siglip_378.webli_ft_in1k"
STUDENT_WIDTH = 0.75
IN_DOMAIN = set(range(300))
STUDENT_BATCH_SIZE = 256
WORKERS = 8
DEVICE = "cuda"


class Images(Dataset):
    def __init__(self, directory):
        folder = ImageFolder(directory)
        self.images = folder.samples
        self.classes = folder.classes
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

torch.backends.cudnn.benchmark = True
student_data = Images(DATA / "val")
loader = DataLoader(
    student_data,
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

teacher_prediction = teacher_scores.argmax(1)
student_prediction = student_scores.argmax(1)
student_probs = torch.softmax(student_scores, dim=1)
top_two = student_probs.topk(2, dim=1).values
margin = top_two[:, 0] - top_two[:, 1]
in_domain_rows = torch.isin(labels, torch.tensor(sorted(IN_DOMAIN)))


def metric(prediction, keep_student):
    return {
        "overall_accuracy": (prediction == labels).float().mean().item(),
        "overall_student_fraction": keep_student.float().mean().item(),
        "in_domain_accuracy": (prediction[in_domain_rows] == labels[in_domain_rows]).float().mean().item(),
        "in_domain_student_fraction": keep_student[in_domain_rows].float().mean().item(),
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


teacher = timm.create_model(TEACHER_NAME, pretrained=True).to(DEVICE).eval()
teacher_config = timm.data.resolve_model_data_config(teacher)
student_ms = latency_ms(student, 224)
teacher_ms = latency_ms(teacher, teacher_config["input_size"][1])
metrics = {
    "run": {
        "gpu": torch.cuda.get_device_name(0),
        "teacher_model": TEACHER_NAME,
        "student_width": STUDENT_WIDTH,
        "method": "class",
        "rho_train": None,
        "in_domain_definition": "true label in configured class IDs",
        "in_domain_class_ids": sorted(IN_DOMAIN),
        "margin_in_domain_threshold": None,
        "student_forward_ms_batch_1": student_ms,
        "teacher_forward_ms_batch_1": teacher_ms,
        "latency_note": "Warm GPU forward-pass time only; excludes image loading, preprocessing, routing, and remote-teacher transport.",
    },
    "teacher_only": metric(teacher_prediction, torch.zeros_like(labels, dtype=torch.bool)),
    "student_only": metric(student_prediction, torch.ones_like(labels, dtype=torch.bool)),
    "margin_delegation": [],
}
metrics["teacher_only"]["expected_cascade_forward_ms"] = teacher_ms
metrics["student_only"]["expected_cascade_forward_ms"] = student_ms

class_keep = torch.isin(student_prediction, torch.tensor(sorted(IN_DOMAIN)))
class_prediction = torch.where(class_keep, student_prediction, teacher_prediction)
metrics["class_delegation"] = metric(class_prediction, class_keep)
metrics["class_delegation"]["expected_cascade_forward_ms"] = student_ms + (1 - metrics["class_delegation"]["overall_student_fraction"]) * teacher_ms

per_class = []
for class_id, synset in enumerate(student_data.classes):
    rows = labels == class_id
    per_class.append(
        {
            "class_id": class_id,
            "synset": synset,
            "in_domain": class_id in IN_DOMAIN,
            "samples": rows.sum().item(),
            "teacher_accuracy": (teacher_prediction[rows] == labels[rows]).float().mean().item(),
            "student_accuracy": (student_prediction[rows] == labels[rows]).float().mean().item(),
            "class_delegation_accuracy": (class_prediction[rows] == labels[rows]).float().mean().item(),
            "class_delegation_student_fraction": class_keep[rows].float().mean().item(),
        }
    )


def min_max(rows, key):
    low = min(rows, key=lambda row: row[key])
    high = max(rows, key=lambda row: row[key])
    return {
        "min": {"class_id": low["class_id"], "synset": low["synset"], "accuracy": low[key]},
        "max": {"class_id": high["class_id"], "synset": high["synset"], "accuracy": high[key]},
    }


in_domain_classes = [row for row in per_class if row["in_domain"]]
metrics["per_class"] = per_class
metrics["per_class_min_max"] = {
    "teacher": min_max(per_class, "teacher_accuracy"),
    "student": min_max(per_class, "student_accuracy"),
    "class_delegation": min_max(per_class, "class_delegation_accuracy"),
    "in_domain_student": min_max(in_domain_classes, "student_accuracy"),
    "in_domain_class_delegation": min_max(in_domain_classes, "class_delegation_accuracy"),
}

for rho in range(101):
    keep_student = margin >= rho / 100
    prediction = torch.where(keep_student, student_prediction, teacher_prediction)
    row = {"rho": rho / 100, **metric(prediction, keep_student)}
    row["expected_cascade_forward_ms"] = student_ms + (1 - row["overall_student_fraction"]) * teacher_ms
    metrics["margin_delegation"].append(row)

with (OUTPUT / "metrics.json").open("w") as file:
    json.dump(metrics, file, indent=2)
print(json.dumps(metrics["per_class_min_max"], indent=2))
print(f"wrote {OUTPUT / 'metrics.json'}")
