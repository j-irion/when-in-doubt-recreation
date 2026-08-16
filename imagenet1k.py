"""A first ImageNet-1k two-stage distillation experiment."""

import json
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor


DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = Path("artifacts/imagenet1k")
METHOD = "class"  # "baseline", "class", or "margin"
ALPHA = 0.6
RHO_TRAIN = 0.8
IN_DOMAIN = set(range(300))
STUDENT_WIDTH = 0.75
BATCH_SIZE = 32
WORKERS = 8
EPOCHS = 90
LEARNING_RATE = 1e-3
DEVICE = "cuda"


class Images(Dataset):
    def __init__(self, directory: Path, train: bool):
        self.images = ImageFolder(directory).samples
        self.student_transform = Compose(
            [
                RandomResizedCrop(224) if train else Resize(256),
                RandomHorizontalFlip() if train else CenterCrop(224),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.teacher_transform = Compose(
            [
                Resize(475),
                CenterCrop(475),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, label = self.images[index]
        image = default_loader(path)
        return self.student_transform(image), self.teacher_transform(image), label


if not (DATA / "train").is_dir() or not (DATA / "val").is_dir():
    raise SystemExit(f"Expected {DATA}/train and {DATA}/val")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

OUTPUT.mkdir(parents=True, exist_ok=True)
train_data = Images(DATA / "train", train=True)
val_data = Images(DATA / "val", train=False)
train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

teacher = timm.create_model("tf_efficientnet_l2.ns_jft_in1k_475", pretrained=True).to(DEVICE).eval()
student_name = f"mobilenetv3_large_{int(STUDENT_WIDTH * 100):03d}"
student = timm.create_model(student_name, pretrained=False, num_classes=1000).to(DEVICE)


def cached_logits(loader, name):
    path = OUTPUT / name
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    scores = []
    with torch.inference_mode():
        for _, teacher_images, _ in loader:
            scores.append(teacher(teacher_images.to(DEVICE)).cpu())
    scores = torch.cat(scores)
    torch.save(scores, path)
    return scores


teacher_train = cached_logits(train_loader, "teacher_train.pt")
teacher_val = cached_logits(val_loader, "teacher_val.pt")
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0.0
    offset = 0
    for student_images, _, labels in train_loader:
        labels = labels.to(DEVICE)
        teacher_scores = teacher_train[offset : offset + len(labels)].to(DEVICE)
        offset += len(labels)
        teacher_probs = torch.softmax(teacher_scores, dim=1)

        if METHOD == "baseline":
            targets = teacher_probs
        elif METHOD == "class":
            targets = teacher_probs.clone()
            hard = ~torch.isin(labels, torch.tensor(list(IN_DOMAIN), device=DEVICE))
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(labels[hard], 1000) + ALPHA / 1000
        elif METHOD == "margin":
            top_two = teacher_probs.topk(2, dim=1).values
            hard = top_two[:, 0] - top_two[:, 1] <= RHO_TRAIN
            targets = teacher_probs.clone()
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(labels[hard], 1000) + ALPHA / 1000
        else:
            raise ValueError("METHOD must be baseline, class, or margin")

        optimizer.zero_grad()
        logits = student(student_images.to(DEVICE))
        loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    print(f"epoch {epoch + 1}/{EPOCHS}: loss={total_loss / len(train_data):.4f}")

torch.save(student.state_dict(), OUTPUT / "student.pt")
student.eval()
student_scores, labels = [], []
with torch.inference_mode():
    for student_images, _, batch_labels in val_loader:
        student_scores.append(student(student_images.to(DEVICE)).cpu())
        labels.append(batch_labels)
student_scores = torch.cat(student_scores)
labels = torch.cat(labels)
print("teacher accuracy:", (teacher_val.argmax(1) == labels).float().mean().item())
print("student accuracy:", (student_scores.argmax(1) == labels).float().mean().item())

results = []
student_probs = torch.softmax(student_scores, dim=1)
margin = student_probs.topk(2, dim=1).values
margin = margin[:, 0] - margin[:, 1]
for rho in range(101):
    keep_student = margin >= rho / 100
    prediction = torch.where(keep_student, student_scores.argmax(1), teacher_val.argmax(1))
    results.append(
        {
            "rho": rho / 100,
            "accuracy": (prediction == labels).float().mean().item(),
            "student_fraction": keep_student.float().mean().item(),
        }
    )
with (OUTPUT / "results.json").open("w") as file:
    json.dump(results, file, indent=2)
