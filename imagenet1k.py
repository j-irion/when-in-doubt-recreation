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
TEACHER_BATCH_SIZE = 32
STUDENT_BATCH_SIZE = 256
WORKERS = 8
EPOCHS = 90
LEARNING_RATE = 1e-3
DEVICE = "cuda"


class Images(Dataset):
    def __init__(self, directory: Path, train: bool, teacher: bool):
        self.images = ImageFolder(directory).samples
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
        return self.transform(default_loader(path)), label


def loader(images, batch_size):
    return DataLoader(
        images,
        batch_size,
        shuffle=False,
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
teacher_train_loader = loader(Images(DATA / "train", train=False, teacher=True), TEACHER_BATCH_SIZE)
teacher_val_loader = loader(Images(DATA / "val", train=False, teacher=True), TEACHER_BATCH_SIZE)
student_train_data = Images(DATA / "train", train=True, teacher=False)
student_val_data = Images(DATA / "val", train=False, teacher=False)
student_train_loader = loader(student_train_data, STUDENT_BATCH_SIZE)
student_val_loader = loader(student_val_data, STUDENT_BATCH_SIZE)

teacher = timm.create_model("tf_efficientnet_l2.ns_jft_in1k_475", pretrained=True).to(DEVICE).eval()
student_name = f"mobilenetv3_large_{int(STUDENT_WIDTH * 100):03d}"
student = timm.create_model(student_name, pretrained=False, num_classes=1000).to(DEVICE)


def cached_logits(data_loader, name):
    path = OUTPUT / name
    if path.exists():
        print(f"using {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    scores = []
    with torch.inference_mode():
        for batch, (images, _) in enumerate(data_loader, 1):
            scores.append(teacher(images.to(DEVICE)).cpu())
            if batch % 100 == 0 or batch == len(data_loader):
                print(f"{name}: {batch}/{len(data_loader)} batches")
    scores = torch.cat(scores)
    torch.save(scores, path)
    return scores


teacher_train = cached_logits(teacher_train_loader, "teacher_train.pt")
teacher_val = cached_logits(teacher_val_loader, "teacher_val.pt")
optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
in_domain = torch.tensor(sorted(IN_DOMAIN), device=DEVICE)

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0.0
    offset = 0
    for batch, (images, labels) in enumerate(student_train_loader, 1):
        labels = labels.to(DEVICE)
        teacher_scores = teacher_train[offset : offset + len(labels)].to(DEVICE)
        offset += len(labels)
        teacher_probs = torch.softmax(teacher_scores, dim=1)

        if METHOD == "baseline":
            targets = teacher_probs
        elif METHOD == "class":
            targets = teacher_probs.clone()
            hard = ~torch.isin(labels, in_domain)
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(labels[hard], 1000) + ALPHA / 1000
        elif METHOD == "margin":
            top_two = teacher_probs.topk(2, dim=1).values
            hard = top_two[:, 0] - top_two[:, 1] <= RHO_TRAIN
            targets = teacher_probs.clone()
            targets[hard] = (1 - ALPHA) * torch.nn.functional.one_hot(labels[hard], 1000) + ALPHA / 1000
        else:
            raise ValueError("METHOD must be baseline, class, or margin")

        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(student(images.to(DEVICE)), dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        if batch % 100 == 0 or batch == len(student_train_loader):
            print(f"epoch {epoch + 1}/{EPOCHS}: {batch}/{len(student_train_loader)} batches")
    print(f"epoch {epoch + 1}/{EPOCHS}: loss={total_loss / len(student_train_data):.4f}")

torch.save(student.state_dict(), OUTPUT / "student.pt")
student.eval()
student_scores, labels = [], []
with torch.inference_mode():
    for images, batch_labels in student_val_loader:
        student_scores.append(student(images.to(DEVICE)).cpu())
        labels.append(batch_labels)
student_scores = torch.cat(student_scores)
labels = torch.cat(labels)
print("teacher accuracy:", (teacher_val.argmax(1) == labels).float().mean().item())
print("student accuracy:", (student_scores.argmax(1) == labels).float().mean().item())

results = []
student_probs = torch.softmax(student_scores, dim=1)
top_two = student_probs.topk(2, dim=1).values
margin = top_two[:, 0] - top_two[:, 1]
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
