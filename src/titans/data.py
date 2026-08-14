"""Local ImageNet-1k and ImageNet-21k loaders; this module never downloads data."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass
class TaskData:
    train: DataLoader
    evaluation: DataLoader
    classes: int
    oracle_teacher: bool = False


class PairedImages(Dataset):
    """Apply the student and 475px EfficientNet transforms to the same image."""

    def __init__(self, samples: list[tuple[str, int]], train: bool):
        self.samples = samples
        self.student_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224) if train else transforms.Resize(256),
                transforms.RandomHorizontalFlip() if train else transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.teacher_transform = transforms.Compose(
            [
                transforms.Resize(475),
                transforms.CenterCrop(475),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, label = self.samples[index]
        image = default_loader(path)
        return {
            "student": {"pixel_values": self.student_transform(image)},
            "teacher": {"pixel_values": self.teacher_transform(image)},
            "labels": torch.tensor(label),
        }


def imagefolder_samples(root: Path) -> list[tuple[str, int]]:
    """Require an ImageFolder layout rather than silently downloading or rearranging files."""
    if not root.is_dir():
        raise FileNotFoundError(f"missing ImageNet directory: {root}")
    return [(path, label) for path, label in datasets.ImageFolder(root).samples]


def imagenet1k(data_dir: str, batch_size: int, workers: int) -> TaskData:
    root = Path(data_dir) / "imagenet1k"
    train = PairedImages(imagefolder_samples(root / "train"), train=True)
    evaluation = PairedImages(imagefolder_samples(root / "val"), train=False)
    return TaskData(
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(evaluation, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        1000,
    )


def imagenet21k_samples(root: Path, seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Paper split: classes with >=100 images, 50 deterministic test images per class."""
    rng = random.Random(seed)
    train, evaluation = [], []
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    kept = 0
    for directory in class_dirs:
        images = sorted(str(path) for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 100:
            continue
        test = set(rng.sample(images, 50))
        train.extend((path, kept) for path in images if path not in test)
        evaluation.extend((path, kept) for path in images if path in test)
        kept += 1
    if kept != 17_203:
        raise ValueError(f"expected 17,203 classes with >=100 images, found {kept}")
    return train, evaluation


def imagenet21k(data_dir: str, batch_size: int, workers: int, seed: int) -> TaskData:
    root = Path(data_dir) / "imagenet21k"
    if not root.is_dir():
        raise FileNotFoundError(f"missing ImageNet-21k directory: {root}")
    train_samples, evaluation_samples = imagenet21k_samples(root, seed)
    return TaskData(
        DataLoader(PairedImages(train_samples, train=True), batch_size=batch_size, shuffle=False, num_workers=workers),
        DataLoader(PairedImages(evaluation_samples, train=False), batch_size=batch_size, shuffle=False, num_workers=workers),
        17_203,
        oracle_teacher=True,
    )


def load(task: str, data_dir: str, batch_size: int, workers: int, seed: int) -> TaskData:
    if task == "imagenet1k":
        return imagenet1k(data_dir, batch_size, workers)
    if task == "imagenet21k":
        return imagenet21k(data_dir, batch_size, workers, seed)
    raise ValueError(f"unsupported task: {task}")


def split_batch(batch: dict[str, object], model: str, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return ({name: value.to(device) for name, value in batch[model].items()}, batch["labels"].to(device))
