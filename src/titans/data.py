"""Paper dataset loaders with each model's own preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class TaskData:
    train: DataLoader
    evaluation: DataLoader
    classes: int


class CifarPairs(Dataset):
    def __init__(self, root: Path, train: bool):
        self.dataset = datasets.CIFAR100(root, train=train, download=True)
        self.student_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        # EfficientNet-L2 is evaluated at the 475-pixel resolution stated in the paper.
        self.teacher_transform = transforms.Compose(
            [
                transforms.Resize(475),
                transforms.CenterCrop(475),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[index]
        return {
            "student": {"pixel_values": self.student_transform(image)},
            "teacher": {"pixel_values": self.teacher_transform(image)},
            "labels": torch.tensor(label),
        }


def cifar100(data_dir: str, batch_size: int) -> TaskData:
    root = Path(data_dir) / "cifar100"
    train = CifarPairs(root, train=True)
    evaluation = CifarPairs(root, train=False)
    return TaskData(
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(evaluation, batch_size=batch_size, shuffle=False, num_workers=2),
        100,
    )


def mnli(data_dir: str, batch_size: int) -> TaskData:
    """Use the public 9,815-example matched validation split named in the paper."""
    from datasets import load_dataset

    student_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    teacher_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
    source = load_dataset("nyu-mll/glue", "mnli", cache_dir=data_dir)
    student_collate = DataCollatorWithPadding(student_tokenizer, return_tensors="pt")
    teacher_collate = DataCollatorWithPadding(teacher_tokenizer, return_tensors="pt")

    def collate(examples: list[dict[str, object]]) -> dict[str, object]:
        premise = [str(example["premise"]) for example in examples]
        hypothesis = [str(example["hypothesis"]) for example in examples]
        return {
            "student": student_collate(student_tokenizer(premise, hypothesis, truncation=True)),
            "teacher": teacher_collate(teacher_tokenizer(premise, hypothesis, truncation=True)),
            "labels": torch.tensor([int(example["label"]) for example in examples]),
        }

    return TaskData(
        DataLoader(source["train"], batch_size=batch_size, shuffle=False, collate_fn=collate),
        DataLoader(source["validation_matched"], batch_size=batch_size, shuffle=False, collate_fn=collate),
        3,
    )


def load(task: str, data_dir: str, batch_size: int) -> TaskData:
    if task == "cifar100":
        return cifar100(data_dir, batch_size)
    if task == "mnli":
        return mnli(data_dir, batch_size)
    raise ValueError(f"unsupported task: {task}")


def split_batch(batch: dict[str, object], model: str, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {name: value.to(device) for name, value in batch[model].items()}
    return inputs, batch["labels"].to(device)
