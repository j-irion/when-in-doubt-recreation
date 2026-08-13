"""The student/teacher architectures named in the paper."""

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


class CifarBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == channels
            else nn.Sequential(nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.convs(x) + self.shortcut(x))


class CifarResNet(nn.Module):
    """The 6n+2 CIFAR ResNets used for ResNet-{8,14,20,32,44,56}."""

    def __init__(self, depth: int = 32, classes: int = 100):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("CIFAR ResNet depth must be 6n+2")
        blocks = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = self._layer(16, 16, blocks)
        self.layer2 = self._layer(16, 32, blocks, 2)
        self.layer3 = self._layer(32, 64, blocks, 2)
        self.head = nn.Linear(64, classes)

    @staticmethod
    def _layer(in_channels: int, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            CifarBlock(in_channels, channels, stride),
            *[CifarBlock(channels, channels) for _ in range(blocks - 1)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return self.head(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


def mnli_models() -> tuple[nn.Module, nn.Module]:
    """Public equivalents for the paper's fine-tuned RoBERTa-Large and MobileBERT."""
    teacher = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli")
    student = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", num_labels=3, ignore_mismatched_sizes=True
    )
    return teacher, student


def cifar_models(teacher_checkpoint: str, depth: int) -> tuple[nn.Module, nn.Module]:
    """Load the paper's L2 architecture only from a supplied CIFAR-100 checkpoint."""
    if not teacher_checkpoint:
        raise ValueError("CIFAR-100 requires a fine-tuned EfficientNet-L2 teacher checkpoint")
    import timm

    teacher = timm.create_model("tf_efficientnet_l2", pretrained=False, num_classes=100)
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=True)
    teacher.load_state_dict(checkpoint.get("state_dict", checkpoint))
    return teacher, CifarResNet(depth)


def load(task: str, teacher_checkpoint: str = "", cifar_depth: int = 32) -> tuple[nn.Module, nn.Module]:
    if task == "mnli":
        return mnli_models()
    if task == "cifar100":
        return cifar_models(teacher_checkpoint, cifar_depth)
    raise ValueError(f"unsupported task: {task}")


def logits(task: str, model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if task == "cifar100":
        return model(inputs["pixel_values"])
    return model(**inputs).logits
