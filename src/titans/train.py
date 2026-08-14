"""Student training for the paper's ImageNet experiments."""

import torch
from torch import nn

from .data import split_batch
from .losses import (
    class_specific_targets,
    distillation_loss,
    margin_targets,
    oracle_class_targets,
    standard_targets,
)
from .models import logits


def train(
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    teacher_logits: torch.Tensor | None,
    device: torch.device,
    method: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
    rho_train: float,
    in_domain_classes: torch.Tensor,
    classes: int,
) -> list[float]:
    """Train against L2 scores, or the paper's ImageNet-21k true-label oracle."""
    if teacher_logits is None and method == "margin":
        raise ValueError("margin distillation needs EfficientNet logits; ImageNet-21k uses class distillation with an oracle")
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        offset = total = 0
        for batch in loader:
            inputs, labels = split_batch(batch, "student", device)
            in_domain = torch.isin(labels, in_domain_classes.to(device))
            if teacher_logits is None:
                targets = oracle_class_targets(labels, in_domain, classes, alpha) if method == "class" else torch.nn.functional.one_hot(labels, classes).float()
            else:
                teacher = teacher_logits[offset : offset + len(labels)].to(device)
                if method == "baseline":
                    targets = standard_targets(teacher)
                elif method == "class":
                    targets = class_specific_targets(teacher, labels, in_domain, alpha)
                else:
                    targets, _ = margin_targets(teacher, labels, rho_train, alpha)
            offset += len(labels)
            optimizer.zero_grad()
            loss = distillation_loss(logits(student, inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        losses.append(total / offset)
    return losses
