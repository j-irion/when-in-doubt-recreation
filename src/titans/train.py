"""Straightforward student training against frozen teacher scores."""

import torch
from torch import nn

from .data import split_batch
from .losses import (
    class_specific_targets,
    distillation_loss,
    margin_targets,
    standard_targets,
)
from .models import logits


def train(
    task: str,
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    teacher_logits: torch.Tensor,
    device: torch.device,
    method: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
    rho_train: float,
    in_domain_classes: torch.Tensor,
) -> list[float]:
    """Train with one of the paper's classification targets; cache order matches loader order."""
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        offset = 0
        total = 0.0
        for batch in loader:
            inputs, labels = split_batch(batch, "student", device)
            batch_teacher = teacher_logits[offset : offset + len(labels)].to(device)
            offset += len(labels)
            if method == "baseline":
                targets = standard_targets(batch_teacher)
            elif method == "class":
                in_domain = torch.isin(labels, in_domain_classes.to(device))
                targets = class_specific_targets(batch_teacher, labels, in_domain, alpha)
            elif method == "margin":
                targets, _ = margin_targets(batch_teacher, labels, rho_train, alpha)
            else:
                raise ValueError(f"unknown method: {method}")
            optimizer.zero_grad()
            loss = distillation_loss(logits(task, student, inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        losses.append(total / offset)
    return losses
