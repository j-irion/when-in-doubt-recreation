"""Target distributions from equations 3--5 and 8--10 of the paper."""

import torch
import torch.nn.functional as F


def probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def smoothed_labels(labels: torch.Tensor, classes: int, alpha: float) -> torch.Tensor:
    """(1 - alpha) p_y + alpha/L * 1."""
    return (1 - alpha) * F.one_hot(labels, classes).float() + alpha / classes


def standard_targets(teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Teacher softmax targets for the a=0, b=1 baseline."""
    return probabilities(teacher_logits, temperature)


def class_specific_targets(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    in_domain: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Equation 4: teacher targets in-domain, smoothed labels otherwise."""
    classes = teacher_logits.shape[-1]
    teacher = standard_targets(teacher_logits, temperature)
    fallback = smoothed_labels(labels, classes, alpha)
    return torch.where(in_domain[:, None], teacher, fallback)


def margin(teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Top-1 minus top-2 softmax probability (equation 6)."""
    top_two = probabilities(teacher_logits, temperature).topk(2, dim=-1).values
    return top_two[:, 0] - top_two[:, 1]


def margin_targets(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    rho_train: float,
    alpha: float,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 9 targets and its teacher-margin easy-instance mask."""
    easy = margin(teacher_logits, temperature) > rho_train
    targets = class_specific_targets(teacher_logits, labels, easy, alpha, temperature)
    return targets, easy


def distillation_loss(student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy H(target, student softmax)."""
    return -(targets * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()
