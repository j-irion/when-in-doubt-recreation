"""The paper's class- and margin-based delegation rules."""

import torch

from .losses import margin


def class_delegation(student_logits: torch.Tensor, in_domain_classes: torch.Tensor) -> torch.Tensor:
    """True when equation 4's class-based rule keeps the student prediction."""
    return torch.isin(student_logits.argmax(dim=-1), in_domain_classes.to(student_logits.device))


def margin_delegation(student_logits: torch.Tensor, rho: float) -> torch.Tensor:
    """True when equation 7 keeps the student prediction."""
    return margin(student_logits) >= rho


def cascade_predictions(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, keep_student: torch.Tensor
) -> torch.Tensor:
    """Return student argmax when retained and teacher argmax after delegation."""
    return torch.where(
        keep_student,
        student_logits.argmax(dim=-1),
        teacher_logits.argmax(dim=-1),
    )
