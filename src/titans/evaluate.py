"""Quality/delegation curves for the paper's ImageNet cascades."""

import torch
from torch import nn

from .data import split_batch
from .models import logits
from .route import cascade_predictions, class_delegation, margin_delegation


def evaluate(
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    teacher_logits: torch.Tensor | None,
    device: torch.device,
    method: str,
    in_domain_classes: torch.Tensor,
) -> dict[str, object]:
    student.eval()
    student_scores, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, batch_labels = split_batch(batch, "student", device)
            student_scores.append(logits(student, inputs).cpu())
            labels.append(batch_labels.cpu())
    student_scores = torch.cat(student_scores)
    labels = torch.cat(labels)
    teacher_scores = teacher_logits.cpu() if teacher_logits is not None else None

    def metrics(keep: torch.Tensor) -> dict[str, float]:
        prediction = student_scores.argmax(-1) if teacher_scores is None else cascade_predictions(student_scores, teacher_scores, keep)
        if teacher_scores is None:
            prediction = torch.where(keep, prediction, labels)  # The ImageNet-21k oracle knows the true label.
        return {
            "accuracy": (prediction == labels).float().mean().item(),
            "student_fraction": keep.float().mean().item(),
        }

    result: dict[str, object] = {
        "teacher_accuracy": 1.0 if teacher_scores is None else (teacher_scores.argmax(-1) == labels).float().mean().item(),
        "student_accuracy": (student_scores.argmax(-1) == labels).float().mean().item(),
        "margin_curve": [{"rho": rho / 100, **metrics(margin_delegation(student_scores, rho / 100))} for rho in range(101)],
    }
    if method == "class":
        result["class_delegation"] = metrics(class_delegation(student_scores, in_domain_classes))
    return result
