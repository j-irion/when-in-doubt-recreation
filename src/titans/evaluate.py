"""Quality/delegation curves for the two-stage classifier."""

import torch
from torch import nn

from .data import split_batch
from .models import logits
from .route import cascade_predictions, class_delegation, margin_delegation


def evaluate(
    task: str,
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    teacher_logits: torch.Tensor,
    device: torch.device,
    method: str,
    in_domain_classes: torch.Tensor,
) -> dict[str, object]:
    student.eval()
    student_scores, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, batch_labels = split_batch(batch, "student", device)
            student_scores.append(logits(task, student, inputs).cpu())
            labels.append(batch_labels.cpu())
    student_scores = torch.cat(student_scores)
    labels = torch.cat(labels)
    teacher_scores = teacher_logits.cpu()
    teacher_accuracy = (teacher_scores.argmax(-1) == labels).float().mean().item()
    student_accuracy = (student_scores.argmax(-1) == labels).float().mean().item()

    def metrics(keep: torch.Tensor) -> dict[str, float]:
        prediction = cascade_predictions(student_scores, teacher_scores, keep)
        return {
            "accuracy": (prediction == labels).float().mean().item(),
            "student_fraction": keep.float().mean().item(),
        }

    result: dict[str, object] = {
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
        "margin_curve": [{"rho": rho / 100, **metrics(margin_delegation(student_scores, rho / 100))} for rho in range(101)],
    }
    if method == "class":
        result["class_delegation"] = metrics(class_delegation(student_scores, in_domain_classes))
    return result
