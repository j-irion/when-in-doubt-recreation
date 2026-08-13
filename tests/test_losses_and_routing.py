import torch

from titans.losses import (
    class_specific_targets,
    margin,
    margin_targets,
    smoothed_labels,
)
from titans.route import cascade_predictions, class_delegation, margin_delegation


def test_class_specific_targets_match_equation_four():
    teacher = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = torch.tensor([0, 2])
    targets = class_specific_targets(teacher, labels, torch.tensor([True, False]), alpha=0.6)
    assert torch.allclose(targets.sum(-1), torch.ones(2))
    assert torch.allclose(targets[1], smoothed_labels(labels[1:], 3, 0.6)[0])


def test_margin_distillation_uses_strict_teacher_threshold():
    teacher = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1])
    threshold = margin(teacher)[0].item()
    _, easy = margin_targets(teacher, labels, rho_train=threshold, alpha=0.4)
    assert easy.tolist() == [False, False]


def test_margin_router_keeps_the_threshold_and_falls_back_below_it():
    student = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    keep = margin_delegation(student, margin(student)[0].item())
    assert keep.tolist() == [True, False]
    assert cascade_predictions(student, teacher, keep).tolist() == [0, 0]


def test_class_router_only_keeps_in_domain_predictions():
    student = torch.tensor([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    assert class_delegation(student, torch.tensor([1])).tolist() == [True, False]
