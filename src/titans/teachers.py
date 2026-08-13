"""Frozen teacher inference cached once per dataset split."""

from pathlib import Path

import torch
from torch import nn

from .data import split_batch
from .models import logits


def cache_logits(
    task: str, teacher: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, path: Path
) -> torch.Tensor:
    """Return one teacher-logit row per loader example, reusing an existing cache."""
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    teacher.eval()
    cached: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = split_batch(batch, "teacher", device)
            cached.append(logits(task, teacher, inputs).cpu())
    scores = torch.cat(cached)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, path)
    return scores
