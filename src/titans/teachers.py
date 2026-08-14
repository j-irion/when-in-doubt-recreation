"""Frozen EfficientNet-L2 inference cached once per ImageNet-1k split."""

from pathlib import Path

import torch
from torch import nn

from .data import split_batch
from .models import logits


def cache_logits(teacher: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, path: Path) -> torch.Tensor:
    """Return one L2 score row per loader example, reusing an existing cache."""
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)

    teacher.eval()
    cached: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            inputs, _ = split_batch(batch, "teacher", device)
            cached.append(logits(teacher, inputs).cpu())
    scores = torch.cat(cached)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scores, path)
    return scores
