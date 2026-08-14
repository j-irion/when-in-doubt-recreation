"""ImageNet architectures used in Rawat et al. (2021)."""

import torch
from torch import nn


def imagenet_models(classes: int, width: float) -> tuple[nn.Module | None, nn.Module]:
    """Public Noisy Student L2-475 teacher and the paper's MobileNetV3 student."""
    import timm

    student = timm.create_model(f"mobilenetv3_large_{int(width * 100):03d}", pretrained=False, num_classes=classes)
    if classes == 17_203:
        return None, student  # The paper uses an oracle teacher on ImageNet-21k.
    teacher = timm.create_model("tf_efficientnet_l2.ns_jft_in1k_475", pretrained=True, num_classes=1000)
    return teacher, student


def logits(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(inputs["pixel_values"])
