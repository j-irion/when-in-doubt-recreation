"""Build or validate EfficientNet-L2 logits with checkpoint-declared preprocessing."""

import argparse
import json
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = ROOT / "artifacts/teacher-cache-efficientnet-l2-475-crop936-bicubic"
MODEL_NAME = "tf_efficientnet_l2.ns_jft_in1k_475"
BATCH_SIZE = 32
WORKERS = 8
DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--validation-only", action="store_true")
args = parser.parse_args()

if not (DATA / "train").is_dir() or not (DATA / "val").is_dir():
    raise SystemExit(f"Expected {DATA}/train and {DATA}/val")
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required")

if not args.validation_only:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    paths = [OUTPUT / "teacher_train.pt", OUTPUT / "teacher_val.pt"]
    if manifest_path.exists() and all(path.exists() for path in paths):
        print(f"using {OUTPUT}")
        raise SystemExit
    if any(path.exists() for path in paths):
        raise SystemExit(f"Incomplete cache in {OUTPUT}; remove it before rebuilding")

teacher = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE).eval()
data_config = resolve_model_data_config(teacher)
transform = create_transform(**data_config, is_training=False)


def evaluate(split: str, save: bool):
    data = ImageFolder(DATA / split, transform=transform)
    loader = DataLoader(
        data,
        BATCH_SIZE,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=WORKERS > 0,
    )
    scores, labels = [], []
    with torch.inference_mode():
        for batch, (images, batch_labels) in enumerate(loader, 1):
            scores.append(teacher(images.to(DEVICE)).cpu())
            labels.append(batch_labels)
            if batch % 100 == 0 or batch == len(loader):
                print(f"{split}: {batch}/{len(loader)} batches")
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    if save:
        torch.save(scores, OUTPUT / f"teacher_{split}.pt")
    return scores, labels


if args.validation_only:
    val_scores, val_labels = evaluate("val", save=False)
    print(
        json.dumps(
            {
                "model": MODEL_NAME,
                "data_config": data_config,
                "transform": str(transform),
                "validation_top1_accuracy": (
                    val_scores.argmax(1) == val_labels
                ).float().mean().item(),
            },
            indent=2,
        )
    )
    raise SystemExit

evaluate("train", save=True)
val_scores, val_labels = evaluate("val", save=True)
manifest = {
    "model": MODEL_NAME,
    "data_config": data_config,
    "transform": str(transform),
    "validation_top1_accuracy": (val_scores.argmax(1) == val_labels).float().mean().item(),
}
with (OUTPUT / "manifest.json").open("w") as file:
    json.dump(manifest, file, indent=2)
print(json.dumps(manifest, indent=2))
