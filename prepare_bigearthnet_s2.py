"""Create the 128x128x12 BigEarthNet TIFF layout expected by SpectralGPT."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.nn import functional as F

torch.set_num_threads(1)

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, default=Path.home() / "data/bigearthnet-s2/raw/BigEarthNet-v1.0")
parser.add_argument("--spectralgpt-root", type=Path, default=Path.home() / "spectralgpt")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--limit", type=int, default=None, help="Convert only this many patches; use for a smoke test.")
args = parser.parse_args()

raw = args.raw.expanduser().resolve()
root = args.spectralgpt_root.expanduser().resolve()
output = (args.output or root / "data/BE_cor").expanduser().resolve()
splits = (root / "txt_file/bigearthnet_train.txt", root / "txt_file/bigearthnet_val.txt")
if not raw.is_dir():
    raise SystemExit(f"Missing raw BigEarthNet data: {raw}")
if any(not split.is_file() for split in splits):
    raise SystemExit(f"Missing BigEarthNet split file under {root / 'txt_file'}")

patches = []
seen = set()
for split in splits:
    for patch in split.read_text().splitlines():
        if patch and patch not in seen:
            patches.append(patch)
            seen.add(patch)
if args.limit:
    patches = patches[: args.limit]
output.mkdir(parents=True, exist_ok=True)


def resize_band(path: Path):
    band = tifffile.imread(path)
    resized = F.interpolate(
        torch.from_numpy(band).float()[None, None],
        size=(128, 128), mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    return np.rint(resized).astype(band.dtype)


def convert(patch: str, raw: Path, output: Path):
    source = raw / patch
    target = output / patch
    image_path = target / f"{patch}.tif"
    label_path = target / f"{patch}_labels_metadata.json"
    if image_path.is_file() and label_path.is_file():
        return "skipped"
    if not source.is_dir():
        raise FileNotFoundError(source)

    target.mkdir(parents=True, exist_ok=True)
    image = np.stack(
        [
            resize_band(source / f"{patch}_{band}.tif")
            for band in BANDS
        ],
        axis=-1,
    )
    if image.shape != (128, 128, 12):
        raise ValueError(f"{patch}: got {image.shape}")

    temporary = image_path.with_suffix(".tif.tmp")
    tifffile.imwrite(temporary, image)
    os.replace(temporary, image_path)
    if not label_path.exists():
        os.link(source / f"{patch}_labels_metadata.json", label_path)
    return "converted"


worker = partial(convert, raw=raw, output=output)
converted = skipped = 0
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for start in range(0, len(patches), 1000):
        for result in pool.map(worker, patches[start : start + 1000], chunksize=16):
            converted += result == "converted"
            skipped += result == "skipped"
        print(f"{min(start + 1000, len(patches))}/{len(patches)} converted={converted} skipped={skipped}", flush=True)

manifest = {
    "raw": str(raw),
    "patches_requested": len(patches),
    "converted_this_run": converted,
    "already_present": skipped,
    "bands": BANDS,
    "shape": [128, 128, 12],
    "interpolation": "bilinear (PyTorch align_corners=False)",
    "dtype": "source TIFF dtype retained",
    "labels": "hard-linked original metadata JSON",
}
(output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
