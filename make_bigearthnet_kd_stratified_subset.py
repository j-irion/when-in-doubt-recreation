"""Create an exact-size, deterministic 10% multi-label-stratified BigEarthNet subset."""

import argparse
import json
import math
import random
import sys

import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--spectralgpt-root", type=Path, default=Path.home() / "spectralgpt")
parser.add_argument("--fraction", type=float, default=.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

root = args.spectralgpt_root.expanduser().resolve()
sys.path.insert(0, str(root))
from util.datasets import GROUP_LABELS, NEW_LABELS

train = root / "txt_file/bigearthnet_train.txt"
data = root / "data/BE_cor"
if not train.is_file() or not data.is_dir() or not 0 < args.fraction < 1:
    raise SystemExit("Missing official train list/BE_cor data, or invalid --fraction")
patches = train.read_text().splitlines()
target_size = round(len(patches) * args.fraction)


def labelset(patch):
    labels = json.loads((data / patch / f"{patch}_labels_metadata.json").read_text())["labels"]
    return tuple(index for index, label in enumerate(NEW_LABELS) if label in {GROUP_LABELS.get(x, x) for x in labels})

with ProcessPoolExecutor(max_workers=args.workers) as pool:
    labelsets = list(pool.map(labelset, patches, chunksize=256))

groups = defaultdict(list)
for patch, labels in zip(patches, labelsets):
    groups[labels].append(patch)
rng = random.Random(args.seed)
quotas = {labels: math.floor(len(group) * args.fraction) for labels, group in groups.items()}
remaining = target_size - sum(quotas.values())
for labels in sorted(groups, key=lambda key: (-(len(groups[key]) * args.fraction % 1), rng.random()))[:remaining]:
    quotas[labels] += 1

selected = set()
for labels, group in groups.items():
    rng.shuffle(group)
    selected.update(group[:quotas[labels]])
assert len(selected) == target_size

# Preserve each complete label combination first, then make only beneficial swaps
# to correct the marginal counts of rare labels split across tiny combinations.
y = np.zeros((len(patches), len(NEW_LABELS)), dtype=np.int8)
for index, labels in enumerate(labelsets):
    y[index, list(labels)] = 1
selected_mask = np.array([patch in selected for patch in patches])
target_counts = np.rint(y.sum(axis=0) * args.fraction).astype(int)
current_counts = y[selected_mask].sum(axis=0)
rng_np = np.random.default_rng(args.seed)
for _ in range(20_000):
    current_error = np.abs(current_counts - target_counts).sum()
    improved = False
    for label in np.argsort(target_counts - current_counts)[::-1]:
        if current_counts[label] >= target_counts[label]:
            break
        additions = np.flatnonzero(~selected_mask & (y[:, label] == 1))
        removals = np.flatnonzero(selected_mask)
        if not len(additions) or not len(removals):
            continue
        additions = rng_np.choice(additions, min(128, len(additions)), replace=False)
        removals = rng_np.choice(removals, min(512, len(removals)), replace=False)
        costs = np.abs(
            current_counts + y[additions, None, :] - y[removals][None, :, :] - target_counts
        ).sum(axis=2)
        add, remove = np.unravel_index(np.argmin(costs), costs.shape)
        if costs[add, remove] < current_error:
            selected_mask[additions[add]] = True
            selected_mask[removals[remove]] = False
            current_counts += y[additions[add]] - y[removals[remove]]
            improved = True
            break
    if not improved:
        break

subset = [patch for patch, keep in zip(patches, selected_mask) if keep]
assert len(subset) == target_size
full_counts = Counter(label for labels in labelsets for label in labels)
subset_counts = Counter(label for labels, keep in zip(labelsets, selected_mask) if keep for label in labels)
name = f"bigearthnet_train_{int(args.fraction * 100)}pct_labelset_stratified_seed{args.seed}.txt"
output = root / "txt_file" / name
output.write_text("\n".join(subset) + "\n")
report = {
    "source": str(train),
    "method": "exact label-combination stratification followed by beneficial swaps toward 10% marginal counts for all 19 labels",
    "fraction": args.fraction,
    "seed": args.seed,
    "full_images": len(patches),
    "subset_images": len(subset),
    "label_combination_groups": len(groups),
    "classes": NEW_LABELS,
    "full_images_per_class": {label: full_counts[index] for index, label in enumerate(NEW_LABELS)},
    "subset_images_per_class": {label: subset_counts[index] for index, label in enumerate(NEW_LABELS)},
}
output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(output), "subset_images": len(subset), "label_combination_groups": len(groups)}, indent=2))
