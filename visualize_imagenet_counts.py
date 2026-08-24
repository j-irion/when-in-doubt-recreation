from pathlib import Path

import matplotlib.pyplot as plt


DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = Path("plots/dataset/class_count_distribution.png")


counts = {}
for split in ("train", "val"):
    folders = sorted(folder for folder in (DATA / split).iterdir() if folder.is_dir())
    counts[split] = sorted(sum(path.is_file() for path in folder.iterdir()) for folder in folders)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for axis, split in zip(axes, counts):
    axis.plot(range(len(counts[split])), counts[split], ".", markersize=3)
    axis.set(title=f"ImageNet-1k {split}", ylabel="images per class")
    axis.grid()
axes[-1].set(xlabel="class rank: fewest to most images")
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200)
print(f"wrote {OUTPUT}")
