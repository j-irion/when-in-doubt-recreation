from pathlib import Path

import matplotlib.pyplot as plt


DATA = Path("/workspace/julius/data/imagenet1k")
OUTPUT = Path("plots/dataset/class_count_distribution.png")


counts = {}
for split in ("train", "val"):
    counts[split] = [
        sum(path.is_file() for path in folder.iterdir())
        for folder in (DATA / split).iterdir()
        if folder.is_dir()
    ]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, split in zip(axes, counts):
    axis.hist(counts[split], bins="auto")
    axis.set(
        title=f"ImageNet-1k {split}",
        xlabel="images per class",
        ylabel="number of classes",
    )
    axis.grid()
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200)
print(f"wrote {OUTPUT}")
