"""Plot epoch training loss and validation top-1 accuracy from history.json."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])


with HISTORY.open() as file:
    history = json.load(file)
epochs = [row["epoch"] for row in history]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
axes[1].plot(
    epochs,
    [row["validation_student_top1_accuracy"] for row in history],
    marker="o",
)
axes[1].set(title="Validation student top-1", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
for axis in axes:
    axis.grid()
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200)
