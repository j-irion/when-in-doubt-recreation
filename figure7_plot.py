"""Plot the rho_train=0.4 and 0.6 margin-distillation runs like Figure 7."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


if len(sys.argv) == 1:
    RUNS = [
        Path("metrics/efficientnet_l2/margin_4/metrics.json"),
        Path("metrics/efficientnet_l2/margin_6/metrics.json"),
    ]
    OUTPUT = Path("plots/efficientnet_l2/figure7/figure7_margin.png")
elif len(sys.argv) == 4:
    RUNS = [Path(sys.argv[1]), Path(sys.argv[2])]
    OUTPUT = Path(sys.argv[3])
else:
    raise SystemExit("Usage: python figure7_plot.py metrics_4.json metrics_6.json output.png")


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for path in RUNS:
    with path.open() as file:
        metrics = json.load(file)
    curve = sorted(metrics["margin_delegation"], key=lambda row: row["overall_student_fraction"])[::10]
    label = f"MD ($\\rho_{{train}}={metrics['run']['rho_train']}$)"
    fraction = [row["overall_student_fraction"] for row in curve]
    axes[0].plot(fraction, [row["overall_accuracy"] for row in curve], marker="x", label=label)
    axes[1].plot(fraction, [row["in_domain_accuracy"] for row in curve], marker="x", label=label)

for axis, title in zip(axes, ["(a) Overall accuracy", "(b) In-domain accuracy"]):
    axis.set(title=title, xlabel="Fraction processed by student", ylabel="Top-1 accuracy")
    axis.legend(loc="lower left")
    axis.grid()
fig.suptitle("In-domain: teacher margin ≥ 0.4")
fig.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200)
