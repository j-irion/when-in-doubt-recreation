# ImageNet-1k two-stage distillation

Prepare ILSVRC-2012 at:

```text
/workspace/julius/data/imagenet1k/train/<synset>/*.JPEG
/workspace/julius/data/imagenet1k/val/<synset>/*.JPEG
```

Run commands from the repository root.

## EfficientNet-L2 teacher

```bash
uv run python efficientnet_l2/class.py
./efficientnet_l2/prepare_cache.sh artifacts/imagenet1k-margin-4
uv run python efficientnet_l2/margin_4.py
./efficientnet_l2/prepare_cache.sh artifacts/imagenet1k-margin-6
uv run python efficientnet_l2/margin_6.py
```

`margin_8.py` is the earlier exploratory run, not a Figure 7 setting.

## SigLIP So400M teacher

```bash
uv run python siglip_so400m/class.py
./siglip_so400m/prepare_cache.sh artifacts/imagenet1k-siglip-margin-4
uv run python siglip_so400m/margin_4.py
./siglip_so400m/prepare_cache.sh artifacts/imagenet1k-siglip-margin-6
uv run python siglip_so400m/margin_6.py
```

Each model writes separate artifacts under `artifacts/`.

## Plots

Only per-class accuracy and the Figure 7 recreation are retained:

```bash
uv run --with matplotlib python plot_metrics.py \
  artifacts/imagenet1k-siglip-class/metrics.json \
  plots/siglip_so400m/class

uv run --with matplotlib python plot_figure7.py \
  artifacts/imagenet1k-siglip-margin-4/metrics.json \
  artifacts/imagenet1k-siglip-margin-6/metrics.json \
  plots/siglip_so400m/margin_4_6/figure7_margin.png
```
