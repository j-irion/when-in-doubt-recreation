# ImageNet-1k two-stage distillation

This repository retains only the paper-style ImageNet-1k experiments:

- Table 4: baseline, CD-I (`alpha=0.0`, `0.4`, `0.6`), and CD-III with class-based delegation.
- Figure 7: margin-based distillation with `rho_train=0.4` and `0.6`.

Prepare ILSVRC-2012 at:

```text
~/data/imagenet1k/train/<synset>/*.JPEG
~/data/imagenet1k/val/<synset>/*.JPEG
```

Run commands from the repository root. The first cache-preparation command builds a versioned EfficientNet-L2 cache in `artifacts/teacher-cache-efficientnet-l2-475-crop936-bicubic/` using the checkpoint’s declared bicubic resize-to-507 then center-crop-to-475 evaluation transform. Experiment directories hard-link that cache; the older `artifacts/imagenet1k/` logits are not used.

## Table 4: MobileNetV3-0.75

```bash
bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-baseline-head300-30e-cosine
uv run python efficientnet_l2/table4_baseline.py

bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-cdi-head300-alpha00-30e-cosine
uv run python efficientnet_l2/table4_cd_i_alpha_00.py

bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-cdi-head300-alpha04-30e-cosine
uv run python efficientnet_l2/table4_cd_i_alpha_04.py

bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-cdi-head300-alpha06-30e-cosine
uv run python efficientnet_l2/table4_cd_i_alpha_06.py

bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-cdiii-head300-30e-cosine
uv run python efficientnet_l2/table4_cd_iii.py
```

The scripts select the 300 largest ImageNet training folders, breaking ties by synset. This is a public proxy for the paper’s unreleased `L_in` list.

All Table 4 scripts use the same 30-epoch AdamW baseline recipe: initial learning rate `1e-3`, cosine decay to `1e-5`, and per-epoch loss, validation top-1, and learning-rate history. Run all five sequentially with `bash run_table4.sh`.

## Figure 7: margin-based distillation

```bash
bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-margin-4
uv run python efficientnet_l2/figure7_md_rho_train_04.py

bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-margin-6
uv run python efficientnet_l2/figure7_md_rho_train_06.py
```

Create the two-panel Figure 7 recreation after copying the two metrics files locally:

```bash
uv run --with matplotlib python figure7_plot.py \
  metrics/efficientnet_l2/margin_4/metrics.json \
  metrics/efficientnet_l2/margin_6/metrics.json \
  plots/efficientnet_l2/figure7/figure7_margin.png
```
