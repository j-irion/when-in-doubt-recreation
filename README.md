# ImageNet-1k first try

Prepare ILSVRC-2012 at:

```text
/workspace/julius/data/imagenet1k/train/<synset>/*.JPEG
/workspace/julius/data/imagenet1k/val/<synset>/*.JPEG
```

Edit the constants at the top of `imagenet1k.py`, then run:

```bash
uv sync
uv run python imagenet1k.py
```

It trains one MobileNetV3 student using the public Noisy Student EfficientNet-L2-475 teacher and writes checkpoints/results to `artifacts/imagenet1k`.
