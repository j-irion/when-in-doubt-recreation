#!/usr/bin/env bash
set -euo pipefail

output=${1:?Usage: bash efficientnet_l2/prepare_teacher_cache.sh artifacts/imagenet1k-margin-4}
cache=artifacts/teacher-cache-efficientnet-l2-475-crop936-bicubic

if [[ ! -f "$cache/manifest.json" || ! -f "$cache/teacher_train.pt" || ! -f "$cache/teacher_val.pt" ]]; then
  uv run python efficientnet_l2/build_teacher_cache.py
fi

mkdir -p "$output"
ln -f "$cache/teacher_train.pt" "$output/teacher_train.pt"
ln -f "$cache/teacher_val.pt" "$output/teacher_val.pt"
ln -f "$cache/manifest.json" "$output/teacher_cache.json"
