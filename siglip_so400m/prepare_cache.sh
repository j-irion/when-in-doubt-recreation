#!/usr/bin/env bash
set -euo pipefail

output=${1:?Usage: ./prepare_siglip_cache.sh artifacts/imagenet1k-siglip-margin-4}
source=artifacts/imagenet1k-siglip-class
mkdir -p "$output"
ln -f "$source/teacher_train.pt" "$output/teacher_train.pt"
ln -f "$source/teacher_val.pt" "$output/teacher_val.pt"
