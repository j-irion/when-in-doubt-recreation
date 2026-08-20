#!/usr/bin/env bash
set -euo pipefail

output=${1:?Usage: ./prepare_margin_cache.sh artifacts/imagenet1k-margin-4}
mkdir -p "$output"
ln -f artifacts/imagenet1k/teacher_train.pt "$output/teacher_train.pt"
ln -f artifacts/imagenet1k/teacher_val.pt "$output/teacher_val.pt"
