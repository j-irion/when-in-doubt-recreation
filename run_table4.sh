#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run() {
  printf '\n=== %s ===\n' "$2"
  bash efficientnet_l2/prepare_teacher_cache.sh "$1"
  uv run python "$2"
}

run artifacts/imagenet1k-baseline-head300-30e-cosine-systematic300 efficientnet_l2/table4_baseline.py
run artifacts/imagenet1k-cdi-head300-alpha00-30e-cosine-systematic300 efficientnet_l2/table4_cd_i_alpha_00.py
run artifacts/imagenet1k-cdi-head300-alpha04-30e-cosine-systematic300 efficientnet_l2/table4_cd_i_alpha_04.py
run artifacts/imagenet1k-cdi-head300-alpha06-30e-cosine-systematic300 efficientnet_l2/table4_cd_i_alpha_06.py
run artifacts/imagenet1k-cdiii-head300-30e-cosine-systematic300 efficientnet_l2/table4_cd_iii.py
