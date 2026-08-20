#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/imagenet1k-margin
ln -f artifacts/imagenet1k/teacher_train.pt artifacts/imagenet1k-margin/teacher_train.pt
ln -f artifacts/imagenet1k/teacher_val.pt artifacts/imagenet1k-margin/teacher_val.pt
