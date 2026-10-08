#!/usr/bin/env bash
# Run this only after ingestion, cleaning, dedup and augmentation.
# Both classification arms share seed, epochs and optimizer step budget.
set -euo pipefail
for ds in cifar100 fashionmnist; do
  for arm in baseline augmented; do
    visionforge train --dataset "$ds" --mode "$arm" \
      --seed 42 --epochs 30 --batch-size 128 --max-steps-per-epoch 400
  done
done
visionforge report --readme README.md
