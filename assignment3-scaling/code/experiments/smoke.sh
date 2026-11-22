#!/usr/bin/env bash
# Smoke test: short runs of three family members to measure throughput (GPU 1 only).
cd "$(dirname "$0")/.." && export CUDA_VISIBLE_DEVICES=1
for cfg in "256 4 8e6 32" "512 8 5e6 16" "768 12 4e6 16" "768 12 8e6 64"; do
  set -- $cfg
  uv run --extra study python -m cs336_scaling.local_study.train --name smoke_d$1L$2_b$4 --group smoke \
    --d-model $1 --n-layer $2 --tokens $3 --batch-size $4 --n-evals 4 --lr 1e-3 > experiments/logs/smoke_d$1L$2_b$4.log 2>&1
done
