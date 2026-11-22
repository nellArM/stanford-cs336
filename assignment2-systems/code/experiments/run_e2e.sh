#!/usr/bin/env bash
# End-to-end timing sweeps: benchmarking_script (b, c), benchmarking_mixed_precision (c), torch_compile (b).
# Usage: CUDA_VISIBLE_DEVICES=1 bash experiments/run_e2e.sh
set -u
R=experiments/results
B="uv run python -m cs336_systems.bench.benchmark"
# fit <MiB> <args...>: a configuration that fits on an empty 48 GB card -> retried if a neighbour's job caused an OOM
fit() { need=$1; shift; bash experiments/retry.sh $need $B "$@" --retry-on-oom || $B "$@"; }
declare -A NEED=( [small]=8000 [medium]=17000 [large]=32000 )
for size in small medium large; do
  for mode in forward train; do
    fit ${NEED[$size]} --size $size --mode $mode --out $R/e2e_fp32.csv --tag fp32
    fit ${NEED[$size]} --size $size --mode $mode --bf16 --out $R/e2e_bf16.csv --tag bf16
  done
done
# xl: the forward pass fits; a training step needs > 50 GiB for weights + gradients + AdamW state alone.
fit 24000 --size xl --mode forward --out $R/e2e_fp32.csv --tag fp32
fit 24000 --size xl --mode forward --bf16 --out $R/e2e_bf16.csv --tag bf16
$B --size xl --mode train --out $R/e2e_fp32.csv --tag fp32
$B --size xl --mode train --bf16 --out $R/e2e_bf16.csv --tag bf16
# 10B: 12.7 B parameters = 47 GiB in fp32, does not fit even for inference.
for mode in forward train; do
  $B --size 10B --mode $mode --out $R/e2e_fp32.csv --tag fp32
  $B --size 10B --mode $mode --bf16 --out $R/e2e_bf16.csv --tag bf16
done
for warmup in 0 1 2 5; do
  for size in small medium; do
    fit ${NEED[$size]} --size $size --mode train --warmup $warmup --out $R/e2e_warmup.csv --tag warmup$warmup
  done
done
for size in small medium large; do
  for mode in forward train; do
    fit ${NEED[$size]} --size $size --mode $mode --out $R/e2e_compile.csv --tag eager
    fit ${NEED[$size]} --size $size --mode $mode --compile --out $R/e2e_compile.csv --tag compiled
  done
done
