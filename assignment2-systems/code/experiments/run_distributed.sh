#!/usr/bin/env bash
# Two-GPU experiments (NCCL).  Usage: CUDA_VISIBLE_DEVICES=1,0 bash experiments/run_distributed.sh [light|heavy]
# Rows already present in the result files are not recomputed (the script is safe to re-run).
set -u
export WAIT_GPUS="0 1"
R=experiments/results
RT="bash experiments/retry.sh"
export TRIES=${TRIES:-2} WAIT_TIMEOUT=${WAIT_TIMEOUT:-600}
have() { [ -f "$1" ] && grep -q "$2" "$1"; }   # have <csv> <pattern>
part=${1:-light}
if [ "$part" = light ]; then
  have $R/allreduce.csv "^nccl,2," || $RT 6000 uv run python -m cs336_systems.bench.allreduce_bench --backend nccl --world-sizes 2 --out $R/allreduce.csv
  for mode in naive flat overlap; do
    have $R/ddp.csv "^medium,.*,$mode,ok" || $RT 12000 uv run python -m cs336_systems.bench.ddp_bench --size medium --mode $mode --out $R/ddp.csv
  done
  [ -f $R/trace_ddp_naive.csv ] || $RT 12000 uv run python -m cs336_systems.bench.ddp_bench --size medium --mode naive --warmup 3 --steps 3 --trace $R/trace_ddp_naive.csv
  [ -f $R/trace_ddp_overlap.csv ] || $RT 12000 uv run python -m cs336_systems.bench.ddp_bench --size medium --mode overlap --warmup 3 --steps 3 --trace $R/trace_ddp_overlap.csv
  have $R/sharded.csv "^medium,24,0," || $RT 12000 uv run python -m cs336_systems.bench.sharded_bench --size medium --out $R/sharded.csv
  have $R/sharded.csv "^medium,24,1," || $RT 12000 uv run python -m cs336_systems.bench.sharded_bench --size medium --sharded --out $R/sharded.csv
  # FSDP on half of the xl model (fits comfortably), with and without prefetching, bf16 and fp32
  have $R/fsdp.csv "^xl,16,512,4,1,1," || $RT 18000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --num-layers 16 --bf16 --out $R/fsdp.csv --trace $R/trace_fsdp.csv
  have $R/fsdp.csv "^xl,16,512,4,1,0," || $RT 18000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --num-layers 16 --bf16 --no-prefetch --out $R/fsdp.csv
  have $R/fsdp.csv "^xl,16,512,4,0,1," || $RT 22000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --num-layers 16 --out $R/fsdp.csv
else
  for mode in naive flat overlap; do
    have $R/ddp.csv "^large,.*,$mode,ok" || $RT 24000 uv run python -m cs336_systems.bench.ddp_bench --size large --mode $mode --out $R/ddp.csv
  done
  have $R/sharded.csv "^large,36,0," || $RT 24000 uv run python -m cs336_systems.bench.sharded_bench --size large --out $R/sharded.csv
  have $R/sharded.csv "^large,36,1," || $RT 24000 uv run python -m cs336_systems.bench.sharded_bench --size large --sharded --out $R/sharded.csv
  # the full xl model: 12.5 GiB of fp32 weights; unsharded training would need > 50 GiB per GPU
  have $R/fsdp.csv "^xl,32,512,4,1,1," || $RT 36000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --bf16 --out $R/fsdp.csv --trace $R/trace_fsdp.csv
  have $R/fsdp.csv "^xl,32,512,4,1,0," || $RT 36000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --bf16 --no-prefetch --out $R/fsdp.csv
  [ -f $R/trace_fsdp.csv ] || $RT 18000 uv run python -m cs336_systems.bench.fsdp_bench --size xl --num-layers 16 --bf16 --warmup 2 --steps 2 --instrumented-steps 1 --trace $R/trace_fsdp.csv
fi
echo DISTRIBUTED_${part}_DONE
