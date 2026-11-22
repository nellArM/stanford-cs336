#!/usr/bin/env bash
# Remaining single-GPU experiments, run on whichever of GPU 0/1 has room (see on_gpu.sh).
# Usage: bash experiments/run_remaining.sh
set -u
R=experiments/results
S=experiments/snapshots
F=experiments/figures
mkdir -p $S $F
G="bash experiments/on_gpu.sh"
BB="uv run python -m cs336_systems.bench.benchmark --warmup 2 --steps 2"
A="uv run python -m cs336_systems.bench.memory_analysis"
# fit <MiB> <benchmark args>: expected to fit on a free card -> retry on OOM, record OOM only as a last resort
fit() { need=$1; shift; $G $need $BB "$@" --retry-on-oom || $G $need $BB "$@"; }

# ---- memory_profiling -------------------------------------------------------------------
for ctx in 128 2048; do
  fit 24000 --size xl --context-length $ctx --mode forward --memory-snapshot $S/xl_ctx${ctx}_forward.pickle --out $R/memory.csv --tag xl_full
  $A $S/xl_ctx${ctx}_forward.pickle --title "xl (32 layers), ctx $ctx, forward only (fp32)" --fig $F/mem_xl_ctx${ctx}_forward.png > $R/memory_top_allocs_xl_ctx${ctx}_forward.out
  fit 24000 --size xl --context-length $ctx --mode forward --bf16 --out $R/memory.csv --tag xl_full
  # full training step: > 50 GiB of weights + gradients + AdamW state alone -> cannot fit on a 48 GB card
  TRIES=1 WAIT_TIMEOUT=0 $G 0 $BB --size xl --context-length $ctx --mode train --out $R/memory.csv --tag xl_full
  TRIES=1 WAIT_TIMEOUT=0 $G 0 $BB --size xl --context-length $ctx --mode train --bf16 --out $R/memory.csv --tag xl_full
done
for layers in 2 4 8; do
  fit 20000 --size xl --num-layers $layers --context-length 128 --mode train --out $R/memory.csv --tag xl_reduced
  fit 20000 --size xl --num-layers $layers --context-length 128 --mode train --bf16 --out $R/memory.csv --tag xl_reduced
done
for layers in 1 2 3; do
  fit 24000 --size xl --num-layers $layers --context-length 2048 --mode train --out $R/memory.csv --tag xl_reduced
  fit 24000 --size xl --num-layers $layers --context-length 2048 --mode train --bf16 --out $R/memory.csv --tag xl_reduced
done
fit 20000 --size xl --num-layers 8 --context-length 128 --mode train --memory-snapshot $S/xl8_ctx128_train.pickle --tag snapshot
$A $S/xl8_ctx128_train.pickle --title "xl (8 of 32 layers), ctx 128, full training step (fp32)" --fig $F/mem_xl8_ctx128_train.png > $R/memory_top_allocs_xl8_ctx128_train.out
fit 24000 --size xl --num-layers 2 --context-length 2048 --mode train --memory-snapshot $S/xl2_ctx2048_train.pickle --tag snapshot
$A $S/xl2_ctx2048_train.pickle --title "xl (2 of 32 layers), ctx 2048, full training step (fp32)" --fig $F/mem_xl2_ctx2048_train.png > $R/memory_top_allocs_xl2_ctx2048_train.out
$G 16000 uv run python -m cs336_systems.bench.residuals --size xl --num-layers 4 --context-length 512 > $R/residuals_xl_ctx512.out 2>&1
$G 24000 uv run python -m cs336_systems.bench.residuals --size xl --num-layers 2 --context-length 2048 > $R/residuals_xl_ctx2048.out 2>&1
rm -rf $S

# ---- gradient_checkpointing (b): xl, batch 4, ctx 2048, one level of checkpointing ---------
CC="uv run python -m cs336_systems.bench.checkpoint_bench --size xl --context-length 2048 --out $R/checkpointing.csv"
cfit() { need=$1; shift; $G $need $CC "$@" --retry-on-oom || $G $need $CC "$@"; }
for seg in 1 2 4; do cfit 24000 --num-layers 8 --segment $seg; done
cfit 24000 --num-layers 8 --segment 0
for seg in 1 2 4; do cfit 24000 --num-layers 8 --segment $seg --bf16; done
WAIT_TIMEOUT=600 cfit 40000 --num-layers 32 --segment 1
WAIT_TIMEOUT=600 cfit 40000 --num-layers 32 --segment 1 --bf16

# ---- attention / FlashAttention ------------------------------------------------------------
$G 24000 uv run python -m cs336_systems.bench.attention_bench --out $R/attention.csv --budget-s 8
$G 6000 uv run python -m cs336_systems.bench.flash_bench --tune --out $R/flash_tiles.csv
$G 24000 uv run python -m cs336_systems.bench.flash_bench --out $R/flash.csv

# ---- scaled leaderboard --------------------------------------------------------------------
L="uv run python -m cs336_systems.bench.leaderboard --out $R/leaderboard.csv"
lfit() { need=$1; shift; $G $need $L "$@" --retry-on-oom || $G $need $L "$@"; }
for v in baseline flash flash+ce flash+ce+adamw; do lfit 24000 --variant $v; done
TRIES=1 $G 24000 $L --variant baseline --ctx-len 16384 --num-layers 12
lfit 24000 --variant flash+ce+adamw --ctx-len 16384 --num-layers 12
echo REMAINING_DONE
