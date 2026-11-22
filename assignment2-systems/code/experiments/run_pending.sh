#!/usr/bin/env bash
# The few measurements that could not be taken while both usable GPUs were occupied by other jobs.
# Restart-safe: each item is skipped once its result exists.  Needs ONE card with >= 40 GB free
# (it waits up to WAIT_TIMEOUT seconds per item for that, then runs anyway).  ~45 minutes in total.
#
#   cd assignment2-systems && WAIT_TIMEOUT=3600 bash experiments/run_pending.sh
#   uv run python writeup/make_tables.py && uv run --no-project --with typst python writeup/compile.py
set -u
R=experiments/results
G="bash experiments/on_gpu.sh"
export TRIES=${TRIES:-2} WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
BB="uv run python -m cs336_systems.bench.benchmark"
have() { [ -f "$1" ] && grep -q "$2" "$1"; }

# 1. attention sweep on a free card (the shared-card run is kept as attention_shared.csv)
if [ ! -f $R/attention_shared.csv ]; then
  mv $R/attention.csv $R/attention_shared.csv
  $G 40000 uv run python -m cs336_systems.bench.attention_bench --out $R/attention.csv --budget-s 6 || mv $R/attention_shared.csv $R/attention.csv
fi
# 2. checkpointing: 2 blocks per checkpoint, fp32, 8-layer xl
have $R/checkpointing.csv "^xl,8,2048,4,2,0,ok" || $G 40000 uv run python -m cs336_systems.bench.checkpoint_bench --size xl --context-length 2048 --num-layers 8 --segment 2 --out $R/checkpointing.csv --retry-on-oom
# 3. memory profiling: xl forward at ctx 2048 in bf16, and the 3-layer training points
have $R/memory.csv "^xl_full,xl,32,2048,4,forward,2,2,1,0,ok" || $G 40000 $BB --warmup 2 --steps 2 --size xl --context-length 2048 --mode forward --bf16 --out $R/memory.csv --tag xl_full --retry-on-oom
have $R/memory.csv "^xl_reduced,xl,3,2048,4,train,2,2,0,0,ok" || $G 40000 $BB --warmup 2 --steps 2 --size xl --num-layers 3 --context-length 2048 --mode train --out $R/memory.csv --tag xl_reduced --retry-on-oom
have $R/memory.csv "^xl_reduced,xl,3,2048,4,train,2,2,1,0,ok" || $G 40000 $BB --warmup 2 --steps 2 --size xl --num-layers 3 --context-length 2048 --mode train --bf16 --out $R/memory.csv --tag xl_reduced --retry-on-oom
grep -q "backward block" $R/residuals_xl_ctx2048.out || { $G 40000 uv run python -m cs336_systems.bench.residuals --size xl --num-layers 2 --context-length 2048 > $R/residuals_xl_ctx2048.new 2>&1 && grep -v "Warn\|warn\|on_gpu" $R/residuals_xl_ctx2048.new > $R/residuals_xl_ctx2048.out; rm -f $R/residuals_xl_ctx2048.new; }
# 4. torch.compile on the large model's training step
have $R/e2e_compile.csv "^compiled,large,36,512,4,train,5,10,0,1,ok" || $G 40000 $BB --size large --mode train --compile --out $R/e2e_compile.csv --tag compiled --retry-on-oom
# 5. nsys: large model, context 1024, training step
if ! grep -q "^\[forward\]" $R/nsys_large_ctx1024_train.summary; then
  NSYS=${NSYS:-$HOME/.local/opt/nsys/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys}
  D=experiments/nsys; name=large_ctx1024_train; mkdir -p $D
  $G 44000 $NSYS profile -o $D/$name --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    $BB --size large --context-length 1024 --mode train --nvtx --warmup 3 --steps 5 --retry-on-oom --out $R/nsys_python_timings.csv --tag $name > $D/$name.log 2>&1 \
  && $NSYS export --type sqlite --force-overwrite=true -o $D/$name.sqlite $D/$name.nsys-rep > /dev/null 2>&1 \
  && uv run python -m cs336_systems.bench.nsys_analysis $D/$name.sqlite --label $name > $R/nsys_$name.summary 2>&1
  rm -f $D/$name.sqlite
fi
echo PENDING_DONE
