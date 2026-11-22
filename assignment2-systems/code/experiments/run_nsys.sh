#!/usr/bin/env bash
# nsys_profile: two model sizes x three context lengths, forward-only and full training step.
# Usage: CUDA_VISIBLE_DEVICES=1 bash experiments/run_nsys.sh
set -u
NSYS=${NSYS:-$HOME/.local/opt/nsys/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys}
D=experiments/nsys
mkdir -p $D
profile_one() {
  name=$1; shift
  rm -f $D/$name.nsys-rep
  $NSYS profile -o $D/$name --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    uv run python -m cs336_systems.bench.benchmark "$@" --nvtx --warmup 3 --steps 5 --retry-on-oom \
    --out experiments/results/nsys_python_timings.csv --tag $name > $D/$name.log 2>&1
  grep -q "'status': 'ok'" $D/$name.log
}
for size in small large; do
  for ctx in 256 512 1024; do
    for mode in forward train; do
      name=${size}_ctx${ctx}_${mode}
      need=12000; [ $size = large ] && need=30000; [ $size = large ] && [ $ctx = 1024 ] && need=40000
      ok=0
      for attempt in 1 2 3 4; do
        bash experiments/wait_free.sh ${GPU_ID:-1} $need 1500
        if profile_one $name --size $size --context-length $ctx --mode $mode; then ok=1; break; fi
        sleep 60
      done
      if [ $ok = 1 ]; then
        $NSYS export --type sqlite --force-overwrite=true -o $D/$name.sqlite $D/$name.nsys-rep > /dev/null 2>&1
        uv run python -m cs336_systems.bench.nsys_analysis $D/$name.sqlite --label $name --csv experiments/results/nsys_summary.csv > experiments/results/nsys_$name.summary 2>&1
        rm -f $D/$name.sqlite
      else
        echo "nsys: $name did not fit (OOM)" | tee experiments/results/nsys_$name.summary
      fi
    done
  done
done
