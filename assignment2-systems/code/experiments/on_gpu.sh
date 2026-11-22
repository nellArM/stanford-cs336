#!/usr/bin/env bash
# on_gpu <MiB needed> <command...>
# Run a single-GPU command on whichever of the two cards we may use (physical GPUs 0 and 1)
# currently has the most free memory.  Both cards are shared with other jobs, so this waits up to
# WAIT_TIMEOUT seconds for <MiB needed> to be free, then runs regardless; a failing command is
# retried up to TRIES times.  Returns non-zero if every attempt failed.
need=$1; shift
for i in $(seq 1 "${TRIES:-2}"); do
  waited=0
  while true; do
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 0,1 | sort -t, -k2 -n -r | head -1)
    gpu=${best%%,*}; free=${best##*, }
    if [ "$free" -ge "$need" ] || [ "$waited" -ge "${WAIT_TIMEOUT:-300}" ]; then break; fi
    sleep 15; waited=$((waited + 15))
  done
  echo "on_gpu: GPU $gpu (free ${free} MiB, wanted ${need}): $*"
  CUDA_VISIBLE_DEVICES=$gpu "$@" && exit 0
  echo "on_gpu: attempt $i failed"
  sleep 20
done
exit 1
