#!/usr/bin/env bash
# wait_free <physical gpu index> <MiB needed> [timeout seconds]: block until that much memory is free.
# The GPUs are shared with other jobs, so large runs wait for room instead of failing spuriously.
gpu=$1; need=$2; timeout=${3:-1800}; waited=0
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
  if [ "$free" -ge "$need" ]; then exit 0; fi
  if [ "$waited" -ge "$timeout" ]; then echo "wait_free: timeout (free=${free} MiB, wanted ${need})"; exit 1; fi
  sleep 20; waited=$((waited + 20))
done
