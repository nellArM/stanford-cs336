#!/usr/bin/env bash
# retry <MiB needed> <command...>
# The GPUs are shared with other jobs whose memory use changes from minute to minute.  Before each
# attempt this waits (up to WAIT_TIMEOUT s) until <MiB needed> is free on every GPU in WAIT_GPUS
# (physical indices, default "1"), and a failed run (e.g. a transient out-of-memory) is retried
# up to TRIES times.  Returns non-zero if every attempt failed.
need=$1; shift
for i in $(seq 1 "${TRIES:-4}"); do
  for g in ${WAIT_GPUS:-1}; do bash experiments/wait_free.sh "$g" "$need" "${WAIT_TIMEOUT:-900}"; done
  "$@" && exit 0
  echo "retry: attempt $i failed: $*"
  sleep 60
done
exit 1
