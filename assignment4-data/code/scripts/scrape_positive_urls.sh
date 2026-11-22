#!/usr/bin/env bash
# Scrape the subsampled Wikipedia-reference URLs into WARC files with wget, exactly as the
# handout prescribes, but split over several wget processes (each one is sequential and
# every domain appears at most 3 times in the list, so no single host is hit hard).
#
#   bash scripts/scrape_positive_urls.sh [NUM_PROCESSES] [URL_FILE] [OUTPUT_DIR]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)/local-shared-data/wiki"
nproc="${1:-32}"
urls="${2:-$root/subsampled_positive_urls.txt}"
out="${3:-$root/scrape}"
mkdir -p "$out"
cd "$out"
split -n "l/$nproc" -d -a 3 "$urls" part_

for part in part_???; do
    (
        wget --timeout=5 --tries=1 --no-verbose \
            -i "$part" \
            --warc-file="positive_$part" \
            -O /dev/null >"$part.log" 2>&1 || true
    ) &
done
wait
ls "$out"/*.warc.gz | wc -l
