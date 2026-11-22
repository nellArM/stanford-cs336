"""Local, parallel stand-in for the staff's ``EnglishWetFiles`` job (cs336_data/wet_files.py).

Samples raw WET files from a Common Crawl crawl with the staff's seed, groups them
``group_size`` at a time, keeps only the ``conversion`` records whose English probability
(fastText lid.176) is >= 0.7 and writes one gzipped WET file per group.  Raw files are
deleted as soon as they have been processed, so disk usage stays bounded.

    uv run python -m cs336_data.make_english_wets --n-files 400 --workers 24
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import polars as pl
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.language_identification import is_english
from cs336_data.paths import DATA_ROOT, ENGLISH_WET_DIR

BASE_URL = "https://data.commoncrawl.org/"


def _download(url: str, dest: Path, retries: int = 8) -> None:
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # Common Crawl answers 503 "SlowDown" when it is busy; back off and retry.
            dest.unlink(missing_ok=True)
            time.sleep(min(120, 5 * 2**attempt))
    raise RuntimeError(f"failed to download {url}")


def process_chunk(chunk_idx: int, chunk_urls: tuple[str, ...], out_dir: str, tmp_dir: str) -> dict:
    out_path = Path(out_dir) / f"{chunk_idx:05d}-data.warc.wet.gz"
    stats_path = out_path.with_suffix(".stats")
    if out_path.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())

    started = time.perf_counter()
    stats = {"chunk": chunk_idx, "urls": list(chunk_urls), "records": 0, "kept_records": 0, "chars": 0,
             "kept_chars": 0, "raw_bytes": 0, "download_seconds": 0.0}
    tmp_out = Path(tmp_dir) / f"{chunk_idx:05d}.partial.gz"
    with gzip.open(tmp_out, "wb", compresslevel=5) as output_stream:
        writer = WARCWriter(output_stream, gzip=False)
        for wet_url in chunk_urls:
            raw_path = Path(tmp_dir) / wet_url.split("/")[-1]
            t0 = time.perf_counter()
            _download(wet_url, raw_path)
            stats["download_seconds"] += time.perf_counter() - t0
            stats["raw_bytes"] += raw_path.stat().st_size
            with gzip.open(raw_path, "rb") as input_stream:
                for rec in ArchiveIterator(input_stream):
                    if rec.rec_type != "conversion":
                        writer.write_record(rec)
                        continue
                    payload = rec.content_stream().read()
                    text = payload.decode("utf-8", errors="replace")
                    stats["records"] += 1
                    stats["chars"] += len(text)
                    if is_english(text, threshold=0.7):
                        rec.raw_stream = BytesIO(payload)
                        writer.write_record(rec)
                        stats["kept_records"] += 1
                        stats["kept_chars"] += len(text)
            raw_path.unlink(missing_ok=True)
    tmp_out.rename(out_path)
    stats["seconds"] = time.perf_counter() - started
    stats_path.write_text(json.dumps(stats))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=400)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=336)
    parser.add_argument("--crawl-id", default="CC-MAIN-2026-17")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    assert args.n_files % args.group_size == 0

    wet_paths = pl.read_csv(
        f"{BASE_URL}crawl-data/{args.crawl_id}/wet.paths.gz", has_header=False, new_columns=["wet_path"]
    )
    urls = list(
        BASE_URL
        + wet_paths.sample(n=args.n_files, shuffle=True, seed=args.shuffle_seed, with_replacement=False)["wet_path"]
    )
    chunks = [tuple(urls[i : i + args.group_size]) for i in range(0, len(urls), args.group_size)]

    ENGLISH_WET_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_ROOT / "tmp-raw-wet"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    all_stats = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_chunk, idx, chunk, str(ENGLISH_WET_DIR), str(tmp_dir))
            for idx, chunk in enumerate(chunks)
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            all_stats.append(future.result())

    summary = {
        "crawl_id": args.crawl_id,
        "n_raw_files": args.n_files,
        "n_output_files": len(chunks),
        "wall_seconds": time.perf_counter() - started,
        "records": sum(s["records"] for s in all_stats),
        "kept_records": sum(s["kept_records"] for s in all_stats),
        "chars": sum(s["chars"] for s in all_stats),
        "kept_chars": sum(s["kept_chars"] for s in all_stats),
        "raw_bytes": sum(s["raw_bytes"] for s in all_stats),
    }
    (ENGLISH_WET_DIR / "summary.stats").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
