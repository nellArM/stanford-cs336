"""Parallel filtering pipeline: English-prefiltered Common Crawl WET files -> LM training data.

Stage 1 (per WET file, parallel): document-level filters, applied in this order, with the
    number of documents removed by each one recorded:
        language (English >= 0.80) -> Gopher rules -> NSFW / toxic classifiers ->
        C4-style line cleaning + page rules -> quality classifier -> PII masking (modifies)
Stage 2 (global): exact line deduplication.  Lines whose hash occurs more than once in the
    whole corpus are dropped; documents that become too short are dropped.
Stage 3 (global): MinHash + LSH fuzzy document deduplication; candidates are confirmed with
    the true word-5-gram Jaccard similarity and one document per cluster is kept.
Stage 4 (per shard, parallel): GPT-2 tokenization with <|endoftext|> after every document,
    serialized as a flat uint16 array.

    uv run python -m cs336_data.filter_pipeline --workers 48
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import random
import time
from collections import Counter
from pathlib import Path

import mmh3
import numpy as np
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from tqdm import tqdm

from cs336_data import c4_rules
from cs336_data.deduplication import jaccard, minhash_signature, word_ngrams
from cs336_data.gopher import gopher_rejection_reason
from cs336_data.harmful_content import harmful_probability
from cs336_data.language_identification import identify_language
from cs336_data.paths import ENGLISH_WET_DIR, FILTERED_DIR
from cs336_data.pii import mask_pii
from cs336_data.quality_classifier import quality_score

LANGUAGE_THRESHOLD = 0.80
NSFW_THRESHOLD = 0.5
TOXIC_THRESHOLD = 0.5
QUALITY_THRESHOLD = 0.05  # calibrated on Paloma: keeps 87% of validation-like documents, see writeup
MIN_WORDS = 50
NUM_HASHES = 128
NUM_BANDS = 16
NGRAM = 5
JACCARD_THRESHOLD = 0.8
REMOVED_SAMPLES_PER_REASON = 3  # per input file, kept for manual inspection

_TLD = TLDExtract(suffix_list_urls=())  # bundled public-suffix snapshot, no network access


def _read_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------
def filter_document(text: str) -> tuple[str | None, str, dict[str, int]]:
    """Returns (final text or None, name of the step that removed it or "kept", pii counts)."""
    language, score = identify_language(text)
    if language != "en" or score < LANGUAGE_THRESHOLD:
        return None, "language", {}
    if gopher_rejection_reason(text) is not None:
        return None, "gopher", {}
    p_nsfw, p_toxic = harmful_probability(text)
    if p_nsfw >= NSFW_THRESHOLD:
        return None, "nsfw", {}
    if p_toxic >= TOXIC_THRESHOLD:
        return None, "toxic", {}
    cleaned = c4_rules.clean_lines(text)
    if len(cleaned.split()) < MIN_WORDS or c4_rules.page_rejection_reason(cleaned) is not None:
        return None, "c4_rules", {}
    if quality_score(cleaned) < QUALITY_THRESHOLD:
        return None, "quality_classifier", {}
    masked, pii_counts = mask_pii(cleaned)
    return masked, "kept", pii_counts


def stage1_file(wet_path: str, out_dir: str) -> dict:
    out_path = Path(out_dir) / (Path(wet_path).name.split("-")[0] + ".jsonl.gz")
    stats_path = out_path.with_suffix(".stats")
    if out_path.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())

    started = time.perf_counter()
    counts: Counter[str] = Counter()
    removed_samples: dict[str, list[dict]] = {}
    rng = random.Random(wet_path)
    tmp_path = out_path.with_suffix(".partial")
    with open(wet_path, "rb") as stream, gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=4) as out:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            text = record.reader.read().decode("utf-8", errors="replace")
            url = record.headers.get("WARC-Target-URI") or ""
            counts["input"] += 1
            counts["input_chars"] += len(text)
            final, step, pii_counts = filter_document(text)
            counts[step] += 1
            if final is None:
                # Reservoir sample of removed documents per reason.
                bucket = removed_samples.setdefault(step, [])
                sample = {"url": url, "reason": step, "text": text[:1500]}
                if len(bucket) < REMOVED_SAMPLES_PER_REASON:
                    bucket.append(sample)
                elif (j := rng.randrange(counts[step])) < REMOVED_SAMPLES_PER_REASON:
                    bucket[j] = sample
                continue
            if sum(pii_counts.values()):
                counts["pii_masked_docs"] += 1
                for kind, n in pii_counts.items():
                    counts[f"pii_{kind}"] += n
            counts["kept_chars"] += len(final)
            parts = _TLD(url)
            out.write(json.dumps({"url": url, "domain": f"{parts.domain}.{parts.suffix}", "text": final}) + "\n")
    tmp_path.rename(out_path)
    counts["seconds"] = round(time.perf_counter() - started, 1)
    with open(out_path.with_suffix(".removed"), "w", encoding="utf-8") as f:
        for samples in removed_samples.values():
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    stats_path.write_text(json.dumps(counts))
    return dict(counts)


# --------------------------------------------------------------------------------------
# Stage 2: exact line deduplication at corpus scale
# --------------------------------------------------------------------------------------
def _line_hash(line: str) -> int:
    return mmh3.hash64(line, signed=False)[0]


def stage2_hash_lines(path: str) -> np.ndarray:
    hashes = [_line_hash(line) for doc in _read_jsonl(Path(path)) for line in doc["text"].split("\n")]
    return np.array(hashes, dtype=np.uint64)


def stage2_rewrite(path: str, duplicate_hashes_path: str, out_dir: str) -> dict:
    """Drop duplicated lines, re-check document length, and compute MinHash signatures."""
    duplicates = np.load(duplicate_hashes_path)
    out_path = Path(out_dir) / Path(path).name
    counts: Counter[str] = Counter()
    signatures = []
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=4) as out:
        for doc in _read_jsonl(Path(path)):
            lines = doc["text"].split("\n")
            hashes = np.array([_line_hash(line) for line in lines], dtype=np.uint64)
            # Membership test against the sorted array of duplicated hashes.
            idx = np.searchsorted(duplicates, hashes)
            idx[idx == len(duplicates)] = 0
            is_duplicate = duplicates[idx] == hashes if len(duplicates) else np.zeros(len(lines), dtype=bool)
            kept = [line for line, dup in zip(lines, is_duplicate) if not dup]
            counts["lines"] += len(lines)
            counts["duplicate_lines"] += int(is_duplicate.sum())
            text = "\n".join(kept)
            if len(text.split()) < MIN_WORDS or c4_rules.count_sentences(text) < c4_rules.MIN_SENTENCES:
                counts["removed_too_short_after_line_dedup"] += 1
                continue
            counts["kept"] += 1
            doc["text"] = text
            out.write(json.dumps(doc) + "\n")
            signatures.append(minhash_signature(word_ngrams(text, NGRAM), NUM_HASHES).astype(np.uint32))
    sig = np.stack(signatures) if signatures else np.zeros((0, NUM_HASHES), dtype=np.uint32)
    np.save(out_path.with_suffix(".sig.npy"), sig)
    return dict(counts)


# --------------------------------------------------------------------------------------
# Stage 3: MinHash + LSH document deduplication at corpus scale
# --------------------------------------------------------------------------------------
def stage3_find_duplicates(stage2_dir: Path, files: list[Path], seed: int = 0) -> tuple[np.ndarray, dict]:
    """Boolean keep-mask over all stage-2 documents (in file order) plus statistics."""
    signatures = np.concatenate([np.load(f.with_suffix(".sig.npy")) for f in files])
    n = len(signatures)
    rows = NUM_HASHES // NUM_BANDS
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    # Candidate pairs: within every LSH bucket, each document is paired with the bucket's
    # first document (a star instead of all pairs, which keeps huge boilerplate buckets linear).
    candidates: set[tuple[int, int]] = set()
    for band in range(NUM_BANDS):
        block = np.ascontiguousarray(signatures[:, band * rows : (band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        same_as_previous = np.concatenate([[False], sorted_keys[1:] == sorted_keys[:-1]])
        group_start = np.maximum.accumulate(np.where(~same_as_previous, np.arange(n), 0))
        for member, first in zip(order[same_as_previous], order[group_start[same_as_previous]]):
            candidates.add((int(min(member, first)), int(max(member, first))))

    # Confirm with the true n-gram Jaccard similarity.  Only the candidate documents' texts are loaded.
    needed_set = {i for pair in candidates for i in pair}
    ngram_sets: dict[int, set[str]] = {}
    offset = 0
    for f in files:
        size = len(np.load(f.with_suffix(".sig.npy"), mmap_mode="r"))
        for local_idx, doc in enumerate(_read_jsonl(f)):
            if offset + local_idx in needed_set:
                ngram_sets[offset + local_idx] = word_ngrams(doc["text"], NGRAM)
        offset += size
    confirmed = 0
    for i, j in candidates:
        if find(i) != find(j) and jaccard(ngram_sets[i], ngram_sets[j]) >= JACCARD_THRESHOLD:
            parent[find(j)] = find(i)
            confirmed += 1

    roots = np.array([find(i) for i in range(n)])
    rng = np.random.default_rng(seed)
    keep = np.ones(n, dtype=bool)
    clustered = np.flatnonzero(np.bincount(roots, minlength=n)[roots] > 1)
    clusters: dict[int, list[int]] = {}
    for i in clustered:
        clusters.setdefault(int(roots[i]), []).append(int(i))
    for members in clusters.values():
        keep[members] = False
        keep[rng.choice(members)] = True  # one random survivor per cluster
    stats = {
        "documents": int(n),
        "candidate_pairs": len(candidates),
        "confirmed_pairs": confirmed,
        "duplicate_clusters": len(clusters),
        "removed_fuzzy_duplicates": int((~keep).sum()),
        "largest_cluster": max((len(m) for m in clusters.values()), default=0),
    }
    return keep, stats


# --------------------------------------------------------------------------------------
# Stage 4: final shards + tokenization
# --------------------------------------------------------------------------------------
def stage4_tokenize(path: str, keep_path: str, start: int, out_dir: str) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    keep = np.load(keep_path, mmap_mode="r")
    out_path = Path(out_dir) / Path(path).name
    docs = [doc for i, doc in enumerate(_read_jsonl(Path(path))) if keep[start + i]]
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=4) as out:
        for doc in docs:
            out.write(json.dumps(doc) + "\n")
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for start_idx in range(0, len(docs), 256):
        batch = tokenizer([d["text"] for d in docs[start_idx : start_idx + 256]], verbose=False)["input_ids"]
        for doc_ids in batch:
            ids.extend(doc_ids)
            ids.append(eos)
    np.array(ids, dtype=np.uint16).tofile(out_path.with_suffix(".tokens"))
    return {"documents": len(docs), "tokens": len(ids), "chars": sum(len(d["text"]) for d in docs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ENGLISH_WET_DIR))
    parser.add_argument("--output-dir", default=str(FILTERED_DIR))
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    dirs = {name: out_root / name for name in ("stage1", "stage2", "final")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    wet_files = sorted(Path(args.input_dir).glob("*.warc.wet.gz"))[: args.max_files]
    report: dict = {"input_files": len(wet_files), "workers": args.workers}
    timings: dict[str, float] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # ---- Stage 1
        t0 = time.perf_counter()
        totals: Counter[str] = Counter()
        futures = [executor.submit(stage1_file, str(f), str(dirs["stage1"])) for f in wet_files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="stage 1: filter"):
            totals.update(future.result())
        timings["stage1_filter"] = time.perf_counter() - t0
        report["stage1_cpu_seconds"] = totals.pop("seconds")
        report["stage1"] = dict(totals)

        # ---- Stage 2
        t0 = time.perf_counter()
        stage1_files = sorted(dirs["stage1"].glob("*.jsonl.gz"))
        all_hashes = np.concatenate(list(executor.map(stage2_hash_lines, map(str, stage1_files))))
        unique, counts = np.unique(all_hashes, return_counts=True)
        duplicates_path = out_root / "duplicate_line_hashes.npy"
        np.save(duplicates_path, unique[counts > 1])
        del all_hashes
        stage2: Counter[str] = Counter()
        results = executor.map(
            stage2_rewrite, map(str, stage1_files), [str(duplicates_path)] * len(stage1_files),
            [str(dirs["stage2"])] * len(stage1_files),
        )
        for result in tqdm(results, total=len(stage1_files), desc="stage 2: line dedup"):
            stage2.update(result)
        stage2["distinct_lines"] = int(len(unique))
        stage2["distinct_duplicated_lines"] = int((counts > 1).sum())
        timings["stage2_line_dedup"] = time.perf_counter() - t0
        report["stage2"] = dict(stage2)

        # ---- Stage 3
        t0 = time.perf_counter()
        stage2_files = sorted(dirs["stage2"].glob("*.jsonl.gz"))
        keep, stage3 = stage3_find_duplicates(dirs["stage2"], stage2_files)
        keep_path = out_root / "keep_mask.npy"
        np.save(keep_path, keep)
        timings["stage3_minhash_dedup"] = time.perf_counter() - t0
        report["stage3"] = stage3

        # ---- Stage 4
        t0 = time.perf_counter()
        sizes = [len(np.load(f.with_suffix(".sig.npy"), mmap_mode="r")) for f in stage2_files]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
        stage4: Counter[str] = Counter()
        results = executor.map(
            stage4_tokenize, map(str, stage2_files), [str(keep_path)] * len(stage2_files), starts,
            [str(dirs["final"])] * len(stage2_files),
        )
        for result in tqdm(results, total=len(stage2_files), desc="stage 4: tokenize"):
            stage4.update(result)
        token_path = out_root / "filtered_cc_gpt2_tokens.bin"
        with open(token_path, "wb") as out:
            for f in sorted(dirs["final"].glob("*.tokens")):
                out.write(f.read_bytes())
                f.unlink()
        timings["stage4_tokenize"] = time.perf_counter() - t0
        report["stage4"] = dict(stage4)

    report["wall_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    report["total_wall_seconds"] = round(sum(timings.values()), 1)
    report["thresholds"] = {
        "language": LANGUAGE_THRESHOLD, "nsfw": NSFW_THRESHOLD, "toxic": TOXIC_THRESHOLD,
        "quality": QUALITY_THRESHOLD, "min_words": MIN_WORDS, "num_hashes": NUM_HASHES,
        "num_bands": NUM_BANDS, "ngram": NGRAM, "jaccard": JACCARD_THRESHOLD,
    }
    (out_root / "pipeline_report.stats").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
