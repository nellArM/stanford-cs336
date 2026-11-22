"""Train the fastText quality classifier.

Positives ("wiki"): pages referenced from English Wikipedia, scraped with wget
(scripts/scrape_positive_urls.sh).  Negatives ("cc"): random Common Crawl pages from the
example WARC file.  Both go through the same text extraction and the same language / Gopher
filters, so the classifier has to separate them by content rather than by extraction artefacts.

    uv run python -m cs336_data.train_quality_classifier
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import random
from pathlib import Path

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher import gopher_quality_filter
from cs336_data.language_identification import identify_language
from cs336_data.paths import EXAMPLE_WARC, QUALITY_DIR, QUALITY_MODEL_PATH, WIKI_DIR
from cs336_data.quality_classifier import preprocess_for_quality


def extract_examples(warc_path: str) -> list[str]:
    """Cleaned one-line texts of the English, Gopher-passing HTML pages in one WARC file."""
    examples: list[str] = []
    try:
        with open(warc_path, "rb") as stream:
            for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
                headers = record.http_headers
                if headers is None or headers.status_code != 200:
                    continue
                if "html" not in (headers.get("Content-Type") or "").lower():
                    continue
                text = extract_text_from_html_bytes(record.reader.read())
                if not text:
                    continue
                language, score = identify_language(text)
                if language != "en" or score < 0.7 or not gopher_quality_filter(text):
                    continue
                examples.append(preprocess_for_quality(text))
    except Exception as error:  # an interrupted wget leaves a truncated final gzip member
        print(f"{warc_path}: stopped early ({type(error).__name__})")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    positive_warcs = sorted(glob.glob(str(WIKI_DIR / "scrape*" / "*.warc.gz")))
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        positives = [ex for exs in executor.map(extract_examples, positive_warcs) for ex in exs]
    positives = sorted(set(positives))
    negatives = sorted(set(extract_examples(str(EXAMPLE_WARC))))
    print(f"{len(positives)} positive and {len(negatives)} negative examples after filtering")

    rng.shuffle(positives)
    rng.shuffle(negatives)
    n = min(len(positives), len(negatives))  # balanced classes
    data = [("wiki", t) for t in positives[:n]] + [("cc", t) for t in negatives[:n]]
    rng.shuffle(data)
    n_valid = int(len(data) * args.valid_fraction)
    splits = {"valid": data[:n_valid], "train": data[n_valid:]}
    for name, rows in splits.items():
        with open(QUALITY_DIR / f"quality.{name}", "w", encoding="utf-8") as f:
            for label, text in rows:
                f.write(f"__label__{label} {text}\n")

    model = fasttext.train_supervised(
        input=str(QUALITY_DIR / "quality.train"),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        dim=100,
        minCount=3,
        bucket=500_000,
        loss="softmax",
        thread=32,
        seed=args.seed,
    )
    n_examples, precision, recall = model.test(str(QUALITY_DIR / "quality.valid"))
    per_label = model.test_label(str(QUALITY_DIR / "quality.valid"))
    # Quantize so that the model is small enough to ship with the code.
    model.quantize(input=str(QUALITY_DIR / "quality.train"), retrain=True, cutoff=200_000, qnorm=True)
    _, q_precision, _ = model.test(str(QUALITY_DIR / "quality.valid"))
    QUALITY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(QUALITY_MODEL_PATH))

    report = {
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "train_examples": len(splits["train"]),
        "valid_examples": n_examples,
        "valid_accuracy": precision,
        "valid_accuracy_quantized": q_precision,
        "per_label": {k.removeprefix("__label__"): v for k, v in per_label.items()},
        "model_bytes": QUALITY_MODEL_PATH.stat().st_size,
    }
    (QUALITY_DIR / "quality_report.stats").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
