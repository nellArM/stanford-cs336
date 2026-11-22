"""Run every filtering primitive over the text extracted from one WARC file.

Writes one JSON line per HTML response with all classifier outputs, plus summary
statistics.  The written answers (language-ID, PII, harmful-content and Gopher audits)
are based on random samples drawn from this file.

    uv run python -m cs336_data.audit_samples
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher import gopher_rejection_reason
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.paths import DATA_ROOT, EXAMPLE_WARC
from cs336_data.pii import mask_pii


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc", default=str(EXAMPLE_WARC))
    parser.add_argument("--output", default=str(DATA_ROOT / "audit" / "example_warc_records.jsonl"))
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    with open(args.warc, "rb") as stream, open(args.output, "w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
            content_type = (record.http_headers.get("Content-Type") or "") if record.http_headers else ""
            if "html" not in content_type.lower():
                stats["non_html"] += 1
                continue
            text = extract_text_from_html_bytes(record.reader.read())
            if not text or not text.strip():
                stats["empty_text"] += 1
                continue
            language, language_score = identify_language(text)
            nsfw_label, nsfw_score = classify_nsfw(text)
            toxic_label, toxic_score = classify_toxic_speech(text)
            masked, pii_counts = mask_pii(text)
            gopher_reason = gopher_rejection_reason(text)
            stats["records"] += 1
            stats[f"lang_{language}"] += 1
            stats["nsfw"] += nsfw_label == "nsfw"
            stats["toxic"] += toxic_label == "toxic"
            stats["has_pii"] += sum(pii_counts.values()) > 0
            stats["gopher_pass"] += gopher_reason is None
            out.write(
                json.dumps(
                    {
                        "url": record.headers.get("WARC-Target-URI"),
                        "text": text,
                        "masked_text": masked if sum(pii_counts.values()) else None,
                        "language": language,
                        "language_score": language_score,
                        "nsfw": [nsfw_label, nsfw_score],
                        "toxic": [toxic_label, toxic_score],
                        "pii": pii_counts,
                        "gopher_reason": gopher_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if args.max_records and stats["records"] >= args.max_records:
                break
    stats["seconds"] = round(time.perf_counter() - started)
    summary_path = args.output.replace(".jsonl", ".summary")
    with open(summary_path, "w") as f:
        json.dump(dict(stats.most_common()), f, indent=1)
    print(json.dumps(dict(stats.most_common(25)), indent=1))


if __name__ == "__main__":
    main()
