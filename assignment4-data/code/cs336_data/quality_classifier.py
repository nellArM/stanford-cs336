"""fastText quality classifier: Wikipedia-referenced pages ("wiki") vs. random Common Crawl pages ("cc")."""

from __future__ import annotations

import regex as re

from cs336_data.fasttext_utils import predict_top
from cs336_data.paths import QUALITY_MODEL_PATH

_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_for_quality(text: str, max_chars: int = 20_000) -> str:
    """Single-line, whitespace-normalized text (fastText examples are one line each)."""
    # Lower-cased, list bullets removed (they only reflect which HTML extractor was used).
    return _WHITESPACE_RE.sub(" ", text[:max_chars].replace("\u2022", " ")).strip().lower()


def classify_quality(text: str) -> tuple[str, float]:
    """Return ("wiki" | "cc", confidence in that label)."""
    return predict_top(QUALITY_MODEL_PATH, preprocess_for_quality(text))


def quality_score(text: str) -> float:
    """P(high quality) in [0, 1]."""
    label, score = classify_quality(text)
    return score if label == "wiki" else 1.0 - score
