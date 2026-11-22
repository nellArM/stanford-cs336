"""Language identification with fastText lid.176."""

from __future__ import annotations

from cs336_data.fasttext_utils import predict_top
from cs336_data.paths import LANGID_MODEL_PATH


def identify_language(text: str) -> tuple[str, float]:
    """Return (ISO language code, confidence in [0, 1]) for the main language of ``text``."""
    return predict_top(LANGID_MODEL_PATH, text)


def is_english(text: str, threshold: float = 0.7) -> bool:
    language, score = identify_language(text)
    return language == "en" and score >= threshold
