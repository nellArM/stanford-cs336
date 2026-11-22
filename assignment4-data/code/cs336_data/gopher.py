"""A subset of the Gopher (Rae et al., 2021) rule-based quality filters."""

from __future__ import annotations

MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10
MAX_ELLIPSIS_LINE_FRACTION = 0.3
MIN_ALPHABETIC_WORD_FRACTION = 0.8


def gopher_rejection_reason(text: str) -> str | None:
    """Name of the first Gopher rule that ``text`` violates, or None if it passes all of them."""
    # "Words" are whitespace-separated tokens with at least one letter or digit, so that
    # stand-alone symbols (list bullets, separators, "|") are not counted as words.
    words = [token for token in text.split() if any(ch.isalnum() for ch in token)]
    num_words = len(words)
    if num_words < MIN_WORDS:
        return "too_few_words"
    if num_words > MAX_WORDS:
        return "too_many_words"

    mean_word_length = sum(len(w) for w in words) / num_words
    if mean_word_length < MIN_MEAN_WORD_LENGTH or mean_word_length > MAX_MEAN_WORD_LENGTH:
        return "mean_word_length"

    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_lines = sum(line.rstrip().endswith(("...", "…")) for line in lines)
        if ellipsis_lines / len(lines) > MAX_ELLIPSIS_LINE_FRACTION:
            return "ellipsis_lines"

    alphabetic_words = sum(any(ch.isalpha() for ch in w) for w in words)
    if alphabetic_words / num_words < MIN_ALPHABETIC_WORD_FRACTION:
        return "non_alphabetic_words"
    return None


def gopher_quality_filter(text: str) -> bool:
    """True if the document passes all implemented Gopher rules."""
    return gopher_rejection_reason(text) is None
