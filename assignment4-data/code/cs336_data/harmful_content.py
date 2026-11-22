"""NSFW and toxic-speech classification with the Dolma Jigsaw fastText models."""

from __future__ import annotations

from cs336_data.fasttext_utils import predict_top
from cs336_data.paths import NSFW_MODEL_PATH, TOXIC_MODEL_PATH


def classify_nsfw(text: str) -> tuple[str, float]:
    """Return ("nsfw" | "non-nsfw", confidence in that label)."""
    return predict_top(NSFW_MODEL_PATH, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Return ("toxic" | "non-toxic", confidence in that label)."""
    return predict_top(TOXIC_MODEL_PATH, text)


def harmful_probability(text: str) -> tuple[float, float]:
    """(P(nsfw), P(toxic)) regardless of which label is on top."""
    nsfw_label, nsfw_score = classify_nsfw(text)
    toxic_label, toxic_score = classify_toxic_speech(text)
    p_nsfw = nsfw_score if nsfw_label == "nsfw" else 1.0 - nsfw_score
    p_toxic = toxic_score if toxic_label == "toxic" else 1.0 - toxic_score
    return p_nsfw, p_toxic
