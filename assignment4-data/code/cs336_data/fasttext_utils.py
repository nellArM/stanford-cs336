"""Lazy, per-process loading of fastText models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import fasttext

# fastText prints a deprecation warning to stderr on every load_model call.
fasttext.FastText.eprint = lambda *args, **kwargs: None


@lru_cache(maxsize=None)
def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"fastText model not found at {path}. Run `uv run python scripts/download_data.py --offline-only` "
            "to download the public classifiers."
        )
    return fasttext.load_model(str(path))


def predict_top(path: str | Path, text: str) -> tuple[str, float]:
    """Top label (without the ``__label__`` prefix) and its probability, clipped to [0, 1]."""
    # fastText treats a newline as the end of the example, so flatten the text first.
    labels, probs = load_model(path).predict(text.replace("\n", " "), k=1)
    if not labels:
        return "", 0.0
    return labels[0].removeprefix("__label__"), float(min(max(probs[0], 0.0), 1.0))
