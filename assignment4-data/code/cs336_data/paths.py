"""Filesystem locations used by the local (non-Modal) pipeline.

Everything large lives under ``<repo>/local-shared-data`` (git-ignored), mirroring the
layout of the course's ``/shared-data`` volume.  ``CS336_DATA_ROOT`` overrides it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CS336_DATA_ROOT", REPO_ROOT / "local-shared-data"))

CLASSIFIER_DIR = DATA_ROOT / "classifiers"
LANGID_MODEL_PATH = CLASSIFIER_DIR / "lid.176.bin"
NSFW_MODEL_PATH = CLASSIFIER_DIR / "dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_MODEL_PATH = CLASSIFIER_DIR / "dolma_fasttext_hatespeech_jigsaw_model.bin"

# The trained quality classifier is small (quantized) and ships with the code.
QUALITY_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "quality_classifier.ftz"
PALOMA_NGRAM_PATH = Path(__file__).resolve().parent / "assets" / "paloma_unigram.npz"

EXAMPLE_WARC = DATA_ROOT / "CC" / "example.warc.gz"
EXAMPLE_WET = DATA_ROOT / "CC" / "example.warc.wet.gz"
PALOMA_VALID_BIN = DATA_ROOT / "tokenized_paloma_c4_100_domains_validation.bin"
ENGLISH_WET_DIR = DATA_ROOT / "english-wet-data"
WIKI_DIR = DATA_ROOT / "wiki"
QUALITY_DIR = DATA_ROOT / "quality"
FILTERED_DIR = DATA_ROOT / "filtered"
OUTPUT_DIR = DATA_ROOT / "output"
