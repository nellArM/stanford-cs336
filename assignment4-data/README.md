# CS336 Assignment 4 — Data: filtering Common Crawl for language modeling

Self-study solution to Stanford CS336 (Spring 2026) Assignment 4, handout v26.0.1.
The written report is `writeup/writeup.pdf` (source: `writeup/writeup.typ`).

## What the assignment covers

Turning raw Common Crawl pages into language-model training data: HTML-to-text extraction,
language identification, PII masking, harmful-content detection, rule-based and learned quality
filtering, exact and fuzzy (MinHash + LSH) deduplication, a parallel filtering pipeline, GPT-2
tokenization, and training a ~430M-parameter Transformer on the result, evaluated by perplexity on
the Paloma C4-100-domains benchmark.

## What was implemented

| Handout problem | Code |
|---|---|
| `extract_text` | `cs336_data/extract_text.py` |
| `language_identification` | `cs336_data/language_identification.py` (fastText `lid.176.bin`) |
| `mask_pii` | `cs336_data/pii.py` (e-mail, US phone number, IPv4 regexes) |
| `harmful_content` | `cs336_data/harmful_content.py` (Dolma Jigsaw fastText models) |
| `gopher_quality_filters` | `cs336_data/gopher.py` |
| `quality_classifier` | `cs336_data/wiki_urls.py` → `scripts/scrape_positive_urls.sh` → `cs336_data/train_quality_classifier.py`; inference in `cs336_data/quality_classifier.py`; trained model `cs336_data/assets/quality_classifier.ftz` |
| `exact_deduplication`, `minhash_deduplication` | `cs336_data/deduplication.py` |
| `filter_data`, `tokenize_data` | `cs336_data/filter_pipeline.py` (4 stages, `ProcessPoolExecutor`), `cs336_data/c4_rules.py` |
| `look_at_cc` and the 20-example audits | `cs336_data/audit_samples.py` (runs every primitive over the example WARC) |
| `train_model` | `scripts/train_local.py` (configures and calls the unmodified `scripts/train.py`), `writeup/plot_training.py` |
| English WET set (staff infrastructure) | `is_english` in `cs336_data/wet_files.py`; local parallel equivalent `cs336_data/make_english_wets.py` |

`tests/adapters.py` only forwards to these functions. `cs336_data/paths.py` holds all file locations
(everything large lives in the git-ignored `local-shared-data/`, laid out like the course's
`/shared-data` volume).

## Results

- Tests: `uv run pytest` → 21 passed.
- English WET set: 400 raw WET files of `CC-MAIN-2026-17` → 8,552,606 pages → 2,553,123 English pages.
- Quality classifier: 3,570 Wikipedia-referenced pages vs. 3,570 random Common Crawl pages, 84.2%
  validation accuracy (80.4% after quantization to 12 MB).
- Filtering pipeline: 2,553,123 → 706,468 documents (27.7% kept); 709,480,999 GPT-2 tokens;
  383 s wall-clock on 48 processes. Discards by step: language 42.2%, quality classifier 33.0%,
  Gopher 10.8%, C4-style rules 9.8%, too short after line dedup 4.1%, NSFW 0.09%, toxic 0.07%,
  MinHash duplicates 0.02%.
- Training: 1,024 optimizer steps (537M tokens) of the unmodified ~430M model on the filtered data:
  validation loss **4.111** (perplexity 61.0) on Paloma C4-100-domains after 6.0 hours on one RTX A6000.

## Deviations from the handout

- No access to the course's Modal volume or B200 GPUs: all data was rebuilt locally from public URLs.
- 400 raw WET files instead of 2,500 (runtime extrapolated in the report).
- Wikipedia reference URLs come from 4 shards of the 2026-10-01 dump (the 2026-05-01 dump is gone).
- Training uses the same model, optimizer, schedule and global batch (1,024 sequences/step) but on
  one RTX A6000 with gradient accumulation (2-GPU DDP hit an NCCL hang on this machine) and for 1,024
  instead of 16,384 steps; losses are logged to a
  local file instead of wandb.
- `pyproject.toml` pins `torch` to the CUDA 12.6 wheel index (the server's driver predates CUDA 13);
  `SUNET_ID` in `cs336_data/modal_utils.py` is a placeholder (`"local"`) so the staff modules import.

## How to run

```bash
uv sync
uv run python scripts/download_data.py --offline-only     # classifiers, example WARC/WET, Paloma (~3 GB)
uv run pytest                                             # 21 tests (needs the downloaded classifiers)

# Data (CPU only)
uv run python -m cs336_data.audit_samples                 # statistics + samples for the written audits
uv run python -m cs336_data.make_english_wets --n-files 400 --workers 24
uv run python -m cs336_data.wiki_urls                     # reference URLs -> subsampled_positive_urls.txt
bash scripts/scrape_positive_urls.sh 32                   # wget scrape into WARC files
uv run python -m cs336_data.train_quality_classifier      # -> cs336_data/assets/quality_classifier.ftz
uv run python -m cs336_data.filter_pipeline --workers 48  # -> local-shared-data/filtered/filtered_cc_gpt2_tokens.bin

# Training (one GPU; 32 sequences x 32 accumulation steps) and report
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 scripts/train_local.py \
    --train-bin local-shared-data/filtered/filtered_cc_gpt2_tokens.bin
uv run --with matplotlib python writeup/plot_training.py
uv run --no-project --with typst python -c "import typst; typst.compile('writeup/writeup.typ', output='writeup/writeup.pdf')"
```

The three public fastText classifiers (`*.bin`) are downloaded, not shipped; the trained quality
classifier is shipped as `.ftz` because the packaging script excludes `*.bin`.
