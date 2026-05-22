# NJ Ethics Pipeline

This project provides a **single self-contained script** to:

1. Crawl NJ legal pages for PDF links
2. Download PDFs with polite rate limiting/retries
3. OCR via OpenRouter
4. Chunk + embed via local Ollama
5. Ingest into Postgres + pgvector

## Prerequisites

- Python 3.10+ (recommended)
- PostgreSQL running locally or reachable remotely
- `pgvector` extension available in your Postgres instance
- Ollama running locally (`ollama serve`)
- OpenRouter API key with access to your selected OCR model

## Included files

- `nj_ethics_pipeline.py` — all logic in one file
- `schema.sql` — DB tables/indexes
- `.env.example` — sample environment config
- `requirements.txt` — Python dependencies

## Default seed sources

If you do not pass any `--base-url`, it uses:

- `https://www.nj.gov/education/legal/ethics/index.shtml`
- `https://www.nj.gov/education/legal/commissioner/index.shtml`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy env template and edit values
cp .env.example .env
# then export variables from .env (example):
set -a; source .env; set +a

# NOTE: the export command above is for bash/zsh shells.
# If DB_USER is left blank, script defaults to your current OS user.

# Postgres database + schema
createdb ethics
psql -d ethics -f schema.sql

# Optional: verify pgvector extension exists
psql -d ethics -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Ollama embedding model
ollama pull nomic-embed-text

# Start Ollama server (if not already running)
ollama serve
```

## Environment variables

All supported variables are documented in `.env.example`.

Minimum required for OCR:
- `OPENROUTER_API_KEY`

Commonly adjusted:
- `OPENROUTER_MODEL` (default: `qwen/qwen3-vl-32b-instruct`)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- `TEXT_OUT_DIR` (`''` to disable text file output)

If `DB_USER` is unset/blank, the script falls back to your current OS user.

## OCR model decision notes

Initial OCR testing was done locally with `glm-ocr:latest`, but throughput was too slow for the full corpus.

Because OpenRouter credits were already available, multiple vision models were tested on small batches for speed/cost tradeoffs. The model selected for this pipeline was:

- `qwen/qwen3-vl-32b-instruct`

Reason for selection:
- Good balance of extraction quality, speed, and cost for large-scale batch OCR
- Fast enough to avoid multi-day processing delays seen with local-only OCR

Current cost benchmark (as of this dataset snapshot):
- ~5,643 PDFs processed
- Total OCR spend: approximately **$13 USD**

Notes:
- Costs may vary over time based on provider pricing, page counts, and retries.
- You can change OCR model via `OPENROUTER_MODEL` in your environment.

## Run

### Full pipeline (crawl + download + ingest)

```bash
python nj_ethics_pipeline.py
```

### Dry run discovery only

```bash
python nj_ethics_pipeline.py --dry-run
```

### Fast bounded dry-run (recommended for testing)

```bash
python nj_ethics_pipeline.py --dry-run --max-depth 1 --max-pages 100 --progress-every 10
```

### First-run smoke test (recommended)

Run a small dry-run first to verify network/auth/config before full ingest:

```bash
python nj_ethics_pipeline.py --dry-run --max-depth 1 --max-pages 20 --progress-every 10
```

### Download only

```bash
python nj_ethics_pipeline.py --download-only
```

### Ingest only (existing local PDFs)

```bash
python nj_ethics_pipeline.py --ingest-only --pdf-dir pdf_files
```

## Useful options

- `--base-url <url>` (repeatable)
- `--base-urls-file seed_urls.txt`
- `--max-depth 2`
- `--max-pages 500` (safety cap; `0` disables)
- `--progress-every 25` (crawl status logging)
- `--request-delay 0.8`
- `--request-jitter 0.3`
- `--max-retries 5`
- `--exclude-prefix <url_prefix>` (repeatable)
- `--no-default-exclusions`
- `--text-out-dir ocr_texts` (or `''` to disable text output)
- `--force` to reprocess already ingested documents

### Default exclusions

By default, URLs starting with these prefixes are excluded from crawl/discovery:

- `https://www.nj.gov/education/legal/examiners/`
- `https://www.nj.gov/education/legal/sboe/`

Override behavior:

```bash
# add your own additional exclusion prefixes
python nj_ethics_pipeline.py --dry-run --exclude-prefix https://example.com/unwanted/

# disable built-in defaults entirely
python nj_ethics_pipeline.py --dry-run --no-default-exclusions
```

## Discovery output options

Print discovered PDFs:

```bash
python nj_ethics_pipeline.py --dry-run --print-discovered
```

Print only newly discovered PDFs (not already downloaded by default):

```bash
python nj_ethics_pipeline.py --dry-run --print-discovered --print-new-only
```

Choose what “new” means:

```bash
# compare against downloaded files only (default)
python nj_ethics_pipeline.py --dry-run --print-discovered --print-new-only --new-check-scope downloaded

# compare against DB ingested filenames
python nj_ethics_pipeline.py --dry-run --print-discovered --print-new-only --new-check-scope ingested

# require missing from both local files and DB
python nj_ethics_pipeline.py --dry-run --print-discovered --print-new-only --new-check-scope both
```

Write discovered output to file:

```bash
python nj_ethics_pipeline.py --dry-run --discovered-output discovered.json
python nj_ethics_pipeline.py --dry-run --discovered-output discovered.txt
```

## Notes

- Requires network access to source sites + OpenRouter.
- Requires running Ollama server for embeddings.
- Uses filename-based dedupe for ingestion unless `--force` is used.

## Troubleshooting

- `OPENROUTER_API_KEY is not set`
  - Ensure `.env` is loaded in your shell and key is non-empty.

- `Embedding model ... not pulled in Ollama`
  - Run: `ollama pull nomic-embed-text`

- Postgres auth/connection errors
  - Check `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD`.
  - If using local peer auth, try leaving `DB_PASSWORD` empty and keep `DB_USER` blank to use current OS user.

- `relation ... does not exist` or vector type/index errors
  - Re-run schema setup: `psql -d ethics -f schema.sql`
  - Confirm extension: `psql -d ethics -c "CREATE EXTENSION IF NOT EXISTS vector;"`
