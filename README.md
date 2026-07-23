# Research Paper Extractor

A pipeline that pulls oncology research papers from PubMed and uses an LLM to
judge whether each paper is well-reported enough to support building a machine
learning model on top of it. The LLM provider is switchable — Ollama for local
inference (default) or OpenAI for cloud inference — via `configs/llm.yaml`.

## Objective

Help [HIVE Research Lab](#) build as many ML models as possible on oncology so
the United States can treat people earlier and better.

## Tech Stack

- **Language:** Python 3.11+
- **LLM providers:** Ollama (local, default) or OpenAI (cloud)
- **PubMed access:** Biopython (Entrez API)
- **Schema validation:** Pydantic v2
- **Config:** YAML (`configs/`) + `.env` for secrets
- **Tests:** pytest

## Setup

```bash
git clone https://github.com/BVishal-Geek/research-paper-extractor.git
cd research-paper-extractor
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Copy `.env.example` to `.env` and fill in your credentials (see below).

If using the default Ollama provider, also make sure Ollama is running and
the model is pulled:

```bash
ollama serve
ollama pull qwen3.5:4b
```

## Environment Variables

Add these to `.env` (see `.env.example`):

| Variable | Required for | Purpose |
| --- | --- | --- |
| `EMAIL` | Ingestion | Identifies you to the PubMed Entrez API |
| `API_KEY_PUBMED` | Ingestion | PubMed API key (higher rate limits) |
| `OPENAI_API_KEY` | OpenAI provider only | Used when the pipeline is run with `--provider openai` |

## Project Structure

```
research-paper-extractor/
├── configs/                    # YAML config (pubmed, parser, llm)
├── data/                       # gitignored — generated at runtime
│   ├── raw/                    # Downloaded PubMed XMLs
│   ├── processed/              # Parsed papers as JSON
│   └── extracted/              # LLM extraction results as JSON
├── src/rpextractor/
│   ├── ingestion/              # PubMedClient, Downloader, XMLParser, Preprocessor
│   ├── extraction/             # schema, prompts, input_builder, extractor
│   ├── llm/                    # BaseLLMClient + Ollama/OpenAI clients + factory
│   ├── pipeline/               # main.py — end-to-end orchestrator (CLI entry point)
│   ├── evaluation/             # (to be built) ground truth loader + metrics
│   └── utils/                  # config loader, logger, text cleaner
├── tests/                      # pytest suite (52 tests)
├── requirements.txt
└── setup.py
```

## Running the Pipeline

The orchestrator at `src/rpextractor/pipeline/main.py` runs the three stages
in order: **download → preprocess → extract**.

```bash
# Default: local Ollama, all three stages
python -m rpextractor.pipeline.main

# Use OpenAI instead of Ollama
python -m rpextractor.pipeline.main --provider openai

# Skip stages you've already run (e.g. iterate on prompts without re-downloading)
python -m rpextractor.pipeline.main --skip-download --skip-preprocess
python -m rpextractor.pipeline.main --provider openai --skip-download
```

CLI flags:

| Flag | Effect |
| --- | --- |
| `--provider {local_model,openai}` | LLM backend for the extraction stage. Default: `local_model` (Ollama) |
| `--skip-download` | Reuse whatever is already in `data/raw/` |
| `--skip-preprocess` | Reuse whatever is already in `data/processed/` |
| `--skip-extract` | Download + parse only; no LLM call, no cost |

**How many papers per run:** set `max_results` in `configs/pubmed.yaml`.

**Retry behavior:** if the LLM returns malformed JSON or a response that
violates the schema's "not found" rule, the extractor retries up to 3 times
(4 total attempts). Each retry sends the previous raw LLM output and the
exact `ValidationError` back to the model so it can self-correct.

**Cost guard (OpenAI only):** `configs/llm.yaml` has a
`max_cost_usd_per_run` ceiling. The OpenAI client tracks cumulative spend
per run and raises `CostCeilingExceeded` before making a call that would
push it over.

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**What to expect:** 52 tests, all passing in under a second. Every test
runs offline — no real Ollama, OpenAI, or PubMed calls happen. The LLM
clients are replaced by pre-baked stubs, temp directories isolate file I/O,
and monkeypatched module imports keep the factory tests hermetic.

Coverage by module:

| Test file | Count | What it covers |
| --- | --- | --- |
| `test_schema.py` | 11 | Pydantic model, "not found" validator, case-insensitivity, malformed-JSON rejection |
| `test_input_builder.py` | 5 | Section selection, empty-section skipping, custom section list |
| `test_extractor.py` | 6 | Happy path, retry-with-feedback loop, max-attempts config, skip-if-exists |
| `test_preprocessor.py` | 3 | XML → JSON conversion, idempotent skip, empty-input handling |
| `test_pipeline_main.py` | 9 | Orchestrator wiring, skip flags, CLI provider aliasing |
| `test_factory.py` | 5 | Provider selection, config fallbacks, unknown-provider error |
| `test_openai_client.py` | 5 | Missing key, response parsing, cost accumulation, ceiling guard |
| `test_text_cleaner.py` | 8 | Citation stripping, whitespace collapsing, non-citation preservation |

Run just one file:

```bash
.venv/bin/python -m pytest tests/test_schema.py -v
```
