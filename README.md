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
├── tests/                      # pytest suite (82 tests)
├── Dockerfile                  # Container recipe — see "Running in Docker"
├── .dockerignore
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

# Only extract the newest batch of processed papers (skips older date folders)
python -m rpextractor.pipeline.main --skip-download --skip-preprocess \
    --processed-batch latest

# Only extract a specific batch — reproducible cost / scope
python -m rpextractor.pipeline.main --provider openai --skip-download \
    --skip-preprocess --processed-batch 2026-07-22

# Download an explicit list of PMCIDs instead of running a PubMed search
python -m rpextractor.pipeline.main --pmcids path/to/pmcids.txt

# Same idea, end-to-end with OpenAI on a ground-truth set
python -m rpextractor.pipeline.main --provider openai --pmcids ground_truth_pmcids.txt

# Search-driven, but cap at 2 papers instead of using pubmed.yaml's max_results
python -m rpextractor.pipeline.main --max-downloads 2
```

CLI flags:

| Flag | Effect |
| --- | --- |
| `--provider {local_model,openai}` | LLM backend for the extraction stage. Default: `local_model` (Ollama) |
| `--skip-download` | Reuse whatever is already in `data/raw/` |
| `--skip-preprocess` | Reuse whatever is already in `data/processed/` |
| `--skip-extract` | Download + parse only; no LLM call, no cost |
| `--processed-batch BATCH` | Restrict extraction to a subset of `data/processed/`. See below. |
| `--pmcids PATH` | Download an explicit list of PMCIDs instead of running a PubMed search. See below. |
| `--max-downloads N` | Cap the search-driven download at N papers. Overrides `max_results` in `configs/pubmed.yaml` for this run. Ignored when `--pmcids` is given. |

**`--processed-batch` values:**

| Value | What gets extracted |
| --- | --- |
| unset *(default)* | Every JSON found under `data/processed/`, recursively |
| `all` | Same as unset — explicit "everything" |
| `latest` | Only the newest date-stamped subfolder (e.g. `data/processed/2026-07-24/`) |
| `YYYY-MM-DD` (e.g. `2026-07-22`) | Only that exact subfolder. Empty result (no crash) if it doesn't exist |

Useful when the preprocessor has produced multiple batches on different
days and you want to scope a run to just one — e.g. to control OpenAI cost,
or to iterate on prompts against a fixed set of papers.

**`--pmcids PATH` — download an explicit list**

By default, the download stage runs the PubMed query in
`configs/pubmed.yaml` and downloads whatever it finds (up to `max_results`).
When you pass `--pmcids <file>`, the search step is skipped and the
downloader fetches exactly the PMCIDs listed in that file — no cap, no
query.

*File format:* plain text, whitespace-separated. Any layout works — one per
line, several per line, mixed. Both `PMC12345` and bare `12345` are
accepted; the loader normalizes everything to canonical `PMC12345`
internally. Non-numeric junk (headers, comments) is silently ignored.

Example `pmcids.txt`:

```
PMC12662548
PMC13221707
PMC6685771
```

Or equivalently:

```
# my ground truth set — comments are ignored
PMC12662548 12662548 pmc13221707
PMC6685771
```

*When to use it (three concrete scenarios):*

1. **You have a ground truth set to evaluate against.** Your CSV lists,
   say, 20 specific PMCIDs. You want the extractor to produce output for
   *those exact 20 papers* so the eval module can diff `data/extracted/`
   against your gold values. Run:
   ```bash
   python -m rpextractor.pipeline.main --provider openai --pmcids gt_pmcids.txt
   ```
2. **You're reproducing a published result.** The paper lists specific
   PubMed IDs. Drop them in a text file and rerun the pipeline — same
   inputs every time, no dependence on PubMed query result ordering.
3. **You want to iterate on a small, cheap batch.** Pull 2-3 PMCIDs into
   a file, run with `--provider openai`, and you know exactly what the run
   will cost before it starts (no surprise from `max_results`).

**How many papers per run (search-driven mode):** set `max_results` in
`configs/pubmed.yaml`, OR pass `--max-downloads N` on the CLI to override
it for a single run without editing the config. When `--pmcids` is given,
both are ignored — the full list is downloaded.

**Retry behavior:** if the LLM returns malformed JSON or a response that
violates the schema's "not found" rule, the extractor retries up to 3 times
(4 total attempts). Each retry sends the previous raw LLM output and the
exact `ValidationError` back to the model so it can self-correct.

**Cost guard (OpenAI only):** `configs/llm.yaml` has a
`max_cost_usd_per_run` ceiling. The OpenAI client tracks cumulative spend
per run and raises `CostCeilingExceeded` before making a call that would
push it over.

## Running in Docker

A single `Dockerfile` at the repo root packages the pipeline as a runnable
image. This is the recommended way to hand the project to someone else —
they don't need Python, Ollama, or a matching venv, just Docker and your
`.env`.

### Build the image

```bash
docker build -t rpextractor .
```

### Run the pipeline in a container

Two mounts are always needed:
- `--env-file .env` — supplies `EMAIL`, `API_KEY_PUBMED`, and (for OpenAI)
  `OPENAI_API_KEY`. Never bake secrets into the image.
- `-v $(pwd)/data:/app/data` — the container writes `data/raw/`,
  `data/processed/`, `data/extracted/` here. Without this mount, outputs
  are lost when the container exits.

```bash
# Search-driven download + preprocess, no LLM cost
docker run --rm --env-file .env -v $(pwd)/data:/app/data \
    rpextractor --max-downloads 2 --skip-extract

# Full pipeline against a fixed PMCID list, OpenAI provider
docker run --rm --env-file .env -v $(pwd)/data:/app/data \
    -v $(pwd)/pmcids.txt:/app/pmcids.txt \
    rpextractor --provider openai --pmcids /app/pmcids.txt

# See all CLI flags
docker run --rm rpextractor --help
```

### One gotcha — Ollama in a container

`configs/llm.yaml` defaults to `host: http://localhost:11434`. Inside a
container, `localhost` is the container itself, not your host machine —
so an Ollama server running on your host is unreachable by default.

Three ways around it:
1. **Run Ollama on the host, use host networking:**
   ```bash
   docker run --rm --env-file .env --network host \
       -v $(pwd)/data:/app/data rpextractor --provider local_model
   ```
   Linux only; on Mac/Windows use `--add-host=host.docker.internal:host-gateway` and set the host in llm.yaml to `http://host.docker.internal:11434`.
2. **Use OpenAI instead** — no networking to worry about.
3. **Compose an Ollama sidecar** — a docker-compose.yml is not included in
   this minimal setup, but is a natural next step if this becomes a common
   pattern for you.

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**What to expect:** 82 tests, all passing in under a second. Every test
runs offline — no real Ollama, OpenAI, or PubMed calls happen. The LLM
clients are replaced by pre-baked stubs, temp directories isolate file I/O,
and monkeypatched module imports keep the factory tests hermetic.

Coverage by module:

| Test file | Count | What it covers |
| --- | --- | --- |
| `test_schema.py` | 11 | Pydantic model, "not found" validator, case-insensitivity, malformed-JSON rejection |
| `test_input_builder.py` | 5 | Section selection, empty-section skipping, custom section list |
| `test_extractor.py` | 12 | Happy path, retry-with-feedback loop, max-attempts config, skip-if-exists, `input_batch` modes (none / `all` / `latest` / date / nonexistent / flat fallback) |
| `test_preprocessor.py` | 3 | XML → JSON conversion, idempotent skip, empty-input handling |
| `test_pipeline_main.py` | 22 | Orchestrator wiring, skip flags, CLI provider aliasing, `--processed-batch` parsing + passthrough, `--pmcids` file loading + Downloader passthrough, `--max-downloads` int-parsing + passthrough |
| `test_pmcid_loader.py` | 11 | PMCID normalization: prefix handling, case-folding, dedup, whitespace layouts, junk-token dropping |
| `test_factory.py` | 5 | Provider selection, config fallbacks, unknown-provider error |
| `test_openai_client.py` | 5 | Missing key, response parsing, cost accumulation, ceiling guard |
| `test_text_cleaner.py` | 8 | Citation stripping, whitespace collapsing, non-citation preservation |

Run just one file:

```bash
.venv/bin/python -m pytest tests/test_schema.py -v
```
