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
ollama pull qwen2.5:7b-instruct
```

## Environment Variables

Add these to `.env` (see `.env.example`):

| Variable | Required for | Purpose |
| --- | --- | --- |
| `EMAIL` | Ingestion | Identifies you to the PubMed Entrez API |
| `API_KEY_PUBMED` | Ingestion | PubMed API key (higher rate limits) |
| `OPENAI_API_KEY` | OpenAI provider only | Used when `configs/llm.yaml` sets `provider: openai` |

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
│   ├── evaluation/             # (to be built) ground truth loader + metrics
│   ├── pipeline/               # (to be built) end-to-end orchestrator
│   └── utils/                  # config loader, logger, text cleaner
├── tests/                      # pytest suite
├── requirements.txt
└── setup.py
```

## Running the Pipeline

```python
from rpextractor.ingestion.downloader import Downloader
from rpextractor.ingestion.preprocessor import Preprocessor
from rpextractor.extraction.extractor import Extractor

Downloader().run()      # PubMed → data/raw/<timestamp>/*.xml
Preprocessor().run()    # → data/processed/<pmcid>.json
Extractor().run()       # → data/extracted/<pmcid>.json
```

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
```
