# Research Paper Extractor

A GenAI-powered pipeline that extracts entities from research papers to validate if they contain useful data or results for building ML models.

## What it does

1. **Ingests** papers from PubMed Central via the Entrez API
2. **Parses** XML into structured text (abstract, methods, results, etc.)
3. **Chunks** text for LLM processing
4. **Extracts** entities (datasets, models, metrics, results) using Ollama
5. **Validates** output with guardrails (Pydantic + Guardrails AI)
6. **Evaluates** accuracy against a golden dataset (exact, fuzzy, semantic matching)
7. **Stores** extracted entities in PostgreSQL for future search

## Tech Stack

- **LLM**: Ollama (local, on-prem)
- **Database**: Still deciding
- **Language**: Python 3.11+

## Setup

```bash
git clone https://github.com/your-org/research-paper-extractor.git
cd research-paper-extractor
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Create a `.env` file (see `.env.example`):

```
PUBMED_EMAIL=your_email
PUBMED_API_KEY=your_key
LOG_DIR=logs
```

## Project Structure

```
research-paper-extractor/
├── configs/              # YAML configuration files
├── data/                 # Raw XMLs and processed text
├── src/rpextractor/      # Main package
│   ├── ingestion/        # PubMed API, XML parsing, downloading
│   ├── chunking/         # Text chunking strategies
│   ├── llm/              # Ollama client and prompt management
│   ├── extraction/       # Entity extraction pipeline
│   ├── guardrails/       # Input/output validation
│   ├── evaluation/       # Golden dataset comparison
│   ├── database/         # PostgreSQL persistence
│   ├── pipeline/         # End-to-end orchestration
│   └── utils/            # Logger, config loader, constants
├── tests/                # Unit and integration tests
├── docker/               # Deployment configs
└── docs/                 # Documentation
```

## Current Status

- [x] PubMed search and fetch
- [x] Threaded XML downloader
- [x] XML parser with configurable sections
- [ ] Text chunking
- [ ] Ollama integration
- [ ] Entity extraction
- [ ] Guardrails
- [ ] Evaluation pipeline
- [ ] Database storage
- [ ] Streamlit app