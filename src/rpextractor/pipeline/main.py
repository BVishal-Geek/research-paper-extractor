"""End-to-end pipeline orchestrator.

Runs the three stages in order:
    1. Downloader       — PubMed → data/raw/<timestamp>/*.xml
    2. Preprocessor     — data/raw/**/*.xml → data/processed/<pmcid>.json
    3. Extractor        — data/processed/<pmcid>.json → data/extracted/<pmcid>.json

Each stage owns its own I/O; this file only wires them together and picks the
LLM provider. Load .env at import time so OPENAI_API_KEY / PubMed credentials
are visible to every downstream component regardless of import order.

Usage:
    python -m rpextractor.pipeline.main --provider local_model
    python -m rpextractor.pipeline.main --provider openai --skip-download
"""

import argparse
import sys

from dotenv import load_dotenv

from rpextractor.extraction.extractor import Extractor
from rpextractor.ingestion.downloader import Downloader
from rpextractor.ingestion.pmcid_loader import load_pmcids
from rpextractor.ingestion.preprocessor import Preprocessor
from rpextractor.llm.factory import get_llm_client
from rpextractor.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# CLI-facing name → factory-facing name.
# Keeps the user's mental model ("local_model") distinct from the internal
# provider slug ("ollama"), so we can add more local backends later without
# changing the CLI.
_PROVIDER_ALIASES = {
    "local_model": "ollama",
    "openai": "openai",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. argv is injectable for tests."""
    parser = argparse.ArgumentParser(
        description="Run the full research-paper-extractor pipeline end-to-end.",
    )
    parser.add_argument(
        "--provider",
        choices=list(_PROVIDER_ALIASES.keys()),
        default="local_model",
        help="LLM provider used by the extraction stage (default: local_model).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the PubMed download stage; reuse whatever is in data/raw/.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip XML parsing; reuse whatever is in data/processed/.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip LLM extraction; useful for download + parse only.",
    )
    parser.add_argument(
        "--processed-batch",
        default=None,
        metavar="BATCH",
        help=(
            "Which processed batch to extract from. Values: "
            "'latest' (newest date-stamped subfolder), a date string like "
            "'2026-07-22' (that specific subfolder), or 'all' / unset "
            "(every processed JSON, recursive — default)."
        ),
    )
    parser.add_argument(
        "--pmcids",
        default=None,
        metavar="PATH",
        help=(
            "Path to a whitespace-separated text file of PMCIDs to download. "
            "When set, the PubMed search step is skipped and configs/pubmed.yaml's "
            "max_results cap is ignored. Accepts 'PMC12345' or bare '12345'; "
            "both normalize to canonical 'PMC12345' internally."
        ),
    )
    parser.add_argument(
        "--max-downloads",
        default=None,
        type=int,
        metavar="N",
        help=(
            "Cap on how many papers to download in search-driven mode. "
            "Overrides configs/pubmed.yaml's max_results for this run. "
            "Ignored when --pmcids is given (list is used as-is)."
        ),
    )
    return parser.parse_args(argv)


def run_pipeline(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    provider: str,
    skip_download: bool = False,
    skip_preprocess: bool = False,
    skip_extract: bool = False,
    input_batch: str | None = None,
    pmcids: list[str] | None = None,
    max_downloads: int | None = None,
) -> dict:
    """Run the three stages in order and return a per-stage summary dict.

    `provider` is the factory-facing name (e.g. "ollama", "openai"), NOT the
    CLI alias. Callers coming from the CLI must resolve the alias first.

    `input_batch` is passed through to Extractor so callers can restrict the
    LLM stage to a single date-stamped subfolder. See Extractor docstring.

    `pmcids`, when provided, replaces the PubMed search step in the download
    stage — the Downloader will fetch exactly this list.

    `max_downloads` is a search-mode cap that overrides pubmed.yaml's
    max_results for this run. Ignored when `pmcids` is given.
    """
    summary: dict = {}

    if skip_download:
        logger.info("Skipping download stage")
    else:
        if pmcids is not None:
            logger.info(
                f"STAGE 1/3 — Downloading {len(pmcids)} explicit PMCIDs (search skipped)"
            )
        else:
            logger.info(
                f"STAGE 1/3 — Downloading XMLs from PubMed (max_downloads={max_downloads})"
            )
        summary["download"] = Downloader().run(pmids=pmcids, max_results=max_downloads)

    if skip_preprocess:
        logger.info("Skipping preprocess stage")
    else:
        logger.info("STAGE 2/3 — Parsing XMLs into structured JSON")
        summary["preprocess"] = Preprocessor().run()

    if skip_extract:
        logger.info("Skipping extract stage")
    else:
        logger.info(
            f"STAGE 3/3 — Extracting with LLM (provider={provider}, "
            f"input_batch={input_batch!r})"
        )
        client = get_llm_client(provider=provider)
        summary["extract"] = Extractor(client=client, input_batch=input_batch).run()

    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a shell exit code (0 = success)."""
    args = parse_args(argv)
    provider = _PROVIDER_ALIASES[args.provider]

    logger.info(f"Pipeline starting (cli_provider={args.provider}, resolved={provider})")

    pmcids = load_pmcids(args.pmcids) if args.pmcids else None
    if pmcids is not None:
        logger.info(f"Loaded {len(pmcids)} PMCIDs from {args.pmcids}")

    summary = run_pipeline(
        provider=provider,
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess,
        skip_extract=args.skip_extract,
        input_batch=args.processed_batch,
        pmcids=pmcids,
        max_downloads=args.max_downloads,
    )

    logger.info("Pipeline complete")
    for stage, result in summary.items():
        logger.info(f"  {stage}: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
