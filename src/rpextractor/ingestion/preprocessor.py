"""Preprocess raw XMLs into structured JSON.

Walks data/raw/**/*.xml, runs each through XMLParser, and writes one JSON
file per paper to data/processed/<pmcid>.json. Idempotent — already-processed
papers are skipped, so re-runs after new downloads are cheap.

Mirrors the Downloader class shape (run() + _process_single() + summary).
"""

import json
import os
from datetime import date 
from pathlib import Path

from rpextractor.ingestion.xml_parser import XMLParser
from rpextractor.utils.config import BASE_DIR
from rpextractor.utils.logger import get_logger

file_name = os.path.basename(__file__)
logger = get_logger(file_name)


class Preprocessor:
    """Parses raw XMLs and persists them as structured JSON."""

    def __init__(self, input_dir: Path | None = None, output_dir: Path | None = None):
        """Initialize with input and output directories.

        Args:
            input_dir: Where to find XMLs. Defaults to BASE_DIR/data/raw (all
                subfolders scanned recursively).
            output_dir: Where to write JSONs. Defaults to BASE_DIR/data/processed.
        """
        self.parser = XMLParser()
        self.input_dir = input_dir or (BASE_DIR / "data" / "raw")
        self.output_dir = output_dir or (BASE_DIR / "data" / "processed" / date.today().isoformat())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Preprocessor initialized. Input: {self.input_dir}, Output: {self.output_dir}"
        )

    def _already_processed(self, pmcid: str) -> bool:
        """Return True if the parsed JSON for this pmcid already exists."""
        return (self.output_dir / f"{pmcid}.json").exists()

    def _process_single(self, xml_path: Path) -> dict:
        """Parse one XML file and write the result to data/processed/.

        Uses the XML filename (without extension) as the pmcid fallback if
        the parser cannot extract one from the document.
        """
        fallback_pmcid = xml_path.stem

        if self._already_processed(fallback_pmcid):
            logger.info(f"{fallback_pmcid} already processed, skipping")
            return {"pmcid": fallback_pmcid, "status": "skipped", "message": "Already processed"}

        try:
            parsed = self.parser.parse(str(xml_path))

            if parsed is None:
                logger.warning(f"{fallback_pmcid} parser returned None")
                return {"pmcid": fallback_pmcid, "status": "failed", "message": "Parse returned None"}

            # Prefer the pmcid the parser extracted; fall back to filename.
            pmcid = parsed.get("metadata", {}).get("pmcid") or fallback_pmcid

            output_path = self.output_dir / f"{pmcid}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            logger.info(f"{pmcid} saved to {output_path.name}")
            return {"pmcid": pmcid, "status": "success", "message": str(output_path.name)}

        except Exception as e:
            logger.error(f"{fallback_pmcid} preprocessing failed: {e}")
            return {"pmcid": fallback_pmcid, "status": "failed", "message": str(e)}

    def run(self) -> dict:
        """Process every XML found under input_dir. Returns a summary dict."""
        logger.info("Starting preprocessor")

        xml_files = list(self.input_dir.glob("**/*.xml"))

        if not xml_files:
            logger.warning(f"No XML files found under {self.input_dir}")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        logger.info(f"Found {len(xml_files)} XML files to process")

        results = {"total": len(xml_files), "success": 0, "skipped": 0, "failed": 0}

        for xml_path in xml_files:
            result = self._process_single(xml_path)
            results[result["status"]] += 1

        logger.info(
            f"Preprocessing complete. "
            f"Total: {results['total']}, "
            f"Success: {results['success']}, "
            f"Skipped: {results['skipped']}, "
            f"Failed: {results['failed']}"
        )

        return results
