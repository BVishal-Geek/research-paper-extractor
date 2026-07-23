"""Threaded downloader that fetches PubMed XMLs and persists them to disk."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from rpextractor.ingestion.pubmed_client import PubMedClient
from rpextractor.utils.config import BASE_DIR
from rpextractor.utils.logger import get_logger

file_name = os.path.basename(__file__)
logger = get_logger(file_name)

class Downloader:
    """Downloads raw XMLs from PubMed and saves them to data/raw/."""

    def __init__(self, max_workers: int = 5, sleep_time: float = 0.5):
        """Initialize downloader with PubMedClient and output directory.

        Args:
            max_workers: Number of parallel download threads.
        """
        self.client = PubMedClient()
        self.max_workers = max_workers
        # seconds to sleep between requests to avoid rate limits
        self.sleep_time = sleep_time
        # Resolve output directory relative to project root
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = BASE_DIR / "data" / "raw" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloader initialized. Output: {self.output_dir}, Workers: {self.max_workers}"
        )

    def _already_downloaded(self, pmid: str) -> bool:
        """Check if a paper's XML already exists on disk."""
        file_path = self.output_dir / f"{pmid}.xml"
        return file_path.exists()

    def _download_single(self, pmid: str) -> dict:
        """Download a single paper's XML and save to disk.

        Args:
            pmid: PubMed/PMC ID to download.

        Returns:
            Dict with pmid, status ('success', 'skipped', 'failed'), and message.
        """
        # Skip if already downloaded
        if self._already_downloaded(pmid):
            logger.info(f"{pmid} already exists, skipping")
            return {"pmid": pmid, "status": "skipped", "message": "Already downloaded"}

        time.sleep(self.sleep_time)  # Sleep to respect rate limits

        try:
            xml_data = self.client.fetch(pmid)

            if not xml_data:
                logger.warning(f"{pmid} returned empty XML")
                return {"pmid": pmid, "status": "failed", "message": "Empty XML"}

            # Check if XML has actual content
            if "<article" not in xml_data and "<body" not in xml_data:
                logger.warning(f"{pmid} has no full-text content")
                return {"pmid": pmid, "status": "failed", "message": "No full-text available"}

            # Save raw XML
            file_path = self.output_dir / f"{pmid}.xml"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(xml_data)

            logger.info(f"{pmid} saved ({len(xml_data)} bytes)")
            return {"pmid": pmid, "status": "success", "message": f"Saved ({len(xml_data)} bytes)"}

        except Exception as e:
            logger.error(f"{pmid} download failed: {e}")
            return {"pmid": pmid, "status": "failed", "message": str(e)}

    def run(
        self,
        query: str = None,
        pmids: list[str] | None = None,
        max_results: int | None = None,
    ) -> dict:
        """Run the download pipeline.

        Two modes:
            - Explicit list: pass `pmids=[...]` to download exactly that set.
              The PubMed search step is skipped and `max_results` is ignored.
            - Search-driven (default): the PubMed query from configs/pubmed.yaml
              is used. `max_results` overrides the config cap when provided.

        Args:
            query: Optional search query override (search-driven mode only).
            pmids: Optional explicit PMCID list. Takes precedence over query.
            max_results: Optional cap that overrides pubmed.yaml for this run
                (search-driven mode only).

        Returns:
            Summary dict with counts of success, skipped, and failed.
        """
        logger.info("Starting download pipeline")

        if pmids is not None:
            logger.info(
                f"Using {len(pmids)} explicit PMCIDs (search step skipped)"
            )
        else:
            pmids = self.client.search(query, max_results=max_results)

        if not pmids:
            logger.warning("No PMIDs to download.")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        logger.info(f"Found {len(pmids)} papers to process")

        # Step 2: Download in parallel
        results = {"total": len(pmids), "success": 0, "skipped": 0, "failed": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._download_single, pmid): pmid
                for pmid in pmids
            }

            for future in as_completed(futures):
                pmid = futures[future]
                try:
                    result = future.result()
                    results[result["status"]] += 1
                except Exception as e:
                    logger.error(f"{pmid} unexpected error: {e}")
                    results["failed"] += 1

        # Step 3: Log summary
        logger.info(
            f"Download complete. "
            f"Total: {results['total']}, "
            f"Success: {results['success']}, "
            f"Skipped: {results['skipped']}, "
            f"Failed: {results['failed']}"
        )

        return results
