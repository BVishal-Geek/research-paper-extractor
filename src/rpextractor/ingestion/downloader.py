
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rpextractor.ingestion.pubmed_client import PubMedClient
from rpextractor.utils.logger import get_logger
from rpextractor.utils.config import BASE_DIR
from datetime import datetime

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
        self.output_dir = BASE_DIR / "data" / "raw" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloader initialized. Output: {self.output_dir}, Workers: {self.max_workers}")

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

    def run(self, query: str = None) -> dict:
        """Run the full download pipeline: search → fetch → save.

        Args:
            query: Optional search query. Uses config default if None.

        Returns:
            Summary dict with counts of success, skipped, and failed.
        """
        # Step 1: Search for PMIDs
        logger.info("Starting download pipeline")
        pmids = self.client.search(query)

        if not pmids:
            logger.warning("No PMIDs found. Nothing to download.")
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