"""End-to-end extraction stage.

For each parsed paper JSON in data/processed/, this module builds the LLM
input, calls the LLM, validates the response against the schema, and writes
the validated result to data/extracted/<pmcid>.json. On validation failure,
retries once with the validation error appended to the user prompt so the
LLM has feedback on what went wrong.

Mirrors the Downloader / Preprocessor class shape (run() + _extract_single()
+ summary dict).
"""

import json
import os
from pathlib import Path

from pydantic import ValidationError

from rpextractor.extraction.input_builder import build_paper_text
from rpextractor.extraction.prompts import (
    SYSTEM_PAPER_EVALUATION_PROMPT,
    build_user_prompt,
)
from rpextractor.extraction.schema import ExtractionResult
from rpextractor.llm.base import BaseLLMClient
from rpextractor.llm.factory import get_llm_client
from rpextractor.utils.config import BASE_DIR
from rpextractor.utils.logger import get_logger

file_name = os.path.basename(__file__)
logger = get_logger(file_name)

# Total LLM attempts per paper: 1 initial + up to (MAX_ATTEMPTS - 1) retries.
# The retry loop feeds both the previous raw output AND the ValidationError
# back to the model so it can see what it produced and what was wrong.
DEFAULT_MAX_ATTEMPTS = 4


class Extractor:
    """Runs LLM extraction over parsed papers and persists validated results."""

    def __init__(
        self,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        client: BaseLLMClient | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        """Initialize with input/output dirs and an LLM client.

        Args:
            input_dir: Where parsed JSONs live. Defaults to BASE_DIR/data/processed.
            output_dir: Where extracted JSONs go. Defaults to BASE_DIR/data/extracted.
            client: LLM client. Defaults to the configured one from llm.yaml.
                Inject a fake client here for tests.
            max_attempts: Total LLM attempts per paper (initial + retries).
                Defaults to 4 = 1 initial + 3 retries.
        """
        self.input_dir = input_dir or (BASE_DIR / "data" / "processed")
        self.output_dir = output_dir or (BASE_DIR / "data" / "extracted")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or get_llm_client()
        self.max_attempts = max_attempts

        logger.info(
            f"Extractor initialized. Input: {self.input_dir}, "
            f"Output: {self.output_dir}, max_attempts: {self.max_attempts}"
        )

    def _already_extracted(self, pmcid: str) -> bool:
        """Return True if the extracted JSON for this pmcid already exists."""
        return (self.output_dir / f"{pmcid}.json").exists()

    def _build_retry_message(
        self, original_user_msg: str, attempt: int, prev_raw: str, error: ValidationError
    ) -> str:
        """Compose the retry prompt: original paper + previous output + error."""
        return (
            f"{original_user_msg}\n\n"
            f"Attempt {attempt} failed validation.\n"
            f"Your previous output was:\n{prev_raw}\n\n"
            f"Validation error:\n{error}\n\n"
            "Return ONLY valid JSON that matches the schema exactly. "
            "Fix the specific fields flagged above."
        )

    def _call_and_validate(self, user_msg: str) -> ExtractionResult:
        """Send the prompt, validate, retry up to max_attempts total.

        Each retry sends the previous raw LLM output and the exact
        ValidationError message back to the model, so it has both what it
        produced and what was wrong with it.
        """
        current_msg = user_msg
        last_error: ValidationError | None = None
        last_raw: str = ""

        for attempt in range(1, self.max_attempts + 1):
            raw = self.client.chat_json(SYSTEM_PAPER_EVALUATION_PROMPT, current_msg)
            try:
                return ExtractionResult.model_validate_json(raw)
            except ValidationError as e:
                last_error = e
                last_raw = raw
                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed validation"
                )
                if attempt < self.max_attempts:
                    current_msg = self._build_retry_message(user_msg, attempt, raw, e)

        # All attempts exhausted — surface the last error to the caller.
        logger.error(
            f"All {self.max_attempts} attempts failed. Last raw output: {last_raw!r}"
        )
        raise last_error  # pylint: disable=raising-bad-type  # None-guarded by loop

    def _extract_single(self, processed_path: Path) -> dict:
        """Extract one paper and write the validated result to disk."""
        pmcid = processed_path.stem

        if self._already_extracted(pmcid):
            logger.info(f"{pmcid} already extracted, skipping")
            return {"pmcid": pmcid, "status": "skipped", "message": "Already extracted"}

        try:
            parsed = json.loads(processed_path.read_text(encoding="utf-8"))
            paper_text = build_paper_text(parsed)

            if not paper_text:
                logger.warning(f"{pmcid} produced empty paper text, skipping LLM call")
                return {"pmcid": pmcid, "status": "failed", "message": "Empty paper text"}

            user_msg = build_user_prompt(paper_text)
            result = self._call_and_validate(user_msg)

            output_path = self.output_dir / f"{pmcid}.json"
            output_path.write_text(
                json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(f"{pmcid} extracted to {output_path.name}")
            return {"pmcid": pmcid, "status": "success", "message": str(output_path.name)}

        except Exception as e:
            logger.error(f"{pmcid} extraction failed: {e}")
            return {"pmcid": pmcid, "status": "failed", "message": str(e)}

    def run(self) -> dict:
        """Extract every parsed paper in input_dir. Returns a summary dict."""
        logger.info("Starting extractor")

        # Recursive glob handles both flat (data/processed/*.json) and
        # date-stamped (data/processed/YYYY-MM-DD/*.json) layouts.
        json_files = list(self.input_dir.glob("**/*.json"))

        if not json_files:
            logger.warning(f"No processed JSON files under {self.input_dir}")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        logger.info(f"Found {len(json_files)} processed papers")

        results = {"total": len(json_files), "success": 0, "skipped": 0, "failed": 0}

        for path in json_files:
            r = self._extract_single(path)
            results[r["status"]] += 1

        logger.info(
            f"Extraction complete. "
            f"Total: {results['total']}, "
            f"Success: {results['success']}, "
            f"Skipped: {results['skipped']}, "
            f"Failed: {results['failed']}"
        )

        return results
