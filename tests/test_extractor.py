"""Tests for Extractor.

We inject a fake LLM client so no Ollama/OpenAI call happens. The fake
returns pre-baked JSON responses in order, which lets us exercise the
happy path, the retry-on-ValidationError path, and the skip-if-exists path
without any network.
"""

import json
from pathlib import Path

import pytest

from rpextractor.extraction.extractor import Extractor
from rpextractor.llm.base import BaseLLMClient


VALID_LLM_JSON = json.dumps({
    "paper_title": "Test paper",
    "condition_E": 1,
    "condition_E_reason": "Cohort defined as stage III breast cancer patients.",
    "N_E": 0,
    "N_E_reason": "not found",
    "dataset_E": 1,
    "dataset_E_reason": "GEO GSE12345 is used.",
    "intervention_E": 0,
    "intervention_E_reason": "not found",
    "pr_endpoint_E": 0,
    "pr_endpoint_E_reason": "not found",
    "R_criteria_E": 0,
    "R_criteria_E_reason": "not found",
})

# Violates the "reason must be 'not found' when verdict=0" rule.
INVALID_LLM_JSON = json.dumps({
    "paper_title": "Test paper",
    "condition_E": 0,
    "condition_E_reason": "cohort is unclear",  # should be "not found"
    "N_E": 0, "N_E_reason": "not found",
    "dataset_E": 0, "dataset_E_reason": "not found",
    "intervention_E": 0, "intervention_E_reason": "not found",
    "pr_endpoint_E": 0, "pr_endpoint_E_reason": "not found",
    "R_criteria_E": 0, "R_criteria_E_reason": "not found",
})


class FakeClient(BaseLLMClient):
    """LLM stand-in that returns pre-baked responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("FakeClient ran out of responses")
        return self.responses.pop(0)


@pytest.fixture
def dirs_with_one_paper(tmp_path: Path) -> tuple[Path, Path]:
    """Create processed/ with one fake parsed paper and an empty extracted/."""
    processed = tmp_path / "processed"
    processed.mkdir()
    parsed = {
        "metadata": {"title": "Test paper", "pmcid": "PMC123"},
        "abstract": "Abstract text.",
        "methods": "Methods with N=50 patients from GEO GSE12345.",
    }
    (processed / "PMC123.json").write_text(json.dumps(parsed))

    extracted = tmp_path / "extracted"
    return processed, extracted


def test_writes_valid_extraction_result(dirs_with_one_paper):
    """Happy path — one valid LLM response → one output JSON on disk."""
    processed, extracted = dirs_with_one_paper
    client = FakeClient([VALID_LLM_JSON])

    summary = Extractor(input_dir=processed, output_dir=extracted, client=client).run()

    assert summary == {"total": 1, "success": 1, "skipped": 0, "failed": 0}

    out = extracted / "PMC123.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["paper_title"] == "Test paper"
    assert data["condition_E"] == 1
    assert data["dataset_E_reason"] == "GEO GSE12345 is used."


def test_retries_up_to_max_attempts_with_prior_output_and_error(dirs_with_one_paper):
    """Three bad responses then a valid one — 4th attempt succeeds.

    Each retry prompt must include both the previous raw LLM output and the
    validation error, so the model sees what it produced and what was wrong.
    """
    processed, extracted = dirs_with_one_paper
    client = FakeClient([
        INVALID_LLM_JSON,
        INVALID_LLM_JSON,
        INVALID_LLM_JSON,
        VALID_LLM_JSON,
    ])

    summary = Extractor(input_dir=processed, output_dir=extracted, client=client).run()

    assert summary["success"] == 1
    assert len(client.calls) == 4  # 1 initial + 3 retries

    # Retry #1 message (2nd call) must contain the previous raw output AND the error.
    _, retry1_user = client.calls[1]
    assert INVALID_LLM_JSON in retry1_user, "retry must include previous raw output"
    assert "Validation error" in retry1_user, "retry must include the validation error"
    assert "Attempt 1 failed" in retry1_user

    # Retry #3 (4th call) should reference attempt 3.
    _, retry3_user = client.calls[3]
    assert "Attempt 3 failed" in retry3_user


def test_fails_when_all_attempts_invalid(dirs_with_one_paper):
    """Four bad responses in a row → failed=1, no output written."""
    processed, extracted = dirs_with_one_paper
    client = FakeClient([INVALID_LLM_JSON] * 4)

    summary = Extractor(input_dir=processed, output_dir=extracted, client=client).run()

    assert summary["failed"] == 1
    assert len(client.calls) == 4
    assert not (extracted / "PMC123.json").exists()


def test_max_attempts_is_configurable(dirs_with_one_paper):
    """max_attempts=2 → only 1 initial + 1 retry, no extra calls."""
    processed, extracted = dirs_with_one_paper
    client = FakeClient([INVALID_LLM_JSON, INVALID_LLM_JSON])

    summary = Extractor(
        input_dir=processed, output_dir=extracted, client=client, max_attempts=2
    ).run()

    assert summary["failed"] == 1
    assert len(client.calls) == 2


def test_skips_when_already_extracted(dirs_with_one_paper):
    """If output JSON exists, skip — LLM must not be called."""
    processed, extracted = dirs_with_one_paper
    extracted.mkdir()
    (extracted / "PMC123.json").write_text("{}")

    client = FakeClient([])  # would raise if called

    summary = Extractor(input_dir=processed, output_dir=extracted, client=client).run()

    assert summary == {"total": 1, "success": 0, "skipped": 1, "failed": 0}
    assert len(client.calls) == 0


def test_empty_input_dir_returns_zeros(tmp_path: Path):
    """Nothing to extract → summary reports zeros, no crash, no LLM call."""
    processed = tmp_path / "processed"
    processed.mkdir()
    extracted = tmp_path / "extracted"
    client = FakeClient([])

    summary = Extractor(input_dir=processed, output_dir=extracted, client=client).run()

    assert summary == {"total": 0, "success": 0, "skipped": 0, "failed": 0}
    assert len(client.calls) == 0
