"""Tests for ExtractionResult schema and its validator.

The schema's model_validator enforces the "not found" rule that keeps the
LLM honest — verdict=0 must be paired with reason='not found', verdict=1 must
have a real reason. These tests lock that contract down so a schema change
can't silently loosen it.
"""

import json

import pytest
from pydantic import ValidationError

from rpextractor.extraction.schema import ExtractionResult


def _valid_all_zero() -> dict:
    """Baseline: every verdict=0 with reason='not found'. Valid."""
    return {
        "paper_title": "Test paper",
        "condition_E": 0, "condition_E_reason": "not found",
        "N_E": 0, "N_E_reason": "not found",
        "dataset_E": 0, "dataset_E_reason": "not found",
        "intervention_E": 0, "intervention_E_reason": "not found",
        "pr_endpoint_E": 0, "pr_endpoint_E_reason": "not found",
        "R_criteria_E": 0, "R_criteria_E_reason": "not found",
    }


def test_all_zero_baseline_is_valid():
    ExtractionResult(**_valid_all_zero())


def test_all_one_with_real_reasons_is_valid():
    data = _valid_all_zero()
    for field in [
        "condition_E", "N_E", "dataset_E",
        "intervention_E", "pr_endpoint_E", "R_criteria_E",
    ]:
        data[field] = 1
        data[f"{field}_reason"] = f"quote supporting {field}"

    ExtractionResult(**data)


def test_verdict_0_with_non_not_found_reason_raises():
    data = _valid_all_zero()
    data["condition_E_reason"] = "cohort was unclear"  # not "not found"

    with pytest.raises(ValidationError, match="condition_E=0 requires"):
        ExtractionResult(**data)


def test_verdict_1_with_not_found_reason_raises():
    data = _valid_all_zero()
    data["N_E"] = 1
    data["N_E_reason"] = "not found"  # should be a real reason

    with pytest.raises(ValidationError, match="N_E=1 requires"):
        ExtractionResult(**data)


def test_verdict_1_with_empty_reason_raises():
    data = _valid_all_zero()
    data["dataset_E"] = 1
    data["dataset_E_reason"] = "   "  # whitespace-only counts as empty

    with pytest.raises(ValidationError, match="dataset_E=1 requires"):
        ExtractionResult(**data)


def test_not_found_is_case_insensitive():
    data = _valid_all_zero()
    data["intervention_E_reason"] = "NOT FOUND"  # uppercase should still count

    ExtractionResult(**data)  # must not raise


def test_not_found_ignores_surrounding_whitespace():
    data = _valid_all_zero()
    data["pr_endpoint_E_reason"] = "  not found  "

    ExtractionResult(**data)  # must not raise


def test_invalid_verdict_value_raises():
    data = _valid_all_zero()
    data["condition_E"] = 2  # Literal[0, 1] rejects anything else

    with pytest.raises(ValidationError):
        ExtractionResult(**data)


def test_missing_required_field_raises():
    data = _valid_all_zero()
    del data["paper_title"]

    with pytest.raises(ValidationError):
        ExtractionResult(**data)


def test_model_validate_json_parses_string():
    """This is the exact path the extractor uses on the raw LLM output."""
    raw_json = json.dumps(_valid_all_zero())

    result = ExtractionResult.model_validate_json(raw_json)
    assert result.paper_title == "Test paper"
    assert result.condition_E == 0


def test_model_validate_json_rejects_invalid_json_output():
    """LLM returned malformed JSON — must raise ValidationError, not silently pass."""
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate_json('{"paper_title": "x", "condition_E":')
