"""Pydantic schema for paper-level reportability extraction.

Mirrors the JSON contract enforced by SYSTEM_PAPER_EVALUATION_PROMPT in
prompts.py. Each `_E` field is a binary verdict and each `_reason` is either
a short paraphrased quote from the paper (when verdict=1) or the literal
string "not found" (when verdict=0). The model_validator enforces that rule
so LLM violations raise a ValidationError instead of being silently accepted.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Verdict = Literal[0, 1]
NOT_FOUND = "not found"

_FIELDS_WITH_REASONS = (
    "condition_E",
    "N_E",
    "dataset_E",
    "intervention_E",
    "pr_endpoint_E",
    "R_criteria_E",
)


class ExtractionResult(BaseModel):
    """Reportability assessment for a single paper."""

    paper_title: str = Field(description="Paper title as written by the authors.")

    condition_E: Verdict = Field(
        description="1 if the experimental or responder cohort is clearly defined.",
    )
    condition_E_reason: str = Field(
        description="Short paraphrased quote supporting condition_E=1, or 'not found' if 0.",
    )

    N_E: Verdict = Field(
        description="1 if the sample size or data volume for the experimental group is clearly stated.",
    )
    N_E_reason: str = Field(
        description="Short paraphrased quote supporting N_E=1, or 'not found' if 0.",
    )

    dataset_E: Verdict = Field(
        description="1 if the experimental group data source is clearly specified and accessible (GEO, TCGA, public DBs).",
    )
    dataset_E_reason: str = Field(
        description="Short paraphrased quote supporting dataset_E=1, or 'not found' if 0.",
    )

    intervention_E: Verdict = Field(
        description="1 if the experimental treatment, intervention, or biological condition is clearly described.",
    )
    intervention_E_reason: str = Field(
        description="Short paraphrased quote supporting intervention_E=1, or 'not found' if 0.",
    )

    pr_endpoint_E: Verdict = Field(
        description="1 if a clear primary outcome or response endpoint is defined for the experimental group.",
    )
    pr_endpoint_E_reason: str = Field(
        description="Short paraphrased quote supporting pr_endpoint_E=1, or 'not found' if 0.",
    )

    R_criteria_E: Verdict = Field(
        description="1 if explicit criteria defining responders in the experimental group are stated.",
    )
    R_criteria_E_reason: str = Field(
        description="Short paraphrased quote supporting R_criteria_E=1, or 'not found' if 0.",
    )

    @model_validator(mode="after")
    def _enforce_reason_rule(self) -> "ExtractionResult":
        for field in _FIELDS_WITH_REASONS:
            verdict = getattr(self, field)
            reason = getattr(self, f"{field}_reason").strip()
            if verdict == 0 and reason.lower() != NOT_FOUND:
                raise ValueError(
                    f"{field}=0 requires {field}_reason='{NOT_FOUND}', got {reason!r}"
                )
            if verdict == 1 and (not reason or reason.lower() == NOT_FOUND):
                raise ValueError(
                    f"{field}=1 requires a non-empty supporting reason, got {reason!r}"
                )
        return self
