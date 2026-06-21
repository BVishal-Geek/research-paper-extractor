"""Pydantic schema for paper-level reportability extraction.

Each `_E` field is a binary verdict: 1 = the rubric is satisfied, 0 = it is not.
Each `_reason` field is a short paraphrased justification (1-2 sentences).
Each `_evidence` field is an optional verbatim quote from the paper that
supports the verdict. Evidence makes hallucinations easy to spot during
evaluation and is OK to leave empty when the verdict is 0.

The rubric strings in `Field(description=...)` are surfaced to the LLM via
`model.model_json_schema()`, so they double as the per-field instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal[0, 1]


class ExtractionResult(BaseModel):
    """One row of reportability assessment for a single paper."""

    paper_title: str = Field(
        description="The title of the paper as written by the authors.",
    )

    condition_E: Verdict = Field(
        description=(
            "1 if the experimental or responder cohort is clearly defined "
            "(disease, stage, biomarker status, or other inclusion criteria "
            "are explicit). 0 otherwise."
        ),
    )
    condition_E_reason: str = Field(
        description="1-2 sentence justification for the condition_E verdict.",
    )
    condition_E_evidence: str = Field(
        default="",
        description=(
            "Verbatim quote from the paper supporting condition_E. "
            "Empty string if the verdict is 0 or no clean quote exists."
        ),
    )

    N_E: Verdict = Field(
        description=(
            "1 if the sample size or data volume for the experimental group "
            "is clearly stated as a number. 0 otherwise."
        ),
    )
    N_E_reason: str = Field(
        description="1-2 sentence justification for the N_E verdict.",
    )
    N_E_evidence: str = Field(default="", description="Verbatim supporting quote, or empty.")

    dataset_E: Verdict = Field(
        description=(
            "1 if the experimental group data source is clearly specified AND "
            "accessible (e.g. GEO accession, TCGA project, dbGaP, a named public "
            "repository, or a DOI). 0 if the data is private, undisclosed, or "
            "only vaguely described."
        ),
    )
    dataset_E_reason: str = Field(
        description="1-2 sentence justification for the dataset_E verdict.",
    )
    dataset_E_evidence: str = Field(default="", description="Verbatim supporting quote, or empty.")

    intervention_E: Verdict = Field(
        description=(
            "1 if the experimental treatment, intervention, or biological "
            "condition applied to the cohort is clearly described (drug name, "
            "dose, regimen, perturbation, etc.). 0 otherwise."
        ),
    )
    intervention_E_reason: str = Field(
        description="1-2 sentence justification for the intervention_E verdict.",
    )
    intervention_E_evidence: str = Field(default="", description="Verbatim supporting quote, or empty.")

    pr_endpoint_E: Verdict = Field(
        description=(
            "1 if a clear primary outcome or response endpoint is defined for "
            "the experimental group (e.g. overall survival, progression-free "
            "survival, objective response rate, a named biomarker change). "
            "0 otherwise."
        ),
    )
    pr_endpoint_E_reason: str = Field(
        description="1-2 sentence justification for the pr_endpoint_E verdict.",
    )
    pr_endpoint_E_evidence: str = Field(default="", description="Verbatim supporting quote, or empty.")

    R_criteria_E: Verdict = Field(
        description=(
            "1 if explicit criteria defining responders in the experimental "
            "group are stated (e.g. RECIST 1.1, iRECIST, a numeric threshold, "
            "or a named scoring system). 0 otherwise."
        ),
    )
    R_criteria_E_reason: str = Field(
        description="1-2 sentence justification for the R_criteria_E verdict.",
    )
    R_criteria_E_evidence: str = Field(default="", description="Verbatim supporting quote, or empty.")
