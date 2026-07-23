"""Tests for load_pmcids.

The loader is the single canonicalization point for PMCID input: everything
downstream (Downloader, Preprocessor filenames, extracted output filenames)
assumes the 'PMC<digits>' form. These tests lock the normalization contract
so a well-meaning tweak doesn't quietly desync formats.
"""
# pylint: disable=use-implicit-booleaness-not-comparison
# `== []` is deliberate in test assertions for readability.

from pathlib import Path

from rpextractor.ingestion.pmcid_loader import load_pmcids


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "pmcids.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_newline_separated_pmcids_with_prefix(tmp_path):
    path = _write(tmp_path, "PMC12345\nPMC67890\n")
    assert load_pmcids(path) == ["PMC12345", "PMC67890"]


def test_space_separated_pmcids_on_one_line(tmp_path):
    path = _write(tmp_path, "PMC12345 PMC67890 PMC11111")
    assert load_pmcids(path) == ["PMC12345", "PMC67890", "PMC11111"]


def test_mixed_spaces_and_newlines(tmp_path):
    path = _write(tmp_path, "PMC12345 PMC67890\nPMC11111\n\nPMC22222")
    assert load_pmcids(path) == ["PMC12345", "PMC67890", "PMC11111", "PMC22222"]


def test_bare_numeric_ids_are_prefixed(tmp_path):
    path = _write(tmp_path, "12345\n67890\n")
    assert load_pmcids(path) == ["PMC12345", "PMC67890"]


def test_lowercase_pmc_prefix_is_normalized(tmp_path):
    path = _write(tmp_path, "pmc12345\nPMC67890\npMc11111")
    assert load_pmcids(path) == ["PMC12345", "PMC67890", "PMC11111"]


def test_mixed_prefixed_and_bare_ids(tmp_path):
    path = _write(tmp_path, "PMC12345 67890 pmc11111")
    assert load_pmcids(path) == ["PMC12345", "PMC67890", "PMC11111"]


def test_duplicates_are_removed_preserving_first_occurrence(tmp_path):
    path = _write(tmp_path, "PMC12345 67890 12345 PMC67890 PMC12345")
    # PMC12345 appears three ways; must appear once at first-seen position.
    assert load_pmcids(path) == ["PMC12345", "PMC67890"]


def test_non_numeric_junk_is_silently_dropped(tmp_path):
    # Comments, headers, and stray words must not become fake PMCIDs.
    path = _write(tmp_path, "# my ground truth\npmcid\nPMC12345\nfoo\n67890")
    assert load_pmcids(path) == ["PMC12345", "PMC67890"]


def test_empty_file_returns_empty_list(tmp_path):
    path = _write(tmp_path, "")
    assert load_pmcids(path) == []


def test_whitespace_only_file_returns_empty_list(tmp_path):
    path = _write(tmp_path, "   \n\n\t  \n")
    assert load_pmcids(path) == []


def test_accepts_path_as_string_or_path_object(tmp_path):
    path = _write(tmp_path, "PMC12345")
    assert load_pmcids(str(path)) == ["PMC12345"]
    assert load_pmcids(path) == ["PMC12345"]
