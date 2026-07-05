"""Tests for build_paper_text.

Pure data-transformation tests — no I/O, no LLM, no fixtures needed.
"""

from rpextractor.extraction.input_builder import build_paper_text


def test_includes_title_and_default_sections():
    parsed = {
        "metadata": {"title": "Some Paper"},
        "abstract": "Abstract text.",
        "introduction": "Intro text.",
        "methods": "Methods text.",
        "results": "Results text.",
        "data_availability": "GEO GSE12345",
        "discussion": "Should be dropped.",
    }
    text = build_paper_text(parsed)

    assert "TITLE:\nSome Paper" in text
    assert "ABSTRACT:\nAbstract text." in text
    assert "INTRODUCTION:\nIntro text." in text
    assert "METHODS:\nMethods text." in text
    assert "RESULTS:\nResults text." in text
    assert "DATA_AVAILABILITY:\nGEO GSE12345" in text
    # Discussion is not in DEFAULT_SECTIONS, so it must be dropped.
    assert "DISCUSSION" not in text


def test_skips_empty_and_whitespace_sections():
    parsed = {
        "metadata": {"title": "T"},
        "abstract": "",
        "methods": "M",
        "results": "   ",  # only whitespace
    }
    text = build_paper_text(parsed)

    assert "METHODS:\nM" in text
    assert "ABSTRACT" not in text
    assert "RESULTS" not in text


def test_missing_title_still_works():
    parsed = {"metadata": {}, "methods": "M"}
    text = build_paper_text(parsed)

    assert "TITLE" not in text
    assert "METHODS:\nM" in text


def test_empty_dict_returns_empty_string():
    assert build_paper_text({}) == ""


def test_custom_sections_argument_respected():
    parsed = {"metadata": {"title": "T"}, "methods": "M", "results": "R"}
    text = build_paper_text(parsed, sections=("results",))

    assert "RESULTS:\nR" in text
    assert "METHODS" not in text
