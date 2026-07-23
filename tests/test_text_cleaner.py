"""Tests for clean_text — used by XMLParser on every extracted section."""

from rpextractor.utils.text_cleaner import clean_text


def test_removes_bracketed_numeric_citations():
    assert clean_text("This work [1] extends prior work [2,3].") == "This work extends prior work ."


def test_removes_parenthesized_numeric_citations():
    assert clean_text("Prior studies (1) and (2,3) agree.") == "Prior studies and agree."


def test_removes_ranged_citations():
    assert clean_text("See references [4-7].") == "See references ."


def test_collapses_multiple_whitespace():
    assert clean_text("too    many\t\tspaces\nhere") == "too many spaces here"


def test_strips_leading_and_trailing_whitespace():
    assert clean_text("   padded   ") == "padded"


def test_returns_empty_for_empty_input():
    assert clean_text("") == ""


def test_leaves_regular_text_untouched():
    assert clean_text("The p53 gene regulates cell cycle.") == "The p53 gene regulates cell cycle."


def test_does_not_strip_non_citation_brackets():
    # A bracketed word (not a number) is not a citation and must survive.
    text = "Results [preliminary] were reported."
    assert clean_text(text) == text
