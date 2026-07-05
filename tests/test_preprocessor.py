"""Tests for the Preprocessor.

Uses a hand-written minimal PMC XML fixture so the test is fast, deterministic,
and does not depend on any real paper being on disk. pytest's tmp_path
fixture keeps input and output isolated per test — no test writes to your
actual data/ directory.
"""

import json
from pathlib import Path

import pytest

from rpextractor.ingestion.preprocessor import Preprocessor


# Minimal PMC-style XML exercising every field XMLParser.parse() reads.
FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC9999999</article-id>
      <article-id pub-id-type="pmid">12345678</article-id>
      <article-id pub-id-type="doi">10.1234/test.9999</article-id>
      <title-group>
        <article-title>A Test Paper on Oncology Data Extraction</article-title>
      </title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>Doe</surname><given-names>Jane</given-names></name>
        </contrib>
      </contrib-group>
      <pub-date><year>2026</year></pub-date>
      <kwd-group><kwd>oncology</kwd><kwd>machine learning</kwd></kwd-group>
      <abstract>
        <p>Background paragraph describing the study.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title><p>Intro text here.</p></sec>
    <sec><title>Methods</title><p>Methods text here with N=100 patients.</p></sec>
    <sec><title>Results</title><p>Results text here.</p></sec>
  </body>
</article>
"""


@pytest.fixture
def raw_and_processed_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create isolated raw/ and processed/ dirs with one XML fixture inside."""
    raw = tmp_path / "raw" / "2026-07-02_10-00-00"
    raw.mkdir(parents=True)
    (raw / "PMC9999999.xml").write_text(FIXTURE_XML, encoding="utf-8")

    processed = tmp_path / "processed"
    return tmp_path / "raw", processed


def test_writes_json_to_processed(raw_and_processed_dirs):
    """A single XML in raw/ produces one JSON in processed/ with expected fields."""
    raw, processed = raw_and_processed_dirs

    summary = Preprocessor(input_dir=raw, output_dir=processed).run()

    # Summary counts what happened.
    assert summary == {"total": 1, "success": 1, "skipped": 0, "failed": 0}

    # File actually exists at the expected path.
    output_file = processed / "PMC9999999.json"
    assert output_file.exists(), f"Expected JSON at {output_file}"

    # File contents are valid JSON and hold the parsed fields.
    parsed = json.loads(output_file.read_text(encoding="utf-8"))

    assert parsed["metadata"]["pmcid"] == "PMC9999999"
    assert parsed["metadata"]["pmid"] == "12345678"
    assert parsed["metadata"]["title"] == "A Test Paper on Oncology Data Extraction"
    assert parsed["metadata"]["year"] == "2026"
    assert "oncology" in parsed["metadata"]["keywords"]

    # At least one section came through with real content.
    assert "N=100 patients" in parsed["methods"]


def test_skips_when_json_already_exists(raw_and_processed_dirs):
    """Second run of the preprocessor is idempotent — no re-writes."""
    raw, processed = raw_and_processed_dirs

    first = Preprocessor(input_dir=raw, output_dir=processed).run()
    assert first["success"] == 1

    second = Preprocessor(input_dir=raw, output_dir=processed).run()
    assert second == {"total": 1, "success": 0, "skipped": 1, "failed": 0}


def test_empty_input_dir_returns_zeros(tmp_path: Path):
    """Nothing to process → summary reports zeros, no crash."""
    raw = tmp_path / "raw"
    raw.mkdir()
    processed = tmp_path / "processed"

    summary = Preprocessor(input_dir=raw, output_dir=processed).run()
    assert summary == {"total": 0, "success": 0, "skipped": 0, "failed": 0}
    assert processed.exists()  # output dir is still created
