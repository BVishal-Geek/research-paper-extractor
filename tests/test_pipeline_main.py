"""Tests for the pipeline orchestrator.

We monkeypatch the three stage classes and the LLM factory so no real
downloads, parsing, or LLM calls happen. The tests verify wiring
(each stage is invoked when it should be), skip flags, and provider
resolution (CLI alias → factory slug).
"""
# The `stub_stages` fixture works by side effect (monkey-patching module
# globals); some tests never read its return value directly, so the arg
# looks unused to pylint even though removing it breaks the test.
# `x == []` / `x == {}` are used deliberately for readable empty-collection
# assertions in test code.
# pylint: disable=unused-argument,use-implicit-booleaness-not-comparison

import pytest

from rpextractor.pipeline import main as pipeline_main


class _FakeStage:
    """Records .run() invocations and returns a stub summary dict."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        _FakeStage.instances.append(self)

    def run(self, **_kwargs):
        # **_kwargs makes this a drop-in stand-in for both Downloader.run()
        # (which now takes pmids=) and the arg-less Preprocessor/Extractor.run().
        _FakeStage.calls.append(type(self).__name__)
        return {"total": 1, "success": 1, "skipped": 0, "failed": 0}


@pytest.fixture(autouse=True)
def _reset_stage_state():
    """Fresh call log per test."""
    _FakeStage.instances = []
    _FakeStage.calls = []
    yield


@pytest.fixture
def stub_stages(monkeypatch):
    """Replace all three stage classes and the LLM factory with fakes."""
    class FakeDownloader(_FakeStage):
        """Stub Downloader."""

    class FakePreprocessor(_FakeStage):
        """Stub Preprocessor."""

    class FakeExtractor(_FakeStage):
        """Stub Extractor."""

    fake_client_calls = []

    def fake_get_llm_client(provider: str):
        fake_client_calls.append(provider)
        return object()

    monkeypatch.setattr(pipeline_main, "Downloader", FakeDownloader)
    monkeypatch.setattr(pipeline_main, "Preprocessor", FakePreprocessor)
    monkeypatch.setattr(pipeline_main, "Extractor", FakeExtractor)
    monkeypatch.setattr(pipeline_main, "get_llm_client", fake_get_llm_client)

    return fake_client_calls


def test_runs_all_three_stages_by_default(stub_stages):
    summary = pipeline_main.run_pipeline(provider="ollama")

    assert _FakeStage.calls == ["FakeDownloader", "FakePreprocessor", "FakeExtractor"]
    assert set(summary.keys()) == {"download", "preprocess", "extract"}
    assert stub_stages == ["ollama"]


def test_skip_download(stub_stages):
    summary = pipeline_main.run_pipeline(provider="ollama", skip_download=True)

    assert _FakeStage.calls == ["FakePreprocessor", "FakeExtractor"]
    assert "download" not in summary


def test_skip_preprocess(stub_stages):
    summary = pipeline_main.run_pipeline(provider="ollama", skip_preprocess=True)

    assert _FakeStage.calls == ["FakeDownloader", "FakeExtractor"]
    assert "preprocess" not in summary


def test_skip_extract_never_touches_llm(stub_stages):
    summary = pipeline_main.run_pipeline(provider="ollama", skip_extract=True)

    assert _FakeStage.calls == ["FakeDownloader", "FakePreprocessor"]
    assert "extract" not in summary
    assert stub_stages == []


def test_skip_all_returns_empty_summary(stub_stages):
    summary = pipeline_main.run_pipeline(
        provider="ollama",
        skip_download=True,
        skip_preprocess=True,
        skip_extract=True,
    )

    assert _FakeStage.calls == []
    assert summary == {}


def test_cli_local_model_resolves_to_ollama(stub_stages):
    exit_code = pipeline_main.main(
        ["--provider", "local_model", "--skip-download", "--skip-preprocess"]
    )

    assert exit_code == 0
    assert stub_stages == ["ollama"]


def test_cli_openai_passes_through(stub_stages):
    exit_code = pipeline_main.main(
        ["--provider", "openai", "--skip-download", "--skip-preprocess"]
    )

    assert exit_code == 0
    assert stub_stages == ["openai"]


def test_cli_default_provider_is_local_model(stub_stages):
    pipeline_main.main(["--skip-download", "--skip-preprocess"])

    assert stub_stages == ["ollama"]


def test_cli_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        pipeline_main.parse_args(["--provider", "gemini"])


# ─────────────────────────────────────────────────────────────────────────────
# --processed-batch flag: parsing + passthrough to Extractor
# ─────────────────────────────────────────────────────────────────────────────


def test_processed_batch_defaults_to_none():
    args = pipeline_main.parse_args([])
    assert args.processed_batch is None


def test_processed_batch_accepts_date_string():
    args = pipeline_main.parse_args(["--processed-batch", "2026-07-22"])
    assert args.processed_batch == "2026-07-22"


def test_processed_batch_accepts_latest_and_all():
    assert pipeline_main.parse_args(["--processed-batch", "latest"]).processed_batch == "latest"
    assert pipeline_main.parse_args(["--processed-batch", "all"]).processed_batch == "all"


class _CaptureExtractor(_FakeStage):
    """Records the input_batch it was constructed with."""

    seen_batches: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _CaptureExtractor.seen_batches.append(kwargs.get("input_batch"))


def test_input_batch_reaches_extractor(monkeypatch):
    """--processed-batch value must flow through run_pipeline to Extractor kwargs."""
    _CaptureExtractor.seen_batches = []
    monkeypatch.setattr(pipeline_main, "Extractor", _CaptureExtractor)
    monkeypatch.setattr(pipeline_main, "get_llm_client", lambda provider: object())

    exit_code = pipeline_main.main([
        "--skip-download",
        "--skip-preprocess",
        "--processed-batch", "2026-07-22",
    ])

    assert exit_code == 0
    assert _CaptureExtractor.seen_batches == ["2026-07-22"]


def test_input_batch_defaults_to_none_at_extractor(monkeypatch):
    """No --processed-batch → Extractor receives input_batch=None (all)."""
    _CaptureExtractor.seen_batches = []
    monkeypatch.setattr(pipeline_main, "Extractor", _CaptureExtractor)
    monkeypatch.setattr(pipeline_main, "get_llm_client", lambda provider: object())

    pipeline_main.main(["--skip-download", "--skip-preprocess"])

    assert _CaptureExtractor.seen_batches == [None]


# ─────────────────────────────────────────────────────────────────────────────
# --pmcids flag: parsing, loader wiring, Downloader passthrough
# ─────────────────────────────────────────────────────────────────────────────


def test_pmcids_defaults_to_none():
    args = pipeline_main.parse_args([])
    assert args.pmcids is None


def test_pmcids_accepts_path_arg():
    args = pipeline_main.parse_args(["--pmcids", "/tmp/my-list.txt"])
    assert args.pmcids == "/tmp/my-list.txt"


class _CaptureDownloader(_FakeStage):
    """Records the pmids list passed to run()."""

    seen_pmids: list = []

    def run(self, **kwargs):  # match Downloader.run signature (query=, pmids=)
        _CaptureDownloader.seen_pmids.append(kwargs.get("pmids"))
        _FakeStage.calls.append(type(self).__name__)
        return {
            "total": len(kwargs.get("pmids") or []),
            "success": 0,
            "skipped": 0,
            "failed": 0,
        }


def test_pmcids_file_is_loaded_and_passed_to_downloader(monkeypatch, tmp_path):
    """--pmcids <file> → load_pmcids → Downloader.run(pmids=...)."""
    pmcids_file = tmp_path / "list.txt"
    pmcids_file.write_text("PMC12345 pmc67890\n11111\n", encoding="utf-8")

    _CaptureDownloader.seen_pmids = []
    monkeypatch.setattr(pipeline_main, "Downloader", _CaptureDownloader)
    monkeypatch.setattr(pipeline_main, "Preprocessor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "Extractor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "get_llm_client", lambda provider: object())

    exit_code = pipeline_main.main(["--pmcids", str(pmcids_file)])

    assert exit_code == 0
    # Loader normalizes all three tokens; Downloader saw exactly that list.
    assert _CaptureDownloader.seen_pmids == [["PMC12345", "PMC67890", "PMC11111"]]


def test_downloader_receives_none_when_no_pmcids_flag(monkeypatch):
    """No --pmcids → Downloader.run(pmids=None), search runs as usual."""
    _CaptureDownloader.seen_pmids = []
    monkeypatch.setattr(pipeline_main, "Downloader", _CaptureDownloader)
    monkeypatch.setattr(pipeline_main, "Preprocessor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "Extractor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "get_llm_client", lambda provider: object())

    pipeline_main.main([])

    assert _CaptureDownloader.seen_pmids == [None]


# ─────────────────────────────────────────────────────────────────────────────
# --max-downloads flag: parses to int, passes through to Downloader
# ─────────────────────────────────────────────────────────────────────────────


def test_max_downloads_defaults_to_none():
    args = pipeline_main.parse_args([])
    assert args.max_downloads is None


def test_max_downloads_parses_as_int():
    args = pipeline_main.parse_args(["--max-downloads", "7"])
    assert args.max_downloads == 7


def test_max_downloads_rejects_non_integer():
    with pytest.raises(SystemExit):
        pipeline_main.parse_args(["--max-downloads", "seven"])


class _CaptureDownloaderMax(_FakeStage):
    """Records the max_results kwarg passed to run()."""

    seen_max_results: list = []

    def run(self, **kwargs):
        _CaptureDownloaderMax.seen_max_results.append(kwargs.get("max_results"))
        _FakeStage.calls.append(type(self).__name__)
        return {"total": 0, "success": 0, "skipped": 0, "failed": 0}


def test_max_downloads_reaches_downloader(monkeypatch):
    """--max-downloads 5 → Downloader.run(max_results=5)."""
    _CaptureDownloaderMax.seen_max_results = []
    monkeypatch.setattr(pipeline_main, "Downloader", _CaptureDownloaderMax)
    monkeypatch.setattr(pipeline_main, "Preprocessor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "Extractor", _FakeStage)
    monkeypatch.setattr(pipeline_main, "get_llm_client", lambda provider: object())

    pipeline_main.main(["--max-downloads", "5"])

    assert _CaptureDownloaderMax.seen_max_results == [5]
