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

    def run(self):
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
