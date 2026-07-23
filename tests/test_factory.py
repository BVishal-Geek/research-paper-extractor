"""Tests for get_llm_client — the provider factory.

We monkeypatch load_yaml (so we don't read the real config), OllamaClient
(so we don't spin up a real httpx client), and OpenAIClient (so we don't
require OPENAI_API_KEY). This isolates the factory's routing logic.
"""

import pytest

from rpextractor.llm import factory


def _install_stub_clients(monkeypatch):
    """Replace both concrete clients with capture stubs. Returns the call log."""
    captured: list[tuple[str, dict]] = []

    def make_stub(name: str):
        def _stub(**kwargs):
            captured.append((name, kwargs))
            return f"stub-{name}"
        return _stub

    monkeypatch.setattr(factory, "OllamaClient", make_stub("ollama"))
    monkeypatch.setattr(factory, "OpenAIClient", make_stub("openai"))
    return captured


def test_explicit_ollama_provider_calls_ollama_client(monkeypatch):
    captured = _install_stub_clients(monkeypatch)
    monkeypatch.setattr(factory, "load_yaml", lambda _: {
        "provider": "openai",  # config default is openai — should be overridden
        "ollama": {"model": "qwen3.5:4b", "host": "http://localhost:11434"},
        "openai": {"model": "gpt-4o-mini"},
    })

    result = factory.get_llm_client(provider="ollama")

    assert result == "stub-ollama"
    assert captured == [("ollama", {
        "model": "qwen3.5:4b",
        "host": "http://localhost:11434",
        "timeout": 120,
        "options": {},
    })]


def test_explicit_openai_provider_calls_openai_client(monkeypatch):
    captured = _install_stub_clients(monkeypatch)
    monkeypatch.setattr(factory, "load_yaml", lambda _: {
        "provider": "ollama",  # config default is ollama — should be overridden
        "openai": {
            "model": "gpt-4o-mini",
            "timeout": 60,
            "temperature": 0.0,
            "max_cost_usd_per_run": 5.0,
        },
    })

    result = factory.get_llm_client(provider="openai")

    assert result == "stub-openai"
    assert captured == [("openai", {
        "model": "gpt-4o-mini",
        "timeout": 60,
        "temperature": 0.0,
        "max_cost_usd_per_run": 5.0,
    })]


def test_default_provider_from_config_when_no_arg(monkeypatch):
    captured = _install_stub_clients(monkeypatch)
    monkeypatch.setattr(factory, "load_yaml", lambda _: {
        "provider": "openai",
        "openai": {"model": "gpt-4o-mini"},
    })

    factory.get_llm_client()  # no provider arg → falls back to config

    assert captured[0][0] == "openai"


def test_falls_back_to_ollama_when_config_missing_provider(monkeypatch):
    captured = _install_stub_clients(monkeypatch)
    monkeypatch.setattr(factory, "load_yaml", lambda _: {
        "ollama": {"model": "qwen3.5:4b"},
    })

    factory.get_llm_client()

    assert captured[0][0] == "ollama"


def test_unknown_provider_raises(monkeypatch):
    _install_stub_clients(monkeypatch)
    monkeypatch.setattr(factory, "load_yaml", lambda _: {"provider": "gemini"})

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        factory.get_llm_client()
