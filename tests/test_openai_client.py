"""Tests for OpenAIClient.

Focus is the cost guard — the piece that prevents an accidental large run
from silently overrunning a budget. Every test replaces `openai.OpenAI`
with a stub so no real API call is ever made.
"""
# pylint: disable=protected-access
# The cost tracker is intentionally private (_cumulative_cost_usd); tests
# need to inspect it directly to verify the guard's accounting.

from types import SimpleNamespace

import pytest

from rpextractor.llm import openai_client as oc_mod
from rpextractor.llm.openai_client import CostCeilingExceeded, OpenAIClient


class _StubOpenAI:
    """Stand-in for openai.OpenAI that returns pre-baked responses."""

    def __init__(self, responses: list[tuple[str, int, int]]):
        # (content_string, prompt_tokens, completion_tokens)
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("_StubOpenAI ran out of responses")
        content, in_tok, out_tok = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        )


@pytest.fixture
def stub_openai(monkeypatch):
    """Patch openai.OpenAI at the module import site so OpenAIClient uses the stub."""
    stub_holder = {}

    def factory(**_kwargs):
        return stub_holder["instance"]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(oc_mod, "OpenAI", factory)

    def install(responses):
        stub_holder["instance"] = _StubOpenAI(responses)
        return stub_holder["instance"]

    return install


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY must be set"):
        OpenAIClient(model="gpt-4o-mini")


def test_returns_json_string_from_response(stub_openai):
    stub_openai([('{"ok": 1}', 100, 5)])
    client = OpenAIClient(model="gpt-4o-mini", max_cost_usd_per_run=1.0)

    out = client.chat_json("system", "user")

    assert out == '{"ok": 1}'


def test_cost_accumulates_across_calls(stub_openai):
    stub = stub_openai([
        ('{"a": 1}', 1_000_000, 0),  # $0.15 in prompt cost
        ('{"a": 2}', 0, 1_000_000),  # $0.60 in completion cost
    ])
    client = OpenAIClient(model="gpt-4o-mini", max_cost_usd_per_run=10.0)

    client.chat_json("s", "u")
    client.chat_json("s", "u")

    assert len(stub.calls) == 2
    assert client._cumulative_cost_usd == pytest.approx(0.15 + 0.60)


def test_cost_ceiling_blocks_next_call(stub_openai):
    # First call spends more than the ceiling; second should be refused.
    stub = stub_openai([
        ('{"a": 1}', 10_000_000, 0),  # $1.50 — well past the $0.50 ceiling
        ('{"a": 2}', 100, 5),
    ])
    client = OpenAIClient(model="gpt-4o-mini", max_cost_usd_per_run=0.50)

    client.chat_json("s", "u")  # succeeds but pushes cumulative over ceiling

    with pytest.raises(CostCeilingExceeded, match="ceiling is"):
        client.chat_json("s", "u")

    assert len(stub.calls) == 1  # second attempt aborted before the API call


def test_unknown_model_still_works_but_skips_cost_tracking(stub_openai):
    """Cost guard doesn't crash on an unknown model — it warns and lets calls through."""
    stub = stub_openai([('{"a": 1}', 1_000_000, 1_000_000)])
    client = OpenAIClient(model="gpt-99-imaginary", max_cost_usd_per_run=0.01)

    client.chat_json("s", "u")

    assert len(stub.calls) == 1
    # No pricing entry → cost stays at 0, guard cannot trigger for this model.
    assert client._cumulative_cost_usd == 0.0
