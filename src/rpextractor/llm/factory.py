"""LLM client factory.

Reads configs/llm.yaml and returns the configured concrete client. Keeps the
extraction layer ignorant of which provider is in use.
"""

from rpextractor.llm.base import BaseLLMClient
from rpextractor.llm.ollama_client import OllamaClient
from rpextractor.llm.openai_client import OpenAIClient
from rpextractor.utils.config import load_yaml
from rpextractor.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """Return a configured LLM client.

    Args:
        provider: 'ollama' or 'openai'. If None, falls back to the value in
            configs/llm.yaml.
    """
    config = load_yaml("llm.yaml")
    provider = provider or config.get("provider", "ollama")

    if provider == "ollama":
        cfg = config.get("ollama", {})
        return OllamaClient(
            model=cfg["model"],
            host=cfg.get("host", "http://localhost:11434"),
            timeout=cfg.get("timeout", 120),
            options=cfg.get("options", {}),
        )

    if provider == "openai":
        cfg = config.get("openai", {})
        return OpenAIClient(
            model=cfg["model"],
            timeout=cfg.get("timeout", 60),
            temperature=cfg.get("temperature", 0.0),
            max_cost_usd_per_run=cfg.get("max_cost_usd_per_run", 5.0),
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
