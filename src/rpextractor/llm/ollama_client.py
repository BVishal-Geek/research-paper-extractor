"""Ollama LLM client.

Uses the official `ollama` Python package. Forces JSON output via
format='json' so the model is constrained at sampling time.
"""

from ollama import Client

from rpextractor.llm.base import BaseLLMClient
from rpextractor.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaClient(BaseLLMClient):
    """Thin wrapper around ollama.Client.chat with format='json'."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 120,
        options: dict | None = None,
    ):
        self.model = model
        self.options = options or {}
        self._client = Client(host=host, timeout=timeout)
        logger.info(f"OllamaClient ready: model={model}, host={host}")

    def chat_json(self, system: str, user: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options=self.options,
        )
        return response["message"]["content"]
