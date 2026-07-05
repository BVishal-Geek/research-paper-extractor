"""OpenAI LLM client.

Uses response_format={"type": "json_object"} so the API rejects non-JSON
outputs. Tracks cumulative spend across calls and refuses new requests once
max_cost_usd_per_run is exceeded — a guard so an accidental large batch run
cannot rack up an unbounded bill.

API key is read from OPENAI_API_KEY in the environment (.env).
"""

import os

from openai import OpenAI

from rpextractor.llm.base import BaseLLMClient
from rpextractor.utils.logger import get_logger

logger = get_logger(__name__)


# Per-million-token pricing in USD. Update when OpenAI changes prices.
# Only models we expect to use are listed; unknown models cost-track as 0
# and log a warning instead of crashing.
_PRICING_USD_PER_MTOK = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
}


class CostCeilingExceeded(RuntimeError):
    """Raised when a chat call would push cumulative spend over the ceiling."""


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        timeout: int = 60,
        temperature: float = 0.0,
        max_cost_usd_per_run: float = 5.0,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env to use OpenAIClient")

        self.model = model
        self.temperature = temperature
        self.max_cost_usd_per_run = max_cost_usd_per_run
        self._cumulative_cost_usd = 0.0
        self._client = OpenAI(api_key=api_key, timeout=timeout)

        if model not in _PRICING_USD_PER_MTOK:
            logger.warning(
                f"No pricing entry for model={model}; cost guard will not protect this run"
            )

        logger.info(
            f"OpenAIClient ready: model={model}, ceiling=${max_cost_usd_per_run:.2f}"
        )

    def chat_json(self, system: str, user: str) -> str:
        if self._cumulative_cost_usd >= self.max_cost_usd_per_run:
            raise CostCeilingExceeded(
                f"Spent ${self._cumulative_cost_usd:.4f} this run; "
                f"ceiling is ${self.max_cost_usd_per_run:.2f}"
            )

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        usage = response.usage
        if usage and self.model in _PRICING_USD_PER_MTOK:
            pricing = _PRICING_USD_PER_MTOK[self.model]
            call_cost = (
                usage.prompt_tokens * pricing["input"]
                + usage.completion_tokens * pricing["output"]
            ) / 1_000_000
            self._cumulative_cost_usd += call_cost
            logger.info(
                f"OpenAI call: in={usage.prompt_tokens}, out={usage.completion_tokens}, "
                f"cost=${call_cost:.4f}, total=${self._cumulative_cost_usd:.4f}"
            )

        return response.choices[0].message.content
