"""LLM client interface.

Every concrete client returns a raw JSON string from chat_json — schema
validation lives in the extraction layer, not here. Keeping the surface area
to one method makes it trivial to add a third provider later.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Minimal interface every LLM client must implement."""

    @abstractmethod
    def chat_json(self, system: str, user: str) -> str:
        """Send a chat request and return the model's raw JSON string output.

        Implementations are responsible for instructing the model to emit
        JSON only (via provider-specific format/response_format flags).
        """
        ...
