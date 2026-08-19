from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from proteus.utils.config import ModelConfig


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseModel(ABC):
    """A servable model M_i. Implement generate() and you are done."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def params_b(self) -> float | None:
        return self._config.params_b

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Generation:
        """kwargs override the config's generation settings for this call only.
        `system_prompt` is the one non-sampling key: it goes into a system turn of
        the chat template rather than being prepended to the user message, which
        is what instruct models are tuned to obey. Served models never pass it --
        a system prompt is a defense, and defenses belong in wrappers, where the
        menu can see them."""

    def generate_text(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs).text
