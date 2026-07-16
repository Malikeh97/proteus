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
    def generate(self, prompt: str, **kwargs) -> Generation: ...

    def generate_text(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs).text
