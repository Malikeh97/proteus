"""Defense wrappers w in the library W.

A wrapper sees the request on its way in and/or the response on its way out and
may allow, transform, or refuse. Both hooks default to pass-through, so an
input-only classifier overrides on_input and nothing else.

This covers all three mechanism families in the proposal: classifiers refuse,
prompt-level interventions transform, and smoothing wrappers do both.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from proteus.utils.config import WrapperConfig


@dataclass
class Verdict:
    allow: bool
    text: str  # the (possibly transformed) prompt or response
    reason: str = ""
    tokens: int = 0  # tokens the wrapper itself spent, for FLOP accounting

    @classmethod
    def passthrough(cls, text: str) -> Verdict:
        return cls(allow=True, text=text)

    @classmethod
    def refuse(cls, reason: str, tokens: int = 0) -> Verdict:
        return cls(allow=False, text="", reason=reason, tokens=tokens)


class Wrapper(ABC):
    def __init__(self, config: WrapperConfig) -> None:
        self._config = config

    @property
    def wrapper_id(self) -> str:
        return self._config.wrapper_id

    @property
    def config(self) -> WrapperConfig:
        return self._config

    @property
    def runs_on_input(self) -> bool:
        return self._config.stage in ("input", "both")

    @property
    def runs_on_output(self) -> bool:
        return self._config.stage in ("output", "both")

    @property
    def params_b(self) -> float | None:
        return self._config.params_b

    def on_input(self, prompt: str) -> Verdict:
        return Verdict.passthrough(prompt)

    def on_output(self, prompt: str, response: str) -> Verdict:
        return Verdict.passthrough(response)
