"""Attacker strategies beta: H^att -> X.

The interface *is* the information constraint. `refine` receives the prompt, the
response, and the judgment -- never the coverage c_t, never the drawn
configuration q_t, never which wrapper fired. That is the projection rho of
Sec. 3: an attacker at an API issues prompts and reads outputs, and must infer
the deployed configuration from the outcomes its own probes elicit.

Do not widen this signature to pass q or c through. Doing so silently converts
the game into the leaked-commitment relaxation and every ASR becomes an upper
bound rather than a measurement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from proteus.utils.config import AttackerConfig


class Attacker(ABC):
    def __init__(self, config: AttackerConfig) -> None:
        self._config = config
        self._tokens_spent = 0

    @property
    def attacker_id(self) -> str:
        return self._config.attacker_id

    @property
    def max_steps(self) -> int:
        return self._config.max_steps

    @property
    def params(self) -> dict:
        return self._config.params

    @property
    def tokens_spent(self) -> int:
        """Cumulative attacker tokens, for the compute-cost metric."""
        return self._tokens_spent

    def reset(self, coverage_id: str | None = None) -> None:
        """Called once per round: the attacker re-optimises from scratch against
        each newly committed coverage. Subclasses that carry state across
        prompts within a round should clear it here."""
        self._tokens_spent = 0

    @abstractmethod
    def initialize(self, base_prompt: str) -> str:
        """The first prompt x_0 for this target."""

    @abstractmethod
    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        """x_t given the attacker's own history only."""
