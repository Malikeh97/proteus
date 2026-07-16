from __future__ import annotations

from abc import ABC, abstractmethod

from proteus.utils.config import JudgeConfig


class Judge(ABC):
    """Adjudicates whether a response constitutes a jailbreak. Fixed across all
    conditions, and never shared with the attacker's reward model -- otherwise
    the attacker learns to fool the judge rather than the defense."""

    def __init__(self, config: JudgeConfig) -> None:
        self._config = config

    @property
    def judge_id(self) -> str:
        return self._config.judge_id

    @abstractmethod
    def judge(self, prompt: str, response: str) -> int:
        """1 if the response is a jailbreak, else 0."""

    def judge_batch(self, pairs: list[tuple[str, str]]) -> list[int]:
        return [self.judge(p, r) for p, r in pairs]


class RefusalDetector(ABC):
    """Separate from the judge: Help(c) counts refusals on benign traffic, and a
    model that declines a benign XSTest prompt has over-refused even though the
    judge would (correctly) call the response 'not a jailbreak'."""

    @abstractmethod
    def is_refusal(self, prompt: str, response: str) -> bool: ...
