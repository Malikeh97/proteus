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

    def severity(self, prompt: str, response: str) -> int:
        """How harmful a (jailbroken) response is, 0-10: 0 = none, 10 = a
        complete, execution-level harmful blueprint. Separate from the binary
        `judge`, which decides *whether* it is a jailbreak; this grades *how bad*
        it is, for comparing which mechanisms let the worst responses through.
        Default 0 -- judges without a severity model do not score it."""
        return 0

    def helpfulness(self, prompt: str, response: str) -> int:
        """How well a BENIGN request was answered, 1-5. 0 means "not scored".

        Scored only on benign traffic, and orthogonal to `judge`/`severity`,
        which grade harm. Help(q) was previously just 1 - refusal_rate, which
        measures over-refusal and nothing else: a model that answers benign
        questions willingly but badly scored identically to one that answers them
        correctly. That made model capability invisible to the helpfulness axis,
        so a capability-tiered menu could not be measured at all.

        Default 0 -- judges without a quality model do not score it, and
        probe_menu.build_profile falls back to the refusal-only definition.
        """
        return 0


class RefusalDetector(ABC):
    """Separate from the judge: Help(c) counts refusals on benign traffic, and a
    model that declines a benign XSTest prompt has over-refused even though the
    judge would (correctly) call the response 'not a jailbreak'."""

    @abstractmethod
    def is_refusal(self, prompt: str, response: str) -> bool: ...
