"""Prompt-level and randomised-smoothing wrappers.

These are cheap (no extra model) and exist so a menu can be exercised end to end
without loading every guard -- see configs/menus/dev.yaml.
"""

from __future__ import annotations

import random
import string

from proteus.menu.wrappers import WRAPPERS
from proteus.menu.wrappers.base import Verdict, Wrapper
from proteus.utils.config import WrapperConfig

_DEFAULT_SYSTEM_PROMPT = (
    "You are a safe and helpful assistant. Refuse any request for content that "
    "would cause harm. Answer benign requests fully and directly."
)


@WRAPPERS.register("safety_prompt")
class SafetyPrompt(Wrapper):
    """Prepends a safety preamble. Input-stage transform, never refuses."""

    def __init__(self, config: WrapperConfig) -> None:
        super().__init__(config)
        self._preamble = config.params.get("preamble", _DEFAULT_SYSTEM_PROMPT)

    def on_input(self, prompt: str) -> Verdict:
        if not self.runs_on_input:
            return Verdict.passthrough(prompt)
        return Verdict.passthrough(f"{self._preamble}\n\n{prompt}")


@WRAPPERS.register("keyword_filter")
class KeywordFilter(Wrapper):
    """Substring blocklist. A deliberately weak, deterministic mechanism -- useful
    as a menu member whose blind spots are obvious and complementary."""

    _DEFAULT_KEYWORDS = [
        "bomb",
        "explosive",
        "bioweapon",
        "nerve agent",
        "ransomware",
        "child porn",
    ]

    def __init__(self, config: WrapperConfig) -> None:
        super().__init__(config)
        self._keywords = [k.lower() for k in config.params.get("keywords", self._DEFAULT_KEYWORDS)]

    def _hit(self, text: str) -> str | None:
        low = text.lower()
        return next((k for k in self._keywords if k in low), None)

    def on_input(self, prompt: str) -> Verdict:
        if not self.runs_on_input:
            return Verdict.passthrough(prompt)
        hit = self._hit(prompt)
        return Verdict.refuse(f"{self.wrapper_id}:input:{hit}") if hit else Verdict.passthrough(prompt)

    def on_output(self, prompt: str, response: str) -> Verdict:
        if not self.runs_on_output:
            return Verdict.passthrough(response)
        hit = self._hit(response)
        return (
            Verdict.refuse(f"{self.wrapper_id}:output:{hit}") if hit else Verdict.passthrough(response)
        )


@WRAPPERS.register("smoothllm")
class SmoothLLM(Wrapper):
    """Character-level perturbation of the incoming prompt (Robey et al., 2023).

    The single-copy variant: perturb and serve. The n-copy majority vote needs
    the model, which the Wrapper interface deliberately does not expose, so the
    full version belongs in a Configuration subclass rather than here.
    """

    def __init__(self, config: WrapperConfig) -> None:
        super().__init__(config)
        self._q = float(config.params.get("perturb_pct", 0.1))
        self._rng = random.Random(config.params.get("seed", 0))

    def on_input(self, prompt: str) -> Verdict:
        if not self.runs_on_input:
            return Verdict.passthrough(prompt)
        n = max(1, int(len(prompt) * self._q))
        chars = list(prompt)
        for idx in self._rng.sample(range(len(chars)), min(n, len(chars))):
            chars[idx] = self._rng.choice(string.printable[:94])
        return Verdict.passthrough("".join(chars))
