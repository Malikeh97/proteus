"""Reasoner-based reweighting: the only prompt-conditional selector.

STATUS: stub. The scaffolding (PromptConditionalCoverage, the risk-scorer call
site, the tilt) is here; wire a real risk model into `_risk_score`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from proteus.coverage.coverage import Coverage, PromptConditionalCoverage
from proteus.coverage.selectors import SELECTORS
from proteus.coverage.selectors.base import Selector


@SELECTORS.register("reasoner")
class ReasonerSelector(Selector):
    """An LLM classifier scores each incoming prompt for risk and tilts the
    coverage toward safer configurations accordingly.

    It outputs a reweighting rather than a route: a hard assignment would
    collapse c(x) to a point mass and forfeit randomisation entirely, so the
    tilt is a temperature on the validation weights, bounded away from 0 and 1.
    """

    def _risk_score(self, prompt: str) -> float:
        """Risk in [0, 1] for one incoming prompt."""
        raise NotImplementedError(
            "ReasonerSelector is a stub. Implement _risk_score(prompt) -> [0, 1] "
            "using a guard model from configs/wrappers/ or a dedicated risk model, "
            "then the tilt below works as written."
        )

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        profile = self._require_profile()
        mask = self._feasible_mask()
        jb = np.array([profile.jailbreak_rate(q) for q in self._menu.qids])
        safety = np.clip(1.0 - jb, 1e-6, None) * mask
        beta = float(self.params.get("beta", 4.0))  # tilt strength at max risk

        def scorer(prompt: str) -> np.ndarray:
            risk = float(np.clip(self._risk_score(prompt), 0.0, 1.0))
            w = np.power(safety, 1.0 + beta * risk)
            w = np.maximum(w, self._c_min * mask)
            return w / w.sum()

        return PromptConditionalCoverage(self._menu, scorer, self.selector_id)
