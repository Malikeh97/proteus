"""Selectors that need no optimisation: uniform, validation-weighted, and the
deterministic baseline the whole proposal is measured against."""

from __future__ import annotations

from typing import Any

import numpy as np

from proteus.coverage.coverage import Coverage, StaticCoverage
from proteus.coverage.selectors import SELECTORS
from proteus.coverage.selectors.base import Selector
from proteus.utils.logging import get_logger

logger = get_logger("selectors")


@SELECTORS.register("uniform")
class UniformSelector(Selector):
    """Uniform mixing over the feasible menu."""

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        mask = self._feasible_mask()
        w = mask.astype(float)
        w = np.maximum(w, self._c_min)
        return StaticCoverage(self._menu, w, self.selector_id)


@SELECTORS.register("validation")
class ValidationSelector(Selector):
    """Mass allocated inversely to each configuration's measured jailbreak rate.

    w_q ∝ (1 - JB_q)^alpha over the feasible set. alpha=1 is inverse-risk;
    larger alpha concentrates toward the safest configurations, and alpha -> inf
    recovers the deterministic baseline.
    """

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        profile = self._require_profile()
        alpha = float(self.params.get("alpha", 1.0))
        mask = self._feasible_mask()

        jb = np.array([profile.jailbreak_rate(q) for q in self._menu.qids])
        w = np.power(np.clip(1.0 - jb, 0.0, 1.0), alpha) * mask
        if w.sum() <= 0:
            logger.warning("All feasible configurations jailbreak always; falling back to uniform")
            w = mask.astype(float)
        w = w / w.sum()
        w = np.maximum(w, self._c_min * mask)
        return StaticCoverage(self._menu, w, self.selector_id)


@SELECTORS.register("deterministic")
class DeterministicSelector(Selector):
    """The best single (model, wrapper-set) configuration: a point mass.

    This is the baseline RQ1 compares against -- "the best deterministic
    model-defense configuration" -- so it must be selected the same way the
    mixtures are, from the same fit split.
    """

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        profile = self._require_profile()
        mask = self._feasible_mask()
        jb = np.array([profile.jailbreak_rate(q) for q in self._menu.qids])
        jb = np.where(mask, jb, np.inf)
        w = np.zeros(len(self._menu))
        w[int(np.argmin(jb))] = 1.0
        logger.info(f"Deterministic baseline: {self._menu.qids[int(np.argmin(jb))]}")
        return StaticCoverage(self._menu, w, self.selector_id)
