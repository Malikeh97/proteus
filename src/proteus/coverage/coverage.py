"""Coverage c: X x H -> Delta(Q), the defender's committed strategy.

A deterministic pipeline is the special case where c is a point mass, so
baselines and mixtures share one type and one measurement path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from proteus.menu.configuration import Configuration
from proteus.menu.menu import ResourceMenu


class Coverage(ABC):
    def __init__(self, menu: ResourceMenu, coverage_id: str) -> None:
        self._menu = menu
        self._coverage_id = coverage_id

    @property
    def coverage_id(self) -> str:
        return self._coverage_id

    @property
    def menu(self) -> ResourceMenu:
        return self._menu

    @abstractmethod
    def probs(self, prompt: str, history: Any = None) -> np.ndarray:
        """Distribution over menu.qids, in that order."""

    def sample(self, prompt: str, rng: np.random.Generator, history: Any = None) -> Configuration:
        p = self.probs(prompt, history)
        qid = self._menu.qids[int(rng.choice(len(p), p=p))]
        return self._menu.get(qid)

    def entropy(self, prompt: str = "", history: Any = None) -> float:
        """Shannon entropy in nats. 0 iff the coverage is a point mass."""
        p = self.probs(prompt, history)
        nz = p[p > 0]
        # max() rather than the bare sum: a point mass yields -0.0, which is
        # equal to 0.0 but renders as "-0.000" in every table.
        return max(0.0, float(-np.sum(nz * np.log(nz))))

    def support(self, prompt: str = "", tol: float = 1e-9) -> list[str]:
        p = self.probs(prompt)
        return [q for q, w in zip(self._menu.qids, p, strict=True) if w > tol]

    def to_dict(self, prompt: str = "") -> dict:
        return {
            "coverage_id": self._coverage_id,
            "weights": dict(zip(self._menu.qids, self.probs(prompt).tolist(), strict=True)),
            "entropy": self.entropy(prompt),
        }


class StaticCoverage(Coverage):
    """Prompt-independent mixture. What every selector but the reasoner returns."""

    def __init__(self, menu: ResourceMenu, weights: np.ndarray, coverage_id: str) -> None:
        super().__init__(menu, coverage_id)
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(menu),):
            raise ValueError(f"Expected {len(menu)} weights, got {w.shape}")
        if w.min() < 0:
            raise ValueError("Coverage weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("Coverage weights must sum to something positive")
        self._weights = w / total

    def probs(self, prompt: str, history: Any = None) -> np.ndarray:
        return self._weights


class PromptConditionalCoverage(Coverage):
    """Realises the prompt-dependent c(x) the formal setup admits.

    `scorer` maps a prompt to a weight vector over menu.qids; results are cached
    per prompt so repeated draws within a trial stay consistent and cheap.
    """

    def __init__(self, menu: ResourceMenu, scorer, coverage_id: str) -> None:
        super().__init__(menu, coverage_id)
        self._scorer = scorer
        self._cache: dict[str, np.ndarray] = {}

    def probs(self, prompt: str, history: Any = None) -> np.ndarray:
        if prompt not in self._cache:
            w = np.asarray(self._scorer(prompt), dtype=float)
            self._cache[prompt] = w / w.sum()
        return self._cache[prompt]
