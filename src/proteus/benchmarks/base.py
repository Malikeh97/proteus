from __future__ import annotations

from abc import ABC, abstractmethod

from proteus.game.runner import Prompt


class Benchmark(ABC):
    @property
    @abstractmethod
    def benchmark_id(self) -> str: ...

    @abstractmethod
    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]: ...

    @staticmethod
    def split(
        prompts: list[Prompt], fit_frac: float, seed: int = 0
    ) -> tuple[list[Prompt], list[Prompt]]:
        """Prompts that fit a coverage must be disjoint from those scoring it."""
        import numpy as np

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(prompts))
        cut = int(len(prompts) * fit_frac)
        fit = [prompts[i] for i in idx[:cut]]
        ev = [prompts[i] for i in idx[cut:]]
        for p in fit:
            p.split = "fit"
        for p in ev:
            p.split = "eval"
        return fit, ev
