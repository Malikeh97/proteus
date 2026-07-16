"""Benign traffic B, on which Help(c) is measured.

XSTest supplies the adversarial-looking-but-benign cases where over-refusal
bites; a general instruction set is mixed in so the estimate is not dominated by
them (Sec. 4).
"""

from __future__ import annotations

from proteus.benchmarks import BENCHMARKS
from proteus.benchmarks.base import Benchmark
from proteus.game.runner import Prompt
from proteus.utils.logging import get_logger

logger = get_logger("benign")


@BENCHMARKS.register("xstest")
class XSTest(Benchmark):
    """XSTest safe prompts only (Röttger et al., 2024). The unsafe half is a
    harm benchmark, not benign traffic, so it is filtered out."""

    def __init__(self, hf_name: str = "walledai/XSTest") -> None:
        self._hf_name = hf_name

    @property
    def benchmark_id(self) -> str:
        return "xstest"

    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]:
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self._hf_name, split="test")
        prompts = [
            Prompt(prompt_id=f"xs_{i}", text=row["prompt"], split="benign")
            for i, row in enumerate(ds)
            if str(row.get("label", "safe")).lower() == "safe"
        ]
        if n is not None and n < len(prompts):
            idx = np.random.default_rng(seed).choice(len(prompts), size=n, replace=False)
            prompts = [prompts[i] for i in sorted(idx)]
        logger.info(f"Loaded {len(prompts)} XSTest safe prompts")
        return prompts


@BENCHMARKS.register("alpaca")
class Alpaca(Benchmark):
    """General instructions: the bulk of real benign traffic."""

    def __init__(self, hf_name: str = "tatsu-lab/alpaca") -> None:
        self._hf_name = hf_name

    @property
    def benchmark_id(self) -> str:
        return "alpaca"

    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]:
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self._hf_name, split="train")
        rows = [r for r in ds if not r.get("input")]  # skip the ones needing a context field
        prompts = [
            Prompt(prompt_id=f"al_{i}", text=r["instruction"], split="benign")
            for i, r in enumerate(rows)
        ]
        if n is not None and n < len(prompts):
            idx = np.random.default_rng(seed).choice(len(prompts), size=n, replace=False)
            prompts = [prompts[i] for i in sorted(idx)]
        logger.info(f"Loaded {len(prompts)} Alpaca prompts")
        return prompts


@BENCHMARKS.register("xstest_alpaca")
class XSTestPlusAlpaca(Benchmark):
    """XSTest + Alpaca, mixed by `xstest_frac`. The default benign source."""

    def __init__(self, xstest_frac: float = 0.5) -> None:
        self._frac = xstest_frac

    @property
    def benchmark_id(self) -> str:
        return "xstest_alpaca"

    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]:
        n = n or 100
        n_xs = int(n * self._frac)
        return XSTest().load(n_xs, seed) + Alpaca().load(n - n_xs, seed)
