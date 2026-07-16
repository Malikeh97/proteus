from __future__ import annotations

from proteus.benchmarks import BENCHMARKS
from proteus.benchmarks.base import Benchmark
from proteus.game.runner import Prompt
from proteus.utils.logging import get_logger

logger = get_logger("harmbench")


@BENCHMARKS.register("harmbench")
class HarmBench(Benchmark):
    """HarmBench standard behaviours (Mazeika et al., 2024)."""

    def __init__(self, hf_name: str = "walledai/HarmBench", subset: str = "standard") -> None:
        self._hf_name = hf_name
        self._subset = subset

    @property
    def benchmark_id(self) -> str:
        return "harmbench"

    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]:
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self._hf_name, self._subset, split="train")
        prompts = [
            Prompt(prompt_id=f"hb_{i}", text=row["prompt"], split="fit")
            for i, row in enumerate(ds)
        ]
        if n is not None and n < len(prompts):
            idx = np.random.default_rng(seed).choice(len(prompts), size=n, replace=False)
            prompts = [prompts[i] for i in sorted(idx)]
        logger.info(f"Loaded {len(prompts)} HarmBench prompts")
        return prompts


@BENCHMARKS.register("advbench")
class AdvBench(Benchmark):
    """AdvBench harmful behaviours (Zou et al., 2023). A cheaper stand-in."""

    def __init__(self, hf_name: str = "walledai/AdvBench") -> None:
        self._hf_name = hf_name

    @property
    def benchmark_id(self) -> str:
        return "advbench"

    def load(self, n: int | None = None, seed: int = 0) -> list[Prompt]:
        import numpy as np
        from datasets import load_dataset

        ds = load_dataset(self._hf_name, split="train")
        prompts = [
            Prompt(prompt_id=f"ab_{i}", text=row["prompt"], split="fit")
            for i, row in enumerate(ds)
        ]
        if n is not None and n < len(prompts):
            idx = np.random.default_rng(seed).choice(len(prompts), size=n, replace=False)
            prompts = [prompts[i] for i in sorted(idx)]
        logger.info(f"Loaded {len(prompts)} AdvBench prompts")
        return prompts
