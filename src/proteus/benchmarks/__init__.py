"""Prompt sources. Attack benchmarks supply targets; benign benchmarks supply
the traffic Help(c) is measured on.
"""

from __future__ import annotations

from proteus.benchmarks.base import Benchmark
from proteus.utils.registry import Registry

BENCHMARKS: Registry[Benchmark] = Registry("benchmark")

# Registration side effects; must follow BENCHMARKS.
from proteus.benchmarks import benign, harmbench  # noqa: E402,F401


def load_benchmark(name: str, **kwargs) -> Benchmark:
    return BENCHMARKS.create(name, **kwargs)


__all__ = ["BENCHMARKS", "Benchmark", "load_benchmark"]
