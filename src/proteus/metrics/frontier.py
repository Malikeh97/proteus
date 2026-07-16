"""The safety-helpfulness frontier: the operator's real object of choice.

Sweeping the floor tau traces it, so the unit of comparison between selection
strategies is a curve, not a point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrontierPoint:
    tau: float
    asr: float
    helpfulness: float
    entropy: float
    support: int
    selector_id: str


def pareto_front(points: list[FrontierPoint]) -> list[FrontierPoint]:
    """Points not dominated on (low ASR, high helpfulness)."""
    front = []
    for p in points:
        dominated = any(
            (q.asr <= p.asr and q.helpfulness >= p.helpfulness)
            and (q.asr < p.asr or q.helpfulness > p.helpfulness)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.helpfulness)


def frontier_auc(points: list[FrontierPoint]) -> float:
    """Area under the (helpfulness, 1 - ASR) curve: one scalar per strategy for
    ranking, though the curve itself is what should be reported."""
    front = pareto_front(points)
    if len(front) < 2:
        return float("nan")
    x = np.array([p.helpfulness for p in front])
    y = np.array([1.0 - p.asr for p in front])
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))
