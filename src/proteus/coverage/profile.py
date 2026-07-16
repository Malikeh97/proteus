"""Per-configuration measurements: the payoffs selectors optimise over.

The proposal supplies no payoff matrix -- J(q, x) and Help(q) are measured by
execution on a *fit* split (scripts/probe_menu.py), and the coverage they select
is then scored on a disjoint *eval* split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from proteus.utils.io import read_json, write_json


@dataclass
class MenuProfile:
    """Measured jailbreak and helpfulness behaviour of every q in the menu."""

    menu_id: str
    qids: list[str]
    # qid -> prompt_id -> J(q, x) in {0, 1}
    j_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    # qid -> Pr[o != Refused] on benign traffic
    help_rate: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def prompt_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.j_matrix.values():
            for pid in row:
                seen[pid] = None
        return list(seen)

    def jailbreak_rate(self, qid: str) -> float:
        row = self.j_matrix.get(qid, {})
        return float(np.mean(list(row.values()))) if row else 0.0

    def j_array(self, qids: list[str] | None = None) -> np.ndarray:
        """(|Q|, |X_fit|) matrix of J(q, x), aligned to `qids` and prompt_ids."""
        qids = qids or self.qids
        pids = self.prompt_ids
        return np.array(
            [[self.j_matrix.get(q, {}).get(p, 0) for p in pids] for q in qids], dtype=float
        )

    def help_array(self, qids: list[str] | None = None) -> np.ndarray:
        qids = qids or self.qids
        return np.array([self.help_rate.get(q, 1.0) for q in qids], dtype=float)

    def save(self, path: Path) -> None:
        write_json(
            {
                "menu_id": self.menu_id,
                "qids": self.qids,
                "j_matrix": self.j_matrix,
                "help_rate": self.help_rate,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MenuProfile:
        d = read_json(path)
        return cls(
            menu_id=d["menu_id"],
            qids=d["qids"],
            j_matrix=d.get("j_matrix", {}),
            help_rate=d.get("help_rate", {}),
            meta=d.get("meta", {}),
        )
