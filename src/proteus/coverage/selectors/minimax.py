"""The classical security-game solve, over learned rather than given payoffs."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog

from proteus.coverage.coverage import Coverage, StaticCoverage
from proteus.coverage.selectors import SELECTORS
from proteus.coverage.selectors.base import Selector
from proteus.utils.logging import get_logger

logger = get_logger("minimax")


@SELECTORS.register("minimax")
class MinimaxSelector(Selector):
    """Commit the coverage minimising the attacker's best-case target.

        min_c  max_x  sum_q c_q J(q, x)
        s.t.   c in Delta(Q),  c . help >= tau,  c >= c_min

    The inner max is over the attacker's choice of target, so this is exactly
    the SSG program of Tambe (2011) with J and help measured rather than given.
    It is an LP: epigraph variable t bounds every target's coverage-induced
    jailbreak probability.

    With `graded: true` the objective becomes S(q, x) in place of J(q, x), so t
    bounds expected *harm* rather than break probability. The program is
    unchanged -- only the payoff matrix is -- but the solution is not: a
    configuration broken often at rubric level 3 now outranks one broken rarely
    at level 5, which the binary objective cannot express. Off by default so the
    binary condition stays reproducible; see configs/selectors/minimax_graded.yaml.

    Mixing is not assumed. If one configuration dominates every column of the
    payoff matrix, the LP returns a vertex and the coverage collapses to a router
    -- which is the falsification the coverage-entropy metric is there to catch.
    """

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        profile = self._require_profile()
        qids = self._menu.qids
        n = len(qids)

        graded = bool(self.params.get("graded", False))
        if graded and not profile.has_severity:
            logger.warning(
                "graded=true but this profile graded no severity (a judge without a "
                "severity model, or a profile.json written before harm was profiled); "
                "falling back to the binary J matrix."
            )
        J = profile.payoff_array(qids, graded=graded)  # (n, n_prompts)
        if J.shape[1] == 0:
            raise ValueError("Menu profile has no measured prompts; run probe_menu.py first")
        help_vec = profile.help_array(qids)

        # Variables: [c_0..c_{n-1}, t]. Objective: minimise t.
        obj = np.zeros(n + 1)
        obj[-1] = 1.0

        # Per-target epigraph: J[:, x] . c - t <= 0, one row per target.
        A_ub = np.hstack([J.T, -np.ones((J.shape[1], 1))])
        b_ub = np.zeros(J.shape[1])

        # Helpfulness: -help . c <= -tau
        if self._tau > 0:
            A_ub = np.vstack([A_ub, np.append(-help_vec, 0.0)])
            b_ub = np.append(b_ub, -self._tau)

        # Simplex: sum c = 1
        A_eq = np.append(np.ones(n), 0.0).reshape(1, -1)
        b_eq = np.array([1.0])

        bounds = [(self._c_min, 1.0)] * n + [(0.0, 1.0)]

        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not res.success:
            # Infeasible almost always means tau exceeds what the menu can deliver.
            logger.warning(
                f"Minimax LP failed ({res.message}); tau={self._tau} may exceed "
                f"max achievable helpfulness {help_vec.max():.3f}. Falling back to uniform."
            )
            return StaticCoverage(self._menu, np.ones(n), self.selector_id)

        c = np.clip(res.x[:n], 0.0, None)
        objective = "expected harm" if graded and profile.has_severity else "jailbreak prob"
        logger.info(
            f"Minimax: worst-target {objective} {res.x[-1]:.3f}, "
            f"support {int((c > 1e-6).sum())}/{n}"
        )
        return StaticCoverage(self._menu, c, self.selector_id)
