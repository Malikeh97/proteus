"""The Purple Agent: the defender's planner, not a third player.

STATUS: stub. Algorithm 1 of the proposal (per-configuration anticipatory RRT).
The pieces it needs already exist -- `menu.get(qid).serve` is the Serve oracle,
`MinimaxSelector` is the reweighting solve of line 15, and `metrics.equilibrium`
computes the regime classification -- so this is an assembly job, not a
from-scratch one.
"""

from __future__ import annotations

from typing import Any

from proteus.coverage.coverage import Coverage
from proteus.coverage.selectors import SELECTORS
from proteus.coverage.selectors.base import Selector


@SELECTORS.register("purple")
class PurpleSelector(Selector):
    """Simulates adversarial search against each candidate coverage before
    committing, and reweights toward configurations the simulation could not
    reach.

    Sketch of what to implement in select():

      1. Precompute Help[q] over benign traffic          (attacker-independent;
         already available as profile.help_array())
      2. For each q in menu.qids: grow an RRT over prompt space, expanding only
         through Safe/Refused nodes, recording jailbreaking prompts into a
         per-configuration memory M^q. Trees are per-configuration because the
         growth rule makes topology depend on who serves.
      3. Risk profile: rho_q(x) = max over p in M^q of K(d(x, p) / H), a kernel
         over embedding distance. This is what lets rho predict risk for unseen
         prompts -- a live request carries no measured outcome.
      4. Reweight: commit the coverage solving
             min_{c feasible} max_{x' in N(x, r)} c . rho(x')
         which is MinimaxSelector's LP with rho in place of J.
      5. Classify the regime (metrics.equilibrium.classify_regime) and iterate
         until the ASR gap closes or the budget is exhausted.

    Needed first: an embedding distance d, Sample/Extend over prompt space
    (reuse the attacker's mutation operator), and the rollout SimulateRedExpansion.
    """

    def select(self, history: Any = None, round_idx: int = 0) -> Coverage:
        raise NotImplementedError(
            "PurpleSelector is a stub -- see the docstring for the implementation plan. "
            "Use selector=minimax for the non-anticipatory version of the same solve."
        )
