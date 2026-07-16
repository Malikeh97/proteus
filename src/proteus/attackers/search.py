"""Search-based attacker with LLM-driven mutation: the 'Unseen (held-out)'
attacker of Sec. 4.

STATUS: stub. Used only to evaluate a committed coverage, never to select it --
the runner enforces this by loading it after selection.
"""

from __future__ import annotations

from proteus.attackers import ATTACKERS
from proteus.attackers.base import Attacker


@ATTACKERS.register("search")
class SearchAttacker(Attacker):
    """Tree/beam search over prompt space with an LLM mutation operator.

    Implementation plan:

      * Maintain a beam of candidate prompts per target. At each step, mutate
        every beam member k ways with the attacker model (paraphrase, reframe,
        escalate, encode), serve them, and keep the top-b by a surrogate score.
      * The surrogate must be computable from the response alone -- e.g. refusal
        length, on-topic-ness, or a cheap classifier. It cannot call the
        evaluation judge, or the held-out attacker is no longer held out.
      * refine() returns the next beam member to try; the beam itself is state,
        so clear it in reset().

    This differs from PAIR in that it keeps a frontier rather than a single
    trajectory, which is what makes it a genuinely different best-response
    algorithm for the generalisation test in RQ6.
    """

    def initialize(self, base_prompt: str) -> str:
        raise NotImplementedError(
            "SearchAttacker is a stub -- see the docstring for the plan. "
            "Use held_out_attacker=template for a cheap non-adaptive held-out check."
        )

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        raise NotImplementedError("SearchAttacker is a stub.")
