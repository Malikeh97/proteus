"""RL-trained (GRPO) prompt policy: the 'Seen (in-loop)' attacker of Sec. 4.

STATUS: stub. Requires the `rl` extra (`uv sync --extra rl`).
"""

from __future__ import annotations

from proteus.attackers import ATTACKERS
from proteus.attackers.base import Attacker


@ATTACKERS.register("grpo")
class GRPOAttacker(Attacker):
    """A prompt policy trained with GRPO, rewarded for successful, non-duplicate
    jailbreaks with an embedding-diversity term, retrained against each
    committed coverage.

    Implementation plan:

      * reset(coverage_id) is the retraining hook. It is called once per round,
        with the new coverage's id, and must run a fresh GRPO pass against the
        deployment (the runner passes a serve callable via `bind_deployment`).
      * Reward per rollout:
            r = judge(x, y)                       # success
                - lambda_dup * max_sim(x, seen)   # non-duplicate
                + lambda_div * diversity(batch)   # embedding spread
        The judge here MUST be a separate reward model, not the evaluation judge
        (Sec. 4) -- otherwise the attacker learns to fool the judge rather than
        the defense.
      * The policy sees only (x, y) pairs. It must never be conditioned on q or
        c; see proteus/attackers/base.py.
      * initialize() samples from the trained policy; refine() re-samples
        conditioned on the failed attempt.

    Suggested stack: trl.GRPOTrainer over the attacker model with a LoRA adapter
    (peft), rollouts generated against a frozen Deployment.
    """

    def bind_deployment(self, deployment) -> None:
        """The runner calls this before reset() so rollouts can query the
        committed coverage. Store it; do not read c_t off it."""
        self._deployment = deployment

    def reset(self, coverage_id: str | None = None) -> None:
        raise NotImplementedError(
            "GRPOAttacker is a stub -- see the docstring for the training plan. "
            "Install with `uv sync --extra rl`. Use attacker=pair for the v1 "
            "adaptive attacker."
        )

    def initialize(self, base_prompt: str) -> str:
        raise NotImplementedError("GRPOAttacker is a stub.")

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        raise NotImplementedError("GRPOAttacker is a stub.")
