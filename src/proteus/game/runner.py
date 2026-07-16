"""The round loop.

Stackelberg ordering inside each round: the defender commits c_t, the attacker
issues x_t, a configuration q_t ~ c_t is drawn and generates y_t.

A configuration is drawn *per request*, not per round: the attacker's second
probe of the same prompt may land on a different member of the support, which is
precisely what makes the deployed defense hard to identify.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proteus.attackers.base import Attacker
from proteus.coverage.coverage import Coverage
from proteus.game.deployment import Deployment
from proteus.game.outcome import Outcome
from proteus.utils.io import StepRecord, TrialRecord, append_jsonl, load_completed_ids
from proteus.utils.logging import get_logger

logger = get_logger("runner")


@dataclass
class Prompt:
    prompt_id: str
    text: str
    split: str  # fit | eval | benign


def run_trial(
    prompt: Prompt,
    deployment: Deployment,
    attacker: Attacker,
    budget: int,
    round_idx: int,
    seed: int,
) -> TrialRecord:
    """One target, one attack budget, against the committed coverage."""
    record = TrialRecord(
        prompt_id=prompt.prompt_id,
        base_prompt=prompt.text,
        split=prompt.split,
        round=round_idx,
        seed=seed,
        coverage_id=deployment.coverage.coverage_id,
        attacker_id=attacker.attacker_id,
    )

    x = attacker.initialize(prompt.text)
    tokens_before = attacker.tokens_spent

    for step in range(1, budget + 1):
        served = deployment.serve(x, objective=prompt.text)
        judgment = int(served.outcome == Outcome.JAILBREAK)

        record.steps.append(
            StepRecord(
                step=step,
                prompt=x,
                response=served.response,
                qid=served.qid,
                outcome=served.outcome.value,
                refused_by=served.refused_by,
                prompt_tokens=served.prompt_tokens,
                completion_tokens=served.completion_tokens,
                wrapper_tokens=served.wrapper_tokens,
                attacker_tokens=attacker.tokens_spent - tokens_before,
            )
        )
        tokens_before = attacker.tokens_spent

        if judgment == 1:
            record.success = True
            record.first_success_step = step
            break

        if step < budget:
            # The attacker sees x, y, z -- never q or c. See attackers/base.py.
            x = attacker.refine(x, served.response, judgment, step)

    return record


def run_benign_trial(
    prompt: Prompt, deployment: Deployment, round_idx: int, seed: int
) -> TrialRecord:
    """Benign traffic gets one request, no refinement: Help(c) is measured on
    what a normal user sends, not on an adversarial rewrite of it."""
    served = deployment.serve(prompt.text, objective=prompt.text)
    return TrialRecord(
        prompt_id=prompt.prompt_id,
        base_prompt=prompt.text,
        split="benign",
        round=round_idx,
        seed=seed,
        coverage_id=deployment.coverage.coverage_id,
        attacker_id="none",
        success=False,
        steps=[
            StepRecord(
                step=1,
                prompt=prompt.text,
                response=served.response,
                qid=served.qid,
                outcome=served.outcome.value,
                refused_by=served.refused_by,
                prompt_tokens=served.prompt_tokens,
                completion_tokens=served.completion_tokens,
                wrapper_tokens=served.wrapper_tokens,
            )
        ],
    )


def run_round(
    coverage: Coverage,
    attack_prompts: list[Prompt],
    benign_prompts: list[Prompt],
    deployment: Deployment,
    attacker: Attacker,
    budget: int,
    round_idx: int,
    seed: int,
    output_path: Path,
    resume: bool = True,
) -> list[TrialRecord]:
    """One round: the attacker gets a fresh budget against the committed coverage."""
    done = load_completed_ids(output_path) if resume else set()
    if done:
        logger.info(f"Resuming round {round_idx}: {len(done)} prompts already recorded")

    attacker.reset(coverage_id=coverage.coverage_id)
    records: list[TrialRecord] = []

    for p in attack_prompts:
        if p.prompt_id in done:
            continue
        rec = run_trial(p, deployment, attacker, budget, round_idx, seed)
        append_jsonl(rec, output_path)
        records.append(rec)

    for p in benign_prompts:
        if p.prompt_id in done:
            continue
        rec = run_benign_trial(p, deployment, round_idx, seed)
        append_jsonl(rec, output_path)
        records.append(rec)

    asr = np.mean([r.success for r in records if r.split != "benign"]) if records else 0.0
    benign = [r for r in records if r.split == "benign"]
    help_rate = 1.0 - np.mean([r.refused() for r in benign]) if benign else float("nan")
    logger.info(
        f"Round {round_idx} | coverage={coverage.coverage_id} "
        f"H(c)={coverage.entropy():.3f} ASR={asr:.3f} Help={help_rate:.3f}"
    )
    return records
