#!/usr/bin/env python3
"""Phase 1 -- measure J(q, x) and Help(q) for every configuration in the menu.

No payoff matrix is supplied by the framework: the selectors optimise over
payoffs measured here by execution. Runs on the *fit* split only; the coverage
these numbers select is scored on the disjoint eval split in Phase 2.

Also produces every single-configuration baseline for free: the undefended
models, each wrapper alone, and the naive stack of all wrappers are all just
menu members.

Usage:
    python scripts/probe_menu.py --experiment configs/experiments/base.yaml \\
        --output-dir $PROTEUS_OUTPUT_DIR --resume

    # Override the menu without touching the config
    python scripts/probe_menu.py --experiment configs/experiments/base.yaml \\
        --menu small --seeds 1997 --output-dir $PROTEUS_OUTPUT_DIR

Writes:
    {output_dir}/{menu_id}/profile/{seed}/probe.jsonl    per-trial records
    {output_dir}/{menu_id}/profile/{seed}/profile.json   the MenuProfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from proteus.attackers import load_attacker_by_name  # noqa: E402
from proteus.benchmarks import load_benchmark  # noqa: E402
from proteus.benchmarks.base import Benchmark  # noqa: E402
from proteus.coverage.coverage import StaticCoverage  # noqa: E402
from proteus.coverage.profile import MenuProfile  # noqa: E402
from proteus.game.deployment import Deployment  # noqa: E402
from proteus.game.runner import run_benign_trial, run_trial  # noqa: E402
from proteus.judges import KeywordRefusalDetector, load_judge_by_name  # noqa: E402
from proteus.menu.menu import ResourceMenu  # noqa: E402
from proteus.utils.config import load_experiment_config  # noqa: E402
from proteus.utils.io import append_jsonl, load_completed_ids, read_jsonl  # noqa: E402
from proteus.utils.logging import get_logger  # noqa: E402

load_dotenv()
logger = get_logger("probe_menu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="Path to configs/experiments/*.yaml")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--resume", action="store_true", help="Skip prompts already recorded")
    p.add_argument("--menu", help="Override: menu stem")
    p.add_argument("--attacker", help="Override: attacker stem used to measure J")
    p.add_argument("--judge", help="Override: judge stem")
    p.add_argument("--seeds", type=int, nargs="+", help="Override: seeds")
    p.add_argument("--n-attack-prompts", type=int, help="Override: attack prompt count")
    p.add_argument("--n-benign-prompts", type=int, help="Override: benign prompt count")
    return p.parse_args()


def build_profile(jsonl_path: Path, menu: ResourceMenu) -> MenuProfile:
    """Rebuild the profile from the record file rather than from this run's
    trials, so a resumed probe profiles everything on disk, not just the tail."""
    profile = MenuProfile(menu_id=menu.menu_id, qids=menu.qids)
    benign: dict[str, list[bool]] = {}

    for rec in read_jsonl(jsonl_path):
        qid = rec.meta.get("qid")
        base_id = rec.meta.get("base_prompt_id")
        if qid is None or base_id is None:
            continue
        if rec.split == "benign":
            benign.setdefault(qid, []).append(rec.refused())
        else:
            profile.j_matrix.setdefault(qid, {})[base_id] = int(rec.success)

    for qid, refusals in benign.items():
        profile.help_rate[qid] = 1.0 - float(np.mean(refusals))
    return profile


def probe_seed(cfg, args, menu: ResourceMenu, seed: int) -> None:
    out_dir = Path(args.output_dir) / menu.menu_id / "profile" / str(seed)
    jsonl_path = out_dir / "probe.jsonl"
    profile_path = out_dir / "profile.json"

    attack_bm: Benchmark = load_benchmark(cfg.attack_benchmark)
    benign_bm: Benchmark = load_benchmark(cfg.benign_benchmark)
    attack_prompts = attack_bm.load(cfg.n_attack_prompts, seed)
    fit_prompts, _ = Benchmark.split(attack_prompts, cfg.fit_frac, seed)
    benign_prompts = benign_bm.load(cfg.n_benign_prompts, seed)

    judge = load_judge_by_name(cfg.judge, args.configs_dir)
    refusal = KeywordRefusalDetector()
    attacker = load_attacker_by_name(cfg.attacker, args.configs_dir)

    done = load_completed_ids(jsonl_path) if args.resume else set()

    for qid in tqdm(menu.qids, desc=f"seed {seed}"):
        # A point mass on q: probing a single configuration is the degenerate
        # coverage, so it reuses the whole serving path unchanged.
        w = np.zeros(len(menu))
        w[menu.index(qid)] = 1.0
        coverage = StaticCoverage(menu, w, coverage_id=f"probe:{qid}")
        deployment = Deployment(coverage, judge, refusal, np.random.default_rng(seed))
        attacker.reset(coverage_id=coverage.coverage_id)

        for p in fit_prompts:
            key = f"{qid}::{p.prompt_id}"  # namespaced so resume works across configurations
            if key in done:
                continue
            rec = run_trial(p, deployment, attacker, cfg.attack_budget, 0, seed)
            rec.prompt_id = key
            rec.meta = {"qid": qid, "base_prompt_id": p.prompt_id}
            append_jsonl(rec, jsonl_path)

        for p in benign_prompts:
            key = f"{qid}::{p.prompt_id}"
            if key in done:
                continue
            rec = run_benign_trial(p, deployment, 0, seed)
            rec.prompt_id = key
            rec.meta = {"qid": qid, "base_prompt_id": p.prompt_id}
            append_jsonl(rec, jsonl_path)

    profile = build_profile(jsonl_path, menu)
    profile.meta = {
        "seed": seed,
        "attacker": cfg.attacker,
        "judge": cfg.judge,
        "attack_benchmark": cfg.attack_benchmark,
        "benign_benchmark": cfg.benign_benchmark,
        "attack_budget": cfg.attack_budget,
        "n_fit_prompts": len(fit_prompts),
    }
    profile.save(profile_path)

    for qid in menu.qids:
        logger.info(
            f"{qid}: JB={profile.jailbreak_rate(qid):.3f} "
            f"Help={profile.help_rate.get(qid, float('nan')):.3f}"
        )
    logger.info(f"Wrote {profile_path}")


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment)

    for field in ("menu", "attacker", "judge", "seeds"):
        if getattr(args, field, None):
            setattr(cfg, field, getattr(args, field))
    if args.n_attack_prompts:
        cfg.n_attack_prompts = args.n_attack_prompts
    if args.n_benign_prompts:
        cfg.n_benign_prompts = args.n_benign_prompts

    menu = ResourceMenu.from_name(cfg.menu, args.configs_dir)
    logger.info(f"Probing menu '{cfg.menu}' (|Q|={len(menu)}) with attacker '{cfg.attacker}'")

    for seed in cfg.seeds:
        probe_seed(cfg, args, menu, seed)


if __name__ == "__main__":
    main()
