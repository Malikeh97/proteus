#!/usr/bin/env python3
"""Phase 2 -- commit a coverage, let the attacker adapt against it, measure.

Each round: the defender commits c_t from the menu profile; the attacker gets a
fresh budget against it and observes only responses; a configuration is drawn per
request. ASR, helpfulness, and the residual incentive to deviate are measured by
execution, not supplied.

Requires a menu profile from Phase 1 for every selector but `uniform`.

Usage:
    # Randomized deployment vs the deterministic baseline (RQ1)
    python scripts/run_game.py --experiment configs/experiments/paper/rq1_randomized_vs_deterministic.yaml \\
        --selector minimax --output-dir $PROTEUS_OUTPUT_DIR --resume

    python scripts/run_game.py --experiment configs/experiments/paper/rq1_randomized_vs_deterministic.yaml \\
        --selector deterministic --output-dir $PROTEUS_OUTPUT_DIR --resume

    # Static vs adaptive at one selector (RQ2)
    python scripts/run_game.py --experiment configs/experiments/paper/rq2_static_vs_adaptive.yaml \\
        --selector minimax --attacker static --output-dir $PROTEUS_OUTPUT_DIR

    # Sweep the helpfulness floor to trace the frontier
    python scripts/run_game.py --experiment configs/experiments/base.yaml \\
        --selector minimax --tau 0.9 --output-dir $PROTEUS_OUTPUT_DIR

Writes:
    {output_dir}/{menu}/{selector}/{attacker}/tau{tau}/{seed}/round{t}/results.jsonl
    {output_dir}/{menu}/{selector}/{attacker}/tau{tau}/{seed}/coverages.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from proteus.attackers import load_attacker_by_name  # noqa: E402
from proteus.benchmarks import load_benchmark  # noqa: E402
from proteus.benchmarks.base import Benchmark  # noqa: E402
from proteus.coverage.profile import MenuProfile  # noqa: E402
from proteus.coverage.selectors import load_selector_by_name  # noqa: E402
from proteus.game.deployment import Deployment  # noqa: E402
from proteus.game.runner import run_round  # noqa: E402
from proteus.judges import KeywordRefusalDetector, load_judge_by_name  # noqa: E402
from proteus.menu.menu import ResourceMenu  # noqa: E402
from proteus.utils.config import load_experiment_config  # noqa: E402
from proteus.utils.io import write_json  # noqa: E402
from proteus.utils.logging import get_logger  # noqa: E402

load_dotenv()
logger = get_logger("run_game")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="Path to configs/experiments/*.yaml")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--resume", action="store_true", help="Skip prompts already recorded")
    p.add_argument("--profile", help="Path to profile.json (default: derived from --output-dir)")
    p.add_argument("--menu", help="Override: menu stem")
    p.add_argument("--selector", help="Override: selector stem")
    p.add_argument("--attacker", help="Override: attacker stem")
    p.add_argument("--held-out-attacker", help="Override: held-out attacker stem")
    p.add_argument("--judge", help="Override: judge stem")
    p.add_argument("--tau", type=float, help="Override: helpfulness floor")
    p.add_argument("--c-min", type=float, help="Override: minimum mass per configuration")
    p.add_argument("--rounds", type=int, help="Override: number of rounds")
    p.add_argument("--attack-budget", type=int, help="Override: refinement steps per prompt")
    p.add_argument("--seeds", type=int, nargs="+", help="Override: seeds")
    return p.parse_args()


def apply_overrides(cfg, args) -> None:
    for field in ("menu", "selector", "attacker", "judge", "seeds", "rounds", "tau"):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    if args.held_out_attacker:
        cfg.held_out_attacker = args.held_out_attacker
    if args.c_min is not None:
        cfg.c_min = args.c_min
    if args.attack_budget:
        cfg.attack_budget = args.attack_budget


def load_profile(cfg, args, menu: ResourceMenu, seed: int) -> MenuProfile | None:
    path = (
        Path(args.profile)
        if args.profile
        else Path(args.output_dir) / menu.menu_id / "profile" / str(seed) / "profile.json"
    )
    if not path.exists():
        logger.warning(f"No menu profile at {path}")
        return None
    logger.info(f"Loaded menu profile from {path}")
    return MenuProfile.load(path)


def run_seed(cfg, args, menu: ResourceMenu, seed: int) -> None:
    out_dir = (
        Path(args.output_dir)
        / menu.menu_id
        / cfg.selector
        / cfg.attacker
        / f"tau{cfg.tau}"
        / str(seed)
    )

    attack_bm: Benchmark = load_benchmark(cfg.attack_benchmark)
    benign_bm: Benchmark = load_benchmark(cfg.benign_benchmark)
    attack_prompts = attack_bm.load(cfg.n_attack_prompts, seed)
    # The coverage was fit on the fit half; score it on the disjoint eval half.
    _, eval_prompts = Benchmark.split(attack_prompts, cfg.fit_frac, seed)
    benign_prompts = benign_bm.load(cfg.n_benign_prompts, seed)

    judge = load_judge_by_name(cfg.judge, args.configs_dir)
    refusal = KeywordRefusalDetector()
    profile = load_profile(cfg, args, menu, seed)
    selector = load_selector_by_name(
        cfg.selector, menu, profile, tau=cfg.tau, c_min=cfg.c_min, configs_dir=args.configs_dir
    )
    attacker = load_attacker_by_name(cfg.attacker, args.configs_dir)

    committed: list[dict] = []
    history: list = []

    for t in range(1, cfg.rounds + 1):
        coverage = selector.select(history=history, round_idx=t)
        committed.append({"round": t, **coverage.to_dict()})

        deployment = Deployment(coverage, judge, refusal, np.random.default_rng(seed + t))
        records = run_round(
            coverage=coverage,
            attack_prompts=eval_prompts,
            benign_prompts=benign_prompts,
            deployment=deployment,
            attacker=attacker,
            budget=cfg.attack_budget,
            round_idx=t,
            seed=seed,
            output_path=out_dir / f"round{t}" / "results.jsonl",
            resume=args.resume,
        )
        # The defender's history retains everything; the attacker's is its
        # response-projection, which is enforced at the Attacker interface.
        history.extend(records)

    if cfg.held_out_attacker:
        # Evaluates the committed coverage, never selects it: run after the loop,
        # against the final coverage only.
        logger.info(f"Held-out evaluation with attacker '{cfg.held_out_attacker}'")
        held_out = load_attacker_by_name(cfg.held_out_attacker, args.configs_dir)
        final_coverage = selector.select(history=history, round_idx=cfg.rounds)
        deployment = Deployment(
            final_coverage, judge, refusal, np.random.default_rng(seed + 999)
        )
        run_round(
            coverage=final_coverage,
            attack_prompts=eval_prompts,
            benign_prompts=[],
            deployment=deployment,
            attacker=held_out,
            budget=cfg.attack_budget,
            round_idx=cfg.rounds + 1,
            seed=seed,
            output_path=out_dir / "held_out" / "results.jsonl",
            resume=args.resume,
        )

    write_json({"experiment": cfg.model_dump(), "coverages": committed}, out_dir / "coverages.json")
    logger.info(f"Wrote {out_dir / 'coverages.json'}")


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment)
    apply_overrides(cfg, args)

    menu = ResourceMenu.from_name(cfg.menu, args.configs_dir)
    logger.info(
        f"Game: menu={cfg.menu} (|Q|={len(menu)}) selector={cfg.selector} "
        f"attacker={cfg.attacker} tau={cfg.tau} rounds={cfg.rounds}"
    )

    for seed in cfg.seeds:
        run_seed(cfg, args, menu, seed)


if __name__ == "__main__":
    main()
