#!/usr/bin/env python3
"""Phase 3 -- turn the recorded trials into metrics. CPU only, no GPU needed.

Emits one row per (menu, selector, attacker, tau, seed, round) with ASR,
helpfulness, coverage entropy, support size, attacker FLOPs, and the equilibrium
gap where a held-out run exists.

Usage:
    python scripts/evaluate.py --results-dir $PROTEUS_OUTPUT_DIR --menu full
    python scripts/evaluate.py --results-dir $PROTEUS_OUTPUT_DIR --menu full \\
        --out $PROTEUS_OUTPUT_DIR/analysis/metrics.csv

Writes:
    {results_dir}/analysis/metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd  # noqa: E402

from proteus.metrics.cost import flops_to_first_jailbreak, flops_to_target_asr  # noqa: E402
from proteus.metrics.equilibrium import equilibrium_gap  # noqa: E402
from proteus.metrics.safety import (  # noqa: E402
    asr,
    asr_at_budget,
    coverage_entropy,
    helpfulness,
    support_size,
)
from proteus.utils.io import read_json, read_jsonl  # noqa: E402
from proteus.utils.logging import get_logger  # noqa: E402

logger = get_logger("evaluate")

# Attacker parameter counts, for FLOP accounting. Keyed by the attacker stem.
_ATTACKER_PARAMS_B = {
    "pair": 7.30,
    "pair-gemma": 4.30,
    "grpo": 7.30,
    "search": 4.30,
    "static": 0.0,
    "template": 0.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--menu", help="Only evaluate this menu")
    p.add_argument("--out", help="Output CSV (default: {results_dir}/analysis/metrics.csv)")
    p.add_argument("--target-asr", type=float, default=0.5, help="For the FLOPs-to-target metric")
    return p.parse_args()


def find_runs(results_dir: Path, menu: str | None) -> list[Path]:
    """A run is any directory holding a coverages.json."""
    pattern = f"{menu}/*/*/*/*/coverages.json" if menu else "*/*/*/*/*/coverages.json"
    return sorted(p.parent for p in results_dir.glob(pattern))


def evaluate_run(run_dir: Path, target_asr: float) -> list[dict]:
    meta = read_json(run_dir / "coverages.json")
    cfg = meta["experiment"]
    coverages = {c["round"]: c for c in meta["coverages"]}

    attacker_params = _ATTACKER_PARAMS_B.get(cfg["attacker"], 0.0)
    rows = []

    for round_dir in sorted(run_dir.glob("round*")):
        t = int(round_dir.name.replace("round", ""))
        records = read_jsonl(round_dir / "results.jsonl")
        if not records:
            continue

        cov = coverages.get(t, {})
        weights = cov.get("weights", {})
        attack_records = [r for r in records if r.split != "benign"]

        row = {
            "menu": cfg["menu"],
            "selector": cfg["selector"],
            "attacker": cfg["attacker"],
            "tau": cfg["tau"],
            "seed": records[0].seed,
            "round": t,
            "asr": asr(records),
            "helpfulness": helpfulness(records),
            "entropy": coverage_entropy(weights) if weights else float("nan"),
            "support": support_size(weights) if weights else 0,
            "n_attack": len(attack_records),
            "n_benign": len(records) - len(attack_records),
            "flops_to_first_jb": flops_to_first_jailbreak(attack_records, attacker_params),
            "flops_to_target_asr": flops_to_target_asr(
                attack_records, attacker_params, target_asr
            ),
        }
        for b in (1, 2, 3, 5, 8):
            row[f"asr_at_{b}"] = asr_at_budget(attack_records, b)

        # The gap the held-out attacker opens against the committed coverage:
        # what the defender left on the table by stopping where it did.
        held_out_path = run_dir / "held_out" / "results.jsonl"
        if held_out_path.exists() and t == max(coverages):
            held_out = read_jsonl(held_out_path)
            row["equilibrium_gap"] = equilibrium_gap(attack_records, held_out)
            row["held_out_asr"] = asr(held_out)
        else:
            row["equilibrium_gap"] = float("nan")
            row["held_out_asr"] = float("nan")

        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs = find_runs(results_dir, args.menu)
    if not runs:
        logger.error(f"No runs found under {results_dir}. Did Phase 2 write anything?")
        sys.exit(1)

    logger.info(f"Evaluating {len(runs)} runs")
    rows = [r for run in runs for r in evaluate_run(run, args.target_asr)]
    df = pd.DataFrame(rows)

    out = Path(args.out) if args.out else results_dir / "analysis" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {out} ({len(df)} rows)")

    cols = ["selector", "attacker", "tau", "asr", "helpfulness", "entropy", "support"]
    summary = df.groupby(["selector", "attacker", "tau"], as_index=False).agg(
        asr=("asr", "mean"),
        helpfulness=("helpfulness", "mean"),
        entropy=("entropy", "mean"),
        support=("support", "mean"),
    )
    print("\n" + summary[cols].to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
