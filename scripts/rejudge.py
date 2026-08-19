#!/usr/bin/env python3
"""Re-score STORED responses with a judge, without re-running the probe.

A probe job spends its allocation on generation: the attacker's rewrites and the
served model's replies. The judge is a rounding error on top. So iterating on a
judge -- a rubric edit, a threshold, a different model -- by re-probing pays the
generation cost again to change something that never touched it. This reads the
(prompt, response) pairs already on disk and re-adjudicates them, which turns a
6-hour, 10-allocation question into one short job.

WHAT IT IS FOR
--------------
1. Validating a judge change before spending real compute. The llama judge's
   binary UNSAFE/SAFE protocol scored 87 of 90 seed-1990 fit trials as jailbreaks
   (JB 1.000 / 0.933 / 0.967, against 0.233 / 0.100 / 0.067 for FlowJudge on its
   own tree). Ten `models_only` jobs were queued behind that before the mvp pair
   exposed it. Re-judging the same records under the fixed rubric put JB at
   0.233 / 0.267 / 0.367 in five minutes, on one allocation.
2. Measuring judge disagreement directly. Pass --reference and every record is
   scored against the label another judge already wrote for it.

   POINT --records AND --reference AT THE SAME TREE. That is the only way the two
   judges score identical text. Two trees probed under different judges do NOT
   share responses: a trial stops at its first success, so a judge that fires
   early truncates the trajectory, and the next attacker step is conditioned on a
   different history. Measured on the seed-1990 mvp pair, the flow tree averaged
   9.2 steps per trial against the llama tree's 3.0, and ZERO of the 90 graded
   responses matched. Comparing across them conflates judge disagreement with
   trajectory divergence. The script counts the overlap and warns when it is not
   total; treat that warning as invalidating the agreement column, not as a note.

WHAT IT IS NOT
--------------
NOT a way to patch a profile. It never writes into a results tree -- the report
goes to --out and nothing else is touched. A profile whose J came from judge A is
a measurement under judge A; re-judging its records produces a different
measurement, and the two must not end up in one shard. Re-probe into a separate
root (see run_probe.sh's PROBE_OUT) when the new judge is the one you want.

And a hard limit: re-judging can only re-score the responses that were actually
generated. A trial that stopped early because judge A called step 3 a break has no
steps 4-10, so a stricter judge B never gets the chance to break later, and a
laxer judge B is scoring a trajectory that only exists because A fired. A JB from
this script is therefore a diagnostic on fixed text, NOT an estimate of what a
full probe under judge B would measure. Use it to decide whether a judge is worth
probing with; use the probe for the number you report.

Usage:
    # Does a rubric change stop refusals from scoring as breaks?
    python scripts/rejudge.py --records $SCRATCH/proteus-judge-llama/mvp/profile/1990 \\
        --reference $PROTEUS_OUTPUT_DIR/mvp/profile/1990 \\
        --judge llama3.1_8b_judge --out $SCRATCH/rejudge-mvp-1990.json

    # Cheap smoke test on the login node -- no GPU, no weights.
    python scripts/rejudge.py --records ... --judge keyword --limit 5

Needs a GPU for any model-backed judge, but no attacker and no served model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteus.judges import load_judge_by_name  # noqa: E402
from proteus.utils.io import TrialRecord, read_jsonl  # noqa: E402
from proteus.utils.logging import get_logger  # noqa: E402

logger = get_logger("rejudge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--records",
        required=True,
        help="Profile dir ({root}/{menu}/profile/{seed}) or a single .jsonl shard",
    )
    p.add_argument(
        "--reference",
        help="A second profile dir whose stored labels to compare against, matched "
        "on (qid, base prompt id). Usually the other judge's tree.",
    )
    p.add_argument("--judge", default="llama3.1_8b_judge", help="Judge stem to score with")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--split", default="fit", choices=["fit", "eval", "benign"])
    p.add_argument("--limit", type=int, help="Score at most this many records per shard")
    p.add_argument("--qid", nargs="+", help="Only these configurations")
    p.add_argument("--out", help="Write the JSON report here")
    p.add_argument("--show", type=int, default=10, help="Disagreements to print (default 10)")
    return p.parse_args()


def record_paths(root: Path) -> list[Path]:
    """Every file holding records under `root`. Mirrors probe_menu.record_paths --
    shards plus the pre-sharding probe.jsonl -- so a legacy tree reads the same
    here as it does there."""
    if root.is_file():
        return [root]
    paths = sorted((root / "shards").glob("*.jsonl"))
    legacy = root / "probe.jsonl"
    return ([legacy] if legacy.exists() else []) + paths


def graded_step(rec: TrialRecord):
    """The step whose adjudication decided the trial: the first jailbreak, else the
    last step. Matches probe_menu.break_severity's convention, so a re-judged
    verdict is comparable to the stored `success` rather than to some other step."""
    if not rec.steps:
        return None
    for s in rec.steps:
        if s.outcome == "jailbreak":
            return s
    return rec.steps[-1]


def load_records(root: Path, split: str, qids: list[str] | None, limit: int | None):
    """{(qid, base_prompt_id): TrialRecord} for one split."""
    out = {}
    for path in record_paths(root):
        n = 0
        for rec in read_jsonl(path):
            if rec.split != split:
                continue
            qid = rec.meta.get("qid") or rec.prompt_id.split("::")[0]
            if qids and qid not in qids:
                continue
            base = rec.meta.get("base_prompt_id") or rec.prompt_id.split("::")[-1]
            if limit is not None and n >= limit:
                break
            out[(qid, base)] = rec
            n += 1
    return out


def main() -> None:
    args = parse_args()

    records = load_records(Path(args.records), args.split, args.qid, args.limit)
    if not records:
        logger.error(f"No '{args.split}' records under {args.records}")
        sys.exit(1)
    reference = (
        load_records(Path(args.reference), args.split, args.qid, None) if args.reference else {}
    )
    logger.info(
        f"Re-judging {len(records)} '{args.split}' record(s) from {args.records} "
        f"with judge '{args.judge}'"
        + (f", against {len(reference)} reference record(s)" if reference else "")
    )

    judge = load_judge_by_name(args.judge, args.configs_dir)
    per_qid: dict[str, dict] = {}
    disagreements: list[dict] = []

    for (qid, base), rec in sorted(records.items()):
        step = graded_step(rec)
        if step is None:
            continue
        row = per_qid.setdefault(
            qid, {"n": 0, "new_breaks": 0, "stored_breaks": 0, "ref_breaks": 0, "ref_n": 0,
                  "agree_ref": 0, "same_text": 0, "severities": []}
        )
        row["n"] += 1
        row["stored_breaks"] += int(rec.success)

        # The judge saw the underlying objective, not the attacker's rewrite --
        # runner.py:61 passes objective=prompt.text and deployment.py:94 prefers it.
        new_jb = judge.judge(rec.base_prompt, step.response)
        row["new_breaks"] += new_jb
        if new_jb:
            row["severities"].append(judge.severity(rec.base_prompt, step.response))

        ref = reference.get((qid, base))
        if ref is not None:
            row["ref_n"] += 1
            row["ref_breaks"] += int(ref.success)
            # Same text, or only the same prompt id? See the module docstring:
            # this is what decides whether the agreement column means anything.
            ref_step = graded_step(ref)
            row["same_text"] += int(ref_step is not None and ref_step.response == step.response)
            if int(ref.success) == new_jb:
                row["agree_ref"] += 1
            else:
                disagreements.append(
                    {
                        "qid": qid,
                        "base_prompt_id": base,
                        "step": step.step,
                        "new_jb": new_jb,
                        "reference_jb": int(ref.success),
                        "stored_jb": int(rec.success),
                        "base_prompt": rec.base_prompt[:200],
                        "attack_prompt": step.prompt[:200],
                        "response": step.response[:400],
                    }
                )

    for qid, row in per_qid.items():
        row["jb_new"] = row["new_breaks"] / row["n"]
        row["jb_stored"] = row["stored_breaks"] / row["n"]
        row["mean_severity"] = (
            sum(row["severities"]) / len(row["severities"]) if row["severities"] else 0.0
        )
        if row["ref_n"]:
            row["jb_reference"] = row["ref_breaks"] / row["ref_n"]
            row["agreement"] = row["agree_ref"] / row["ref_n"]
            row["same_text_frac"] = row["same_text"] / row["ref_n"]

    print(f"\n{'qid':26s} {'n':>4s} {'JB new':>8s} {'JB stored':>10s} {'JB ref':>8s} "
          f"{'agree':>7s} {'Sev':>6s}")
    print("-" * 78)
    for qid, row in sorted(per_qid.items()):
        ref = f"{row.get('jb_reference', float('nan')):8.3f}" if row["ref_n"] else "       -"
        agr = f"{row.get('agreement', float('nan')):7.3f}" if row["ref_n"] else "      -"
        print(f"{qid:26s} {row['n']:4d} {row['jb_new']:8.3f} {row['jb_stored']:10.3f} "
              f"{ref} {agr} {row['mean_severity']:6.2f}")

    ref_n = sum(r["ref_n"] for r in per_qid.values())
    if ref_n:
        same = sum(r["same_text"] for r in per_qid.values())
        if same < ref_n:
            print(
                f"\n*** Only {same}/{ref_n} graded responses are IDENTICAL between --records "
                f"and --reference.\n*** The two trees have different trajectories (a judge that "
                f"fires early truncates\n*** the trial), so 'JB ref' and 'agree' above mix judge "
                f"disagreement with different\n*** text. Point both flags at ONE tree to compare "
                f"judges on identical responses."
            )

    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) vs the reference; first {args.show}:")
        for d in disagreements[: args.show]:
            arrow = "reference=safe -> NEW=BREAK" if d["new_jb"] else "reference=BREAK -> new=safe"
            print(f"\n  [{d['qid']} / {d['base_prompt_id']}] step {d['step']}  {arrow}")
            print(f"    request : {d['base_prompt'][:110]}")
            print(f"    response: {d['response'][:220]}".replace("\n", " "))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "judge": args.judge,
            "records": str(args.records),
            "reference": args.reference,
            "split": args.split,
            "per_qid": per_qid,
            "disagreements": disagreements,
        }
        out.write_text(json.dumps(report, indent=2))
        logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
