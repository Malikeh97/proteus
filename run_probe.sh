#!/bin/bash
# Phase 1 -- measure J(q, x), S(q, x) and Help(q) for every configuration in a menu.
# Usage: bash run_probe.sh
# Requires: run from the project root on a klogin* (Killarney) or fir login node.
#
# Everything downstream needs this: the selectors optimise over these measured
# payoffs. Uncomment the menus you want and run. Re-running is safe -- submit()
# skips jobs that are queued, running, or completed in the last 2 days.
#
# ONE JOB PER (MODEL, SEED), not per (menu, seed): configurations are probed
# independently, so the menu fans out across the cluster instead of looping inside
# one 23h allocation. Cost per job is (n_fit x budget + n_benign) x (configs on that
# model) -- for mvp, 30 fit x 10 steps + 30 benign = 330 requests, ~3h, versus ~15h
# for the whole menu serially. A job holding several configurations of one model
# also loads that model once (menu/models/__init__.py caches it).
#
# Records go to profile/{seed}/shards/{qid}.jsonl, one file per configuration --
# concurrent jobs must never append to a shared file. Whichever job finishes last
# merges every shard into profile.json; if a job dies, do the merge by hand:
#     python scripts/probe_menu.py --experiment configs/experiments/base.yaml \
#         --menu mvp --seeds 17 --output-dir $PROTEUS_OUTPUT_DIR --build-profile-only
#
# AFTER these finish, audit the menu before submitting Phase 2 -- it is seconds on
# the login node and tells you whether Phase 2 can be informative at all:
#     python scripts/audit_menu.py --menu full --seed 2
# Exit code 1 means the menu cannot support the claim; fix it and re-probe rather
# than spending an allocation measuring a config artefact. See docs/menu-design.md.

set -e

source setup/start_env.sh

BASE="python scripts/probe_menu.py --resume"

# Per-block knobs, reset by every block that touches them. PROBE_OUT is the root
# of the results tree; PROBE_EXTRA appends CLI overrides. A variant that changes
# what gets MEASURED (judge, attacker, budget) must also change PROBE_OUT: the
# shard path is {root}/{menu}/profile/{seed}/, with no judge in it, so two judges
# at one root would interleave their verdicts in the same file and --resume would
# read the other judge's trials as already done.
PROBE_OUT="$PROTEUS_OUTPUT_DIR"
PROBE_EXTRA=""

# submit_menu <job-prefix> <experiment> <menu> <seed>...
# One job per (model, seed). The model list comes from the menu YAML itself, so
# this never drifts from configs/menus/.
function submit_menu() {
    local prefix="$1" experiment="$2" menu="$3"; shift 3
    local models seed model job
    models=$(python scripts/inspect_menu.py --menu "$menu" --list-models) || return 1
    for seed in "$@"; do
        for model in $models; do
            # SLURM job names go into log filenames; model_ids carry '.' but nothing worse.
            job="${prefix}_s${seed}_${model//[^A-Za-z0-9._-]/_}"
            submit "$job" \
                "$BASE --output-dir $PROBE_OUT --experiment $experiment --menu $menu \
                 --seeds $seed --model $model $PROBE_EXTRA"
        done
    done
}

# =============================================================================
# DEV -- smoke test. No guards, keyword judge, 4 prompts. Run this first.
# =============================================================================

# submit_menu "pr_dev" configs/experiments/dev.yaml dev 1997

# =============================================================================
# MVP MENU -- |Q| = 3. Three bare models, NO wrappers: the model axis alone,
# before any defense mechanism is layered on. base.yaml's default menu and the
# cheapest real run -- 3 concurrent jobs of (30 fit x 10 steps + 30 benign).
#
# This is also the SEVERITY GATE. FlowJudge maps rubric 3/4/5 to severity 5/8/10,
# so S(q, x) has three distinct nonzero values and nothing guarantees the menu
# spreads across them. Check the `Sev=` column in the probe log varies ACROSS
# configurations before building anything on the graded objective: if every break
# grades alike, S is a scalar multiple of J, and since the minimax LP's optimum is
# invariant to positive scaling of its payoff matrix, `minimax_graded` is not
# approximately but exactly `minimax`. See configs/selectors/minimax_graded.yaml.
# =============================================================================

# submit_menu "pr_mvp" configs/experiments/base.yaml mvp 1997 2
# submit_menu "pr_mvp" configs/experiments/base.yaml mvp 1990

# =============================================================================
# JUDGE SENSITIVITY -- the mvp probe repeated with the Llama-3.1-8B judge in
# place of FlowJudge, same menu, same seed, same prompts. How much of JB_q is the
# judge rather than the configuration? See configs/judges/llama3.1_8b_judge.yaml
# for why this one is a comparison arm and not a headline number.
#
# Writes to its own results ROOT so it cannot touch the FlowJudge tree; point the
# downstream tools at it explicitly:
#     python scripts/audit_menu.py --menu mvp --seed 1990 \
#         --output-dir $SCRATCH/proteus-judge-llama
#
# The llama judge grades severity and helpfulness on the same 1-5 scales as
# FlowJudge, so the two profiles carry the same columns and Help(q) is the
# quality-weighted definition in both trees -- see configs/judges/llama3.1_8b_judge.yaml.
# =============================================================================

# The prefix carries a version because the RUBRIC is not in the output path
# either. Superseded runs, archived beside the live tree:
#   pr_mvpllama  -> proteus-judge-llama.longprompt  (the long safety rubric)
#   pr_mvpllama2 -> proteus-judge-llama.nograded    (no severity or quality)
# Bump it whenever judges/llm_judge.py changes what the judge is asked, or a
# completed job of the same name gets skipped and the tree silently mixes rubrics.
# PROBE_OUT="$SCRATCH/proteus-judge-llama"
# PROBE_EXTRA="--judge llama3.1_8b_judge"
# submit_menu "pr_mvpllama3" configs/experiments/base.yaml mvp 1990
# PROBE_OUT="$PROTEUS_OUTPUT_DIR"
# PROBE_EXTRA=""

# =============================================================================
# FULL MENU -- |Q| = 8. The ISO-SAFETY menu: members equally safe on their own,
# differing by SUBSTITUTION along one axis at a time (stage / provenance /
# boundary / alignment / mechanism class). See configs/menus/full.yaml.
# =============================================================================

# submit_menu "pr_full" configs/experiments/base.yaml full 1997 2 42

# =============================================================================
# GRADIENT MENU -- the previous `full`, preserved verbatim. The CONTRAST ARM:
# a capability/safety gradient maximises the spread of JB_q, which is exactly the
# dilution penalty a mixture pays. Probing both measures that price directly, and
# is the sharpest version of RQ4 ("does portfolio diversity matter?").
# =============================================================================

# submit_menu "pr_gradient" configs/experiments/base.yaml gradient 42

# =============================================================================
# FRONTIER MENU -- |Q| = 4. The CAPABILITY-TIERED menu: a deliberate monotone
# chain in both payoffs, for the risk-helpfulness frontier question (RQ3). Needs
# the quality-weighted Help metric to mean anything -- check the Help values are
# well separated before running Phase 2. See configs/menus/frontier.yaml.
#
# Audit it with the spread check relaxed; checks 1 and 2 fail here BY DESIGN:
#   python scripts/audit_menu.py --menu frontier --seed 42 --tau <t> --max-spread 1.0
# Check 5's `dilution` should come out NEGATIVE. That is the result.
# =============================================================================

# submit_menu "pr_frontier" configs/experiments/paper/rq3_frontier.yaml frontier 42

# =============================================================================
# SMALL MENU -- |Q| = 8. The Fallback menu.
# =============================================================================

# submit_menu "pr_small" configs/experiments/base.yaml small 1997 2 42

# =============================================================================
# ABLATION MENUS -- isolate the model axis vs the mechanism axis (RQ4)
# =============================================================================

# submit_menu "pr_single_model" configs/experiments/base.yaml single_model 42

# =============================================================================
# MODELS_ONLY -- |Q| = 10, EVERY served model, pair attacker, BOTH judges.
#
# The reference per-model J/S/Help table. Probed once and kept: later menus pick
# their members from these numbers instead of re-measuring, and because probing
# uses the FIT split while run_game.py uses the held-out EVAL split, choosing
# members off this table does not select on Phase 2 data.
#
# Seed 42, matching run_experiments.sh -- every Phase 2 condition there runs at
# --seeds 42, and run_game.py derives the profile path from the menu and seed, so
# a profile at any other seed would simply not be found.
#
# BOTH JUDGES, two roots, 10 jobs each. Same menu, seed, prompts and attacker, so
# the pair isolates the judge. Since judges/llm_judge.py now grades severity and
# helpfulness on FlowJudge's scales, the two profiles carry the same columns and
# are compared directly rather than one being J-only.
#
#     flow : $PROTEUS_OUTPUT_DIR/models_only/profile/42/profile.json
#     llama: $SCRATCH/proteus-judge-llama/models_only/profile/42/profile.json
#
# Expect a WIDE spread -- 1.5B to 14B is a capability gradient, and the audit's
# iso-safety and anti-chain checks will fail on it by construction. That is the
# point: see the menu YAML. gemma3-12b-abliterated doubles as a positive control,
# and should land near JB = 1; if it does not, suspect the attacker or the judge
# before believing any other row.
# =============================================================================

# CHECK FOR A LEGACY probe.jsonl BEFORE RE-PROBING AN OLD SEED. --resume reads
# {root}/{menu}/profile/{seed}/probe.jsonl (pre-sharding runs) as already-done
# trials, and record_paths() folds it into the merge. The seed-42 file here was
# measured in July under judge=llama3.1_8b_judge with n_benign=100, so four
# members resume-skipped their whole fit split and the merged profile would have
# mixed two judges and two Help definitions. Archived to 42.legacy-llama-jul25/;
# those four were re-probed as pr_mo2_s42_*.
# submit_menu "pr_mo" configs/experiments/base.yaml models_only 42

# Superseded llama arms, and why the prefix carries a version -- the RUBRIC is not
# in the output path, so a job name is the only place it is recorded:
#   pr_mollama  cancelled before starting. Binary UNSAFE/SAFE verdict from a
#               separate pass, which scored 87/90 mvp trials as breaks. See the
#               comment above _SAFETY_SYSTEM_PROMPT in judges/llm_judge.py.
#   pr_mollama2 verdict derived from the graded rubric at threshold 3, exactly as
#               FlowJudge does, so the judge MODEL is the only difference between
#               the two arms.
# VERIFY A JUDGE CHANGE BEFORE SUBMITTING TEN JOBS BEHIND IT:
#   python scripts/rejudge.py --records $SCRATCH/proteus-judge-llama/mvp/profile/1990 \
#       --reference $PROTEUS_OUTPUT_DIR/mvp/profile/1990 --judge llama3.1_8b_judge
# PROBE_OUT="$SCRATCH/proteus-judge-llama"
# PROBE_EXTRA="--judge llama3.1_8b_judge"
# submit_menu "pr_mollama2" configs/experiments/base.yaml models_only 1997
# PROBE_OUT="$PROTEUS_OUTPUT_DIR"
# PROBE_EXTRA=""



# PROBE_OUT="$SCRATCH/proteus-judge-flow"
# submit_menu "pr_moflow" configs/experiments/base.yaml models_only 1997
# PROBE_OUT="$PROTEUS_OUTPUT_DIR"
# PROBE_EXTRA=""
