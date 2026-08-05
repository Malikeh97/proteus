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

BASE="python scripts/probe_menu.py --output-dir $PROTEUS_OUTPUT_DIR --resume"

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
                "$BASE --experiment $experiment --menu $menu --seeds $seed --model $model"
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
submit_menu "pr_mvp" configs/experiments/base.yaml mvp 1990

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

# submit_menu "pr_models_only"  configs/experiments/base.yaml models_only 42
# submit_menu "pr_single_model" configs/experiments/base.yaml single_model 42
