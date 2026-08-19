#!/bin/bash
# Phase 2 -- commit coverages and attack them.
# Usage: bash run_experiments.sh
# Requires: run from the project root on a klogin* (Killarney) or fir login node, AFTER run_probe.sh has
#           finished (every selector but `uniform` needs the menu profile).
#
# Uncomment the blocks for the research questions you want, then run.
# Job naming: pr_<rq>_<selector>_<attacker>_tau<tau>_s<seed>
#
# To add a condition, copy a line and change one override. The config stays
# fixed; all variation is CLI overrides, so a run's identity is legible from its
# job name.

set -e

source setup/start_env.sh

BASE="python scripts/run_game.py --output-dir $PROTEUS_OUTPUT_DIR --resume"

# =============================================================================
# DEV -- smoke test end to end. Run this first.
# =============================================================================

# submit "ex_dev_s42" "$BASE --experiment configs/experiments/dev.yaml --seeds 42"

# =============================================================================
# RQ1 -- Randomized vs the best deterministic configuration
# Does randomized deployment improve robustness? How about helpfulness?
# The comparison is at matched helpfulness, so tau is swept.
# =============================================================================

RQ1="$BASE --experiment configs/experiments/paper/rq1_randomized_vs_deterministic.yaml"

# --- Deterministic baseline (the point-mass comparator) ---
submit "ex_rq1_det_pair_tau0.7_s2002" "$RQ1 --selector deterministic --tau 0.7 --seeds 2002"
# submit "ex_rq1_det_pair_tau0.8_s2002" "$RQ1 --selector deterministic --tau 0.8 --seeds 2002"
# submit "ex_rq1_det_pair_tau0.9_s2002" "$RQ1 --selector deterministic --tau 0.9 --seeds 2002"

# --- Uniform mixing ---
submit "ex_rq1_uni_pair_tau0.7_s2002" "$RQ1 --selector uniform --tau 0.7 --seeds 2002"
# submit "ex_rq1_uni_pair_tau0.8_s2002" "$RQ1 --selector uniform --tau 0.8 --seeds 2002"
# submit "ex_rq1_uni_pair_tau0.9_s2002" "$RQ1 --selector uniform --tau 0.9 --seeds 2002"

# --- Validation-based weighting ---
submit "ex_rq1_val_pair_tau0.7_s2002" "$RQ1 --selector validation --tau 0.7 --seeds 2002"
# submit "ex_rq1_val_pair_tau0.8_s2002" "$RQ1 --selector validation --tau 0.8 --seeds 2002"
# submit "ex_rq1_val_pair_tau0.9_s2002" "$RQ1 --selector validation --tau 0.9 --seeds 2002"

# --- Minimax over measured payoffs ---
submit "ex_rq1_mm_pair_tau0.7_s2002" "$RQ1 --selector minimax --tau 0.7 --seeds 2002"
# submit "ex_rq1_mm_pair_tau0.8_s2002" "$RQ1 --selector minimax --tau 0.8 --seeds 2002"
# submit "ex_rq1_mm_pair_tau0.9_s2002" "$RQ1 --selector minimax --tau 0.9 --seeds 2002"

# --- Reasoner: prompt-conditional routing (capability-first, reroute on risk) ---
submit "ex_rq1_rsn_pair_tau0.7_s2002" "$RQ1 --selector reasoner --tau 0.7 --seeds 2002"
# submit "ex_rq1_rsn_pair_tau0.8_s2002" "$RQ1 --selector reasoner --tau 0.8 --seeds 2002"
# submit "ex_rq1_rsn_pair_tau0.9_s2002" "$RQ1 --selector reasoner --tau 0.9 --seeds 2002"

# =============================================================================
# RQ3 -- The risk-helpfulness frontier, on the capability-tiered menu.
# Does a mixture reach a point on the frontier no single configuration occupies?
#
# Unlike RQ1 this needs no adaptation penalty: on a capability gradient the
# achievable set is the convex hull of the (Help_q, JB_q) points, deterministic is
# confined to its vertices, and a mixture sits exactly on the tau floor while
# deterministic must overshoot. Add --attacker static to isolate that effect from
# adaptation entirely.
#
# !! THE TAU VALUES BELOW ARE PLACEHOLDERS. Quality-weighted Help compresses the
# scale, so run Phase 1 first, read the four Help values off the probe log, and
# replace these with points that fall STRICTLY BETWEEN adjacent Help values --
# that is exactly where mixing wins. A tau at or below the smallest Help makes the
# floor inert (Selector._feasible_mask silently returns all-ones) and the run will
# look like a null result rather than failing.
#
# Expect uniform/validation/reasoner to track deterministic here: they restrict
# SUPPORT via _feasible_mask, so a tau above the small model's Help deletes members
# 3 and 4 outright. Only minimax treats tau as a mixture constraint. That is a real
# property for deterministic and an implementation artifact for the other two --
# see docs/selectors.md.
# =============================================================================

RQ3="$BASE --experiment configs/experiments/paper/rq3_frontier.yaml"

# submit "ex_rq3_det_pair_tau0.55_s42" "$RQ3 --selector deterministic --tau 0.55 --seeds 42"
# submit "ex_rq3_uni_pair_tau0.55_s42" "$RQ3 --selector uniform       --tau 0.55 --seeds 42"
# submit "ex_rq3_val_pair_tau0.55_s42" "$RQ3 --selector validation    --tau 0.55 --seeds 42"
# submit "ex_rq3_mm_pair_tau0.55_s42"  "$RQ3 --selector minimax       --tau 0.55 --seeds 42"
# submit "ex_rq3_rsn_pair_tau0.55_s42" "$RQ3 --selector reasoner      --tau 0.55 --seeds 42"

# --- Baselines: undefended, and always-de-escalate (fully-wrapped SafeRL) ---
# These are single menu members, so Phase 1 already measured them. No job needed;
# read them out of the profile with:
#   python scripts/inspect_menu.py --menu full --verbose

# =============================================================================
# RQ2 -- Static vs adaptive attacks
# Does randomization narrow the gap between static and adaptive evaluation?
# Each selector needs both halves; the metric is their difference.
# =============================================================================

RQ2="$BASE --experiment configs/experiments/paper/rq2_static_vs_adaptive.yaml"

# --- Deterministic ---
# submit "ex_rq2_det_static_s42"   "$RQ2 --selector deterministic --attacker static --seeds 42"
# submit "ex_rq2_det_template_s42" "$RQ2 --selector deterministic --attacker template --seeds 42"
# submit "ex_rq2_det_pair_s42"     "$RQ2 --selector deterministic --attacker pair --seeds 42"

# --- Minimax ---
# submit "ex_rq2_mm_static_s42"   "$RQ2 --selector minimax --attacker static --seeds 42"
# submit "ex_rq2_mm_template_s42" "$RQ2 --selector minimax --attacker template --seeds 42"
# submit "ex_rq2_mm_pair_s42"     "$RQ2 --selector minimax --attacker pair --seeds 42"

# =============================================================================
# RQ4 -- Portfolio diversity and size
# Does the benefit come from mixing models, mixing mechanisms, or both?
# Needs the ablation menus probed first (see run_probe.sh).
# =============================================================================

RQ4="$BASE --experiment configs/experiments/paper/rq4_portfolio_diversity.yaml"

# submit "ex_rq4_full_s42"         "$RQ4 --menu full --seeds 42"
# submit "ex_rq4_models_only_s42"  "$RQ4 --menu models_only --seeds 42"
# submit "ex_rq4_single_model_s42" "$RQ4 --menu single_model --seeds 42"
# submit "ex_rq4_small_s42"        "$RQ4 --menu small --seeds 42"

# =============================================================================
# RQ6 -- Equilibrium gap and generalization to unseen best-responses
# The held-out attacker evaluates the committed coverage but never selects it.
# =============================================================================

RQ6="$BASE --experiment configs/experiments/paper/rq6_equilibrium_gap.yaml"

# submit "ex_rq6_mm_s42"  "$RQ6 --selector minimax --seeds 42"
# submit "ex_rq6_val_s42" "$RQ6 --selector validation --seeds 42"
# submit "ex_rq6_uni_s42" "$RQ6 --selector uniform --seeds 42"
