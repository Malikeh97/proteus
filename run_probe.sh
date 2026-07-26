#!/bin/bash
# Phase 1 -- measure J(q, x) and Help(q) for every configuration in a menu.
# Usage: bash run_probe.sh
# Requires: run from the project root on a klogin* (Killarney) or fir login node.
#
# Everything downstream needs this: the selectors optimise over these measured
# payoffs. Uncomment the menus you want and run. Re-running is safe -- submit()
# skips jobs that are queued, running, or completed in the last 2 days.
#
# One job per (menu, seed). Cost scales as |Q| x (n_fit x budget + n_benign): the
# full menu is |Q|=8 x (50 fit prompts x 10 steps + 100 benign) = 4800 requests.
#
# AFTER these finish, audit the menu before submitting Phase 2 -- it is seconds on
# the login node and tells you whether Phase 2 can be informative at all:
#     python scripts/audit_menu.py --menu full --seed 2
# Exit code 1 means the menu cannot support the claim; fix it and re-probe rather
# than spending an allocation measuring a config artefact. See docs/menu-design.md.

set -e

source setup/start_env.sh

BASE="python scripts/probe_menu.py --output-dir $PROTEUS_OUTPUT_DIR --resume"

# =============================================================================
# DEV -- smoke test. No guards, keyword judge, 4 prompts. Run this first.
# =============================================================================

# submit "pr_dev_s1997" "$BASE --experiment configs/experiments/dev.yaml --seeds 1997"

# =============================================================================
# FULL MENU -- |Q| = 8. The ISO-SAFETY menu: members equally safe on their own,
# differing by SUBSTITUTION along one axis at a time (stage / provenance /
# boundary / alignment / mechanism class). See configs/menus/full.yaml.
# =============================================================================

# submit "pr_full_s1997" "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 1997"
# submit "pr_full_s2"    "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 2"
submit "pr_full_s42"   "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 42"

# =============================================================================
# GRADIENT MENU -- the previous `full`, preserved verbatim. The CONTRAST ARM:
# a capability/safety gradient maximises the spread of JB_q, which is exactly the
# dilution penalty a mixture pays. Probing both measures that price directly, and
# is the sharpest version of RQ4 ("does portfolio diversity matter?").
# =============================================================================

submit "pr_gradient_s2" "$BASE --experiment configs/experiments/base.yaml --menu gradient --seeds 42"

# =============================================================================
# SMALL MENU -- |Q| = 8. The Fallback menu.
# =============================================================================

# submit "pr_small_s1997" "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 1997"
# submit "pr_small_s2"    "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 2"
# submit "pr_small_s42"   "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 42"

# =============================================================================
# ABLATION MENUS -- isolate the model axis vs the mechanism axis (RQ4)
# =============================================================================

submit "pr_models_only_s2"  "$BASE --experiment configs/experiments/base.yaml --menu models_only --seeds 42"
submit "pr_single_model_s2" "$BASE --experiment configs/experiments/base.yaml --menu single_model --seeds 42"
