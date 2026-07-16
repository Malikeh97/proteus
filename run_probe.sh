#!/bin/bash
# Phase 1 -- measure J(q, x) and Help(q) for every configuration in a menu.
# Usage: bash run_probe.sh
# Requires: run from the project root on a klogin* node.
#
# Everything downstream needs this: the selectors optimise over these measured
# payoffs. Uncomment the menus you want and run. Re-running is safe -- submit()
# skips jobs that are queued, running, or completed in the last 2 days.
#
# One job per (menu, seed). Cost scales as |Q| x n_prompts x budget, so the full
# menu is the expensive one: |Q|=28 x 50 fit prompts x 5 steps = 7000 requests.

set -e

source setup/start_env.sh

BASE="python scripts/probe_menu.py --output-dir $PROTEUS_OUTPUT_DIR --resume"

# =============================================================================
# DEV -- smoke test. No guards, keyword judge, 4 prompts. Run this first.
# =============================================================================

submit "pr_dev_s1997" "$BASE --experiment configs/experiments/dev.yaml --seeds 1997"

# =============================================================================
# FULL MENU -- |Q| = 28. The Sec. 4 resource menu.
# =============================================================================

# submit "pr_full_s1997" "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 1997"
# submit "pr_full_s2"    "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 2"
# submit "pr_full_s42"   "$BASE --experiment configs/experiments/base.yaml --menu full --seeds 42"

# =============================================================================
# SMALL MENU -- |Q| = 8. The Fallback menu.
# =============================================================================

# submit "pr_small_s1997" "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 1997"
# submit "pr_small_s2"    "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 2"
# submit "pr_small_s42"   "$BASE --experiment configs/experiments/base.yaml --menu small --seeds 42"

# =============================================================================
# ABLATION MENUS -- isolate the model axis vs the mechanism axis (RQ4)
# =============================================================================

# submit "pr_models_only_s1997"  "$BASE --experiment configs/experiments/base.yaml --menu models_only --seeds 1997"
# submit "pr_single_model_s1997" "$BASE --experiment configs/experiments/base.yaml --menu single_model --seeds 1997"
