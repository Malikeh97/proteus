# Design notes

Running notes on game dynamics and selector design. Captured for future work.

## `rounds` vs `seeds` (they are not the same kind of repetition)

Both appear in the experiment YAMLs (`ExperimentConfig`), but they answer different
questions:

| | `rounds` | `seeds` |
|---|---|---|
| Relationship | sequential, **dependent** (round `t+1` builds on `t`) | parallel, **independent** replicates |
| Purpose | adversarial adaptation over time | statistical averaging / error bars |
| Question | does robustness survive an *adapting* attacker? | how noisy is the measured robustness? |

A full run is: **for each seed** (independent replicate) **play `rounds` rounds** (one
dependent adaptation trajectory). Collapsing the two would be wrong — rounds within a
seed are one evolving story; seeds repeat that whole story to check reproducibility.

- `rounds: 1` = single-shot attacker (fast; `dev` uses it). More rounds = strictly
  stronger, more adaptive attacker.
- The **seed** controls the experimental-design randomness (`scripts/run_game.py:112-183`):
  which prompts are subsampled (`bm.load(n, seed)`), the fit/eval split
  (`Benchmark.split(..., seed)`), and the per-request config draw
  (`Deployment(..., np.random.default_rng(seed + t))` — note `+ t`, so each round draws a
  distinct but deterministic stream). Served models generate greedily (`do_sample=False`),
  so there is no token-level randomness to average — the seed's effect is purely
  design-level.
- **Phase 1 and Phase 2 must share the seed**: Phase 2 re-derives the eval split with the
  same seed to reconstruct the exact eval half that was disjoint from the fit half Phase 1
  probed. Mismatched seeds break fit/eval disjointness.
- `smoothllm` has its **own** seed (`config.params.get("seed", 0)`, `prompting.py:85`), not
  the experiment seed — its perturbations don't vary across experiment seeds today.

## `history`: threaded but currently unused

`scripts/run_game.py` accumulates every round's records into `history`
(`history.extend(records)`, line 164) and passes it to
`selector.select(history=history, round_idx=t)` (line 144). But **no implemented selector
reads it** — `uniform`, `validation`, `deterministic`, `minimax`, the `purple` stub, and
`reasoner` all ignore the argument. So today every selector is **open-loop**: `c_t` is a
function of the static Phase-1 profile (plus the current prompt, for `reasoner`), identical
round to round for the static selectors.

What `history` carries if used (base.py:19): the **defender's full record** — `c_s`
(committed coverages), `q_s` (drawn configs), `y_s` (responses). Deliberately richer than
the attacker's view (responses only). That asymmetry — defender sees everything, attacker
sees only responses — is the point of keeping a separate `history`.

## Which selector *should* use `history`

- **`minimax` — primary.** It solves `min_c max_x c·J` over payoffs `J(q,x)` measured once
  in Phase 1 on the **fit** prompts. As the attacker adapts across rounds it produces
  attacks not in that fit set, so the fixed `J` goes stale. `history` provides the realized
  attacks / outcomes to re-solve each round over an **accumulating payoff matrix** (fit
  columns + newly-revealed attack columns). This is the textbook **double-oracle /
  fictitious-play** structure for security games. Most principled use of `history` here.
- **`purple` — designed for it (stub).** The anticipatory selector: its risk kernel
  `ρ_q(x)` scores configs against attacks in a neighborhood of what it expects. `history`
  supplies the realized attacks as reference points for that kernel. Should consume
  `history` by construction when implemented.
- **`reasoner` — secondary.** Adaptation is *per-prompt* (risk of the current input), not
  per-round. `history` could recalibrate the risk→tilt mapping (which prompts actually
  broke through), but that's a refinement, not its core mechanism. Currently
  history-independent.
- **`uniform` / `deterministic` / `validation` — keep static, on purpose.** These are the
  **non-adaptive controls**. Their value is being fixed comparators; making them adapt
  would destroy the "does closing the loop help?" contrast that adaptive-minimax is meant
  to demonstrate.

### Proposed next feature: adaptive (round-closed-loop) minimax

Turn `minimax` into the history-consuming defender:
1. From `history`, reconstruct realized `J` for the attacks seen in prior rounds
   (`q_s`, `x_s`, outcome).
2. Append those as new columns to the Phase-1 fit payoff matrix.
3. Re-solve the existing LP over the accumulated matrix each round.

Plumbing already exists (`history.extend(records)` → `select(history=...)`); the work is
the reconstruction + matrix append + re-solve. RQ payoff: directly measures whether a
closed-loop defender beats the open-loop static solve as the attacker adapts. Keep the
baselines static so the contrast is clean.

## Current `reasoner` (implemented)

Prompt-conditional selector. An LLM risk classifier scores each incoming prompt in [0,1]
(`_risk_score`, dedicated LLM judge, model stem via `params.risk_model`). Coverage
interpolates in log-space between two measured payoffs:

```
w = help^(1 - risk) · safety^(β · risk)
```

- `risk = 0` → `w ∝ help` (start with the strong/most-helpful model)
- `risk = 1` → `w ∝ safety^β` (reroute to the safest configs)
- `c_min` floor keeps every feasible config in the support → soft tilt, not a hard route;
  randomization (and thus non-identifiability by the attacker) is preserved.

Not history-adaptive (see above).

## `seeds` × `rounds`: the loop structure

The two loops are nested (`scripts/run_game.py`): `for seed in cfg.seeds` (outer) →
`for t in range(1, rounds+1)` (inner). So there are **`#seeds × #rounds` committed
coverages / attack phases** — but the axes are not interchangeable:

- **seed** = outer, *independent* replicate. Each seed is a fresh, self-contained game
  (its own prompt draw, fit/eval split, sampling stream), written to `.../{seed}/`.
- **round** = inner, *sequential* adaptation. Rounds within a seed are one dependent
  trajectory (round `t+1` follows the attacker that adapted in round `t`).

`10 seeds × 1 round ≠ 1 seed × 10 rounds`: the first is 10 single-shot games, the second
is one game with an attacker that adapts 10 times.

Each game-round is itself not one iteration — it runs over **every eval prompt**, each with
up to **`attack_budget`** refinement steps (early-stop on first jailbreak). So:

```
attack trials   = #seeds × #rounds × #eval_prompts
served requests ≈ attack trials × (steps until jailbreak or budget)   (+ benign trials)
```

The `tau` values / selectors / attackers in `run_experiments.sh` are **separate submitted
jobs**, not inner iterations — each `submit` line is its own `#seeds × #rounds` run.

## RQ1: how the "static" (deterministic) baseline is run

`configs/experiments/paper/rq1_randomized_vs_deterministic.yaml` — "does randomized
deployment beat the best deterministic configuration?"

- The **static setup is the `deterministic` selector**, not a separate config file. The
  YAML defaults `selector: uniform`, but the comparison is produced by running the *same*
  config once per selector via `--selector` (see the RQ1 block in `run_experiments.sh`):
  `deterministic` (static comparator) vs `uniform` / `validation` / `minimax` / `reasoner`
  (randomized), all at **matched tau** (swept 0.7/0.8/0.9).
- `deterministic` returns a **point mass**: all coverage mass on the single lowest-jailbreak
  configuration that clears the `tau` helpfulness floor. Entropy 0, support 1 — one fixed
  config serves every request, no randomization. That is the "best deterministic
  model-defense configuration."
- Clean comparison: everything else is held identical (adaptive `pair` attacker, prompts,
  split, seed, rounds); only randomized-vs-static differs. Sweeping `tau` traces each
  selector's safety–helpfulness frontier, and the `frontier` plot shows whether the
  randomized curve dominates the deterministic one.

**Known bug in that file:** line 21 reads `attack_budget: 10ss` — `attack_budget` is an
`int`, so this fails pydantic validation and the experiment won't load. Should be
`attack_budget: 10`.
