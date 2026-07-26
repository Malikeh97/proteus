# How each selector works

A selector answers one question: **given the menu `Q` and the Phase-1 payoffs, what
distribution `c ∈ Δ(Q)` should the defender commit this round?** It is called once per
round from `scripts/run_game.py:144`, returns a `Coverage`, and the game then draws
`q ~ c(x)` *per request* (`Deployment.serve`, `src/proteus/game/deployment.py:56`).

Everything below is in `src/proteus/coverage/selectors/`.

## What every selector is handed

`Selector.__init__` (`base.py:16`) receives exactly five things, and that is the complete
set of what a selection strategy may condition on:

| | Source | Meaning |
|---|---|---|
| `self._menu` | `configs/menus/{menu}.yaml` | `Q`, canonically ordered by `menu.qids` (sorted). Every weight vector is indexed by this order. |
| `self._profile` | `profile.json` (Phase 1) | the measured payoffs `J(q, x)` and `Help(q)` |
| `self._tau` | `--tau` | helpfulness floor |
| `self._c_min` | `--c-min` (default `0.0`) | per-configuration mass floor |
| `history` (in `select`) | accumulated Phase-2 records | prior rounds' `c_s, q_s, y_s` — **no implemented selector reads it** (see `notes.md`) |

Two shared helpers do most of the work:

- **`_require_profile()`** — raises unless Phase 1 ran. Everything but `uniform` calls it.
- **`_feasible_mask()`** (`base.py:47`) — boolean over `menu.qids`, `True` where
  `Help(q) ≥ tau`. This is the *conservative* reading of the floor: the real constraint is
  on the mixture (`c · help ≥ tau`), and restricting the support to individually-feasible
  configs is a sufficient condition for it. `minimax` is the only selector that imposes the
  real constraint instead.

Two behaviours of that mask matter in practice:

- `tau <= 0` or no profile → all-ones, i.e. the floor is inert.
- **If *no* configuration clears `tau`, the mask silently falls back to all-ones.** So a
  `--tau 0.9` run on a menu whose best config helps at 0.85 does not fail — it quietly
  runs with no floor at all. Check the logged `Help` values from Phase 1 before trusting a
  high-`tau` sweep point.

The returned `Coverage` **renormalises** (`StaticCoverage.__init__`, `coverage.py:70`:
`w / w.sum()`). Selectors may therefore return unnormalised weights; the consequence is
that a `c_min` floor applied *after* a normalisation step gets diluted by the
renormalisation (noted per-selector below).

---

## 1. `uniform` — the profile-free control

```python
mask = self._feasible_mask()
w = mask.astype(float)
w = np.maximum(w, self._c_min)
return StaticCoverage(self._menu, w, self.selector_id)
```

Uniform over the configurations that individually clear `tau`. The **only** selector that
runs without `profile.json` — but note it still consults the profile *through* the mask, so
without a profile it is uniform over all of `Q`, and with one it is uniform over the
feasible subset.

Its role is as a control: it isolates "does spreading mass at all help?" from "does
spreading it *cleverly* help?". Any gain `minimax` shows over `uniform` is attributable to
the optimisation rather than to randomisation per se. `H(c) = log|feasible|`, the maximum
achievable at that support size.

**Caveat.** `np.maximum(w, self._c_min)` has no `* mask`, unlike `validation` and
`reasoner`. With `c_min > 0` this re-admits configurations the `tau` floor just excluded,
at weight `c_min`. Inert at the default `c_min: 0.0`; a real inconsistency if the floor is
ever turned on.

## 2. `deterministic` — the baseline the paper argues against

```python
jb = np.array([profile.jailbreak_rate(q) for q in self._menu.qids])
jb = np.where(mask, jb, np.inf)          # infeasible configs can never win
w = np.zeros(len(self._menu)); w[int(np.argmin(jb))] = 1.0
```

A **point mass** on the single feasible configuration with the lowest measured jailbreak
rate, where `jailbreak_rate(q)` is the row mean of `J(q, ·)` over the fit prompts.

This is the RQ1 baseline — "the best deterministic model+defense stack" — and the reason it
lives here rather than in a separate script is fairness: it must be chosen from the *same
fit split*, by the *same measurement*, as the mixtures it is compared against. Picking it
post hoc from eval-split ASR would be selection on the test set.

`H(c) = 0` and `support_size = 1` by construction. That makes it the calibration point for
the entropy metric: if `minimax` also returns entropy 0, it has degenerated into this.

Ties break on the first index in sorted-qid order (`np.argmin`), which is arbitrary but
deterministic.

## 3. `validation` — inverse-risk weighting

```python
alpha = params.get("alpha", 1.0)
w = np.power(np.clip(1.0 - jb, 0.0, 1.0), alpha) * mask
w = w / w.sum()
w = np.maximum(w, self._c_min * mask)
```

`w_q ∝ (1 − JB_q)^α` over the feasible set: mass proportional to measured safety, raised to
a sharpness exponent.

`alpha` is the interpolation dial along the whole randomisation axis:

| `alpha` | behaviour |
|---|---|
| `0` | recovers `uniform` |
| `1` (config default) | plain inverse-risk |
| `→ ∞` | recovers `deterministic` |

It is the cheap heuristic mixture — it uses only the **row means** of `J`, never the
per-prompt structure. That is precisely what distinguishes it from `minimax`: two
configurations with the same average jailbreak rate get the same weight here, even if one
fails on exactly the prompts the other blocks. `validation` cannot see complementarity;
`minimax` is built to exploit it. The gap between the two is the empirical value of the
per-prompt payoff matrix.

If every feasible config jailbreaks always (`w.sum() <= 0`), it logs a warning and falls
back to uniform.

**Caveat.** The `c_min` floor is applied *after* `w / w.sum()`, and `StaticCoverage` then
renormalises again — so the realised floor is `c_min / (1 + excess)`, slightly below the
requested one. Inert at `c_min: 0.0`.

## 4. `minimax` — the Stackelberg solve

The one selector that uses `J` in full. It solves

```
min_c  max_x  Σ_q c_q J(q, x)
s.t.   c ∈ Δ(Q),   c · help ≥ tau,   c_q ≥ c_min
```

— the SSG program of Tambe (2011), with the payoff matrix *measured* rather than supplied.
The inner `max` is over the attacker's choice of target prompt: the defender commits first,
the attacker best-responds, and the coverage minimises what that best response is worth.

**As an LP.** The inner max is linearised with an epigraph variable `t`, so the decision
vector is `[c_0 … c_{n-1}, t]` and the objective is just `min t`:

| Block | Rows | Encodes |
|---|---|---|
| `A_ub = [Jᵀ, −1]`, `b_ub = 0` | one per **fit prompt** | `J[:, x] · c ≤ t` for every target `x` — `t` bounds the worst target's jailbreak probability |
| `[−help, 0] ≤ −tau` | 1 (only if `tau > 0`) | the **mixture** helpfulness constraint |
| `A_eq = [1…1, 0] = 1` | 1 | simplex |
| `bounds = [(c_min, 1)]×n + [(0,1)]` | — | per-config mass floor, as a hard LP bound |

Solved with `scipy.optimize.linprog(method="highs")`.

Two things follow from this shape:

- It is the **only** selector where `tau` is a constraint on the *mixture* rather than a
  support restriction — it never calls `_feasible_mask()`. A config helping at 0.5 can
  still carry mass at `tau = 0.7` provided the rest of the mixture compensates. This makes
  `minimax` strictly less conservative than the others at equal `tau`, which is worth
  keeping in mind when reading the frontier plot.
- It is the only selector where `c_min` is an **exact** floor (an LP bound on a vector
  already constrained to sum to 1), not a post-hoc clamp.

**Mixing is an outcome, not an assumption.** An LP optimum sits at a vertex unless the
geometry forces otherwise. If one configuration dominates every column of `J`, the LP
returns that vertex and the "coverage" is a router with `H(c) = 0`. The docstring names
this explicitly as the falsification the coverage-entropy metric exists to catch — the
central claim of the framework does not apply to a point mass. Conversely, mixing appears
exactly when blind spots are complementary (which is what `tests/` checks).

**Failure mode.** `linprog` failure is almost always `tau` exceeding what the menu can
deliver; it logs the achievable maximum `help_vec.max()` and falls back to uniform. A run
that silently reports uniform-like entropy under `--selector minimax` should be checked
against this warning in the job log.

It also logs `res.x[-1]`, the worst-target jailbreak probability — a *predicted* value on
the fit split, directly comparable to the eval-split ASR that Phase 3 reports. A large gap
between them is fit/eval generalisation failure, not a bug.

## 5. `reasoner` — the only prompt-conditional selector *(implemented)*

The only one returning `PromptConditionalCoverage` rather than `StaticCoverage`, i.e. the
only one realising the `c: X × H → Δ(Q)` dependence on `x` that the formal setup admits.

**Step 1 — score the prompt.** An LLM (`params.risk_model`, default `qwen2.5_3b`, loaded
lazily on first use) is asked for a one-word verdict `NONE | LOW | MEDIUM | HIGH`, mapped to
`{0, 0.34, 0.67, 1.0}`. The 4-way ordinal is deliberate: models are far more reliable at a
coarse rating than at emitting a calibrated float. An unparseable verdict logs a warning and
defaults to `0.5`.

**Step 2 — interpolate between the two payoffs in log space.**

```python
help_vec = np.clip(profile.help_array(qids), 1e-6, None) * mask   # capability
safety   = np.clip(1.0 - jb, 1e-6, None) * mask                   # 1 - measured JB rate
w = np.power(help_vec, 1.0 - risk) * np.power(safety, beta * risk)
w = np.maximum(w, self._c_min * mask)
```

| `risk` | resulting weights |
|---|---|
| `0` | `w ∝ Help(q)` — capability-first; serve the strongest model |
| `1` | `w ∝ (1 − JB_q)^β` — safety-first; reroute to the hardest configs |

`beta` (default `4.0`) sets how sharply it tilts at maximum risk. This is the only place
where **both** Phase-1 payoffs act as objectives simultaneously rather than one being a
constraint on the other.

**Why it reweights instead of routing.** A hard route would collapse `c(x)` to a point mass
and forfeit the entire argument: the attacker could then identify the deployed configuration
from a single request. The `c_min` floor keeps every feasible config in the support, so
randomisation survives the tilt. (With the default `c_min: 0.0` the floor is inert, and a
confident risk score can drive weights arbitrarily close to a point mass — worth setting
`c_min > 0` for any run where the reasoner's entropy is the object of interest.)

**Cost.** `PromptConditionalCoverage` caches `probs` **per prompt string**
(`coverage.py:88`), so the risk model fires once per distinct prompt. Since PAIR emits a
fresh rewrite at every step, that is one extra LLM call per attack step, not one per
request — plus a fourth model resident on the GPU.

## 6. `purple` — stub

Raises `NotImplementedError` with the implementation plan in its docstring. It is
registered, so `--selector purple` resolves and fails loudly with that plan rather than
silently doing something else.

The plan is Algorithm 1 of the proposal: grow a per-configuration RRT through Safe/Refused
nodes to collect jailbreaking prompts into a memory `M^q`, define a kernel risk profile
`ρ_q(x) = max_{p ∈ M^q} K(d(x,p)/H)`, then **run `minimax`'s LP with `ρ` substituted for
`J`**. The point of the substitution is generalisation: `J` is only defined on prompts that
were actually probed, whereas `ρ` extrapolates to unseen ones — a live request carries no
measured outcome. Being anticipatory rather than reactive is the whole difference from
`minimax`.

---

## Summary

| Selector | Uses `J` as | Uses `Help` as | Returns | Typical `H(c)` |
|---|---|---|---|---|
| `uniform` | — | support mask | `StaticCoverage` | `log` of the support size (max) |
| `deterministic` | row means → argmin | support mask | point mass | `0` |
| `validation` | row means → `(1−JB)^α` | support mask | `StaticCoverage` | mid |
| `minimax` | **full matrix**, one LP row per prompt | mixture constraint | `StaticCoverage` | data-dependent; `0` is the falsification |
| `reasoner` | row means → safety objective | capability objective | `PromptConditionalCoverage` | varies per `x` |
| `purple` | *(plan: kernel `ρ` in minimax's LP)* | *(plan: as minimax)* | — | — |

## Shared footguns

- **A configuration missing from `profile.json` looks like a free lunch.**
  `jailbreak_rate` defaults to `0.0` for an unmeasured qid and `help_array` defaults to
  `1.0` — perfectly safe *and* perfectly helpful. `deterministic` will pick it, and
  `validation`/`reasoner` will over-weight it. This bites whenever the menu is edited
  without re-running Phase 1, or when a probe was killed mid-menu. Cross-check the
  per-`qid` `JB=… Help=…` lines that `probe_menu.py` logs at the end against `|Q|`.
- **`tau` means different things to `minimax` than to everything else** (mixture constraint
  vs. support restriction) — they are not directly comparable at equal `tau`, and the
  frontier sweep is the honest comparison.
- **`c_min` is exact only under `minimax`**; elsewhere it is a post-normalisation clamp
  that renormalisation then dilutes, and under `uniform` it leaks mass to infeasible
  configs.
- **`history` is threaded but ignored** by every implemented selector, so today's
  multi-round runs re-commit the same `c` each round. Adaptation over rounds currently
  lives entirely on the attacker's side.
