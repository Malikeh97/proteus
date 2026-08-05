# Menu design: why the menu shape decides the result

The menu is not a list of things to try. Its structure determines, before any Phase 2
compute is spent, whether the central claim *can* come out true. This note records the
argument, the design rule that follows, and the audit that enforces it.

> **Two menus, two questions.** Most of this note derives the *iso-safety* menu
> (`configs/menus/full.yaml`) and concludes that a capability gradient is the worst possible
> shape to randomise over. That conclusion is correct **for the pure-ASR question at a `tau`
> that barely binds**. It does not carry over to the risk–helpfulness frontier, where a
> gradient is the *right* shape — see [The frontier question](#the-frontier-question-why-a-gradient-is-correct-there)
> at the end, which derives `configs/menus/frontier.yaml`. Read that section before
> concluding the two menus contradict each other.

## The arithmetic

Against a static attacker, ASR under coverage `c` on a fixed prompt set is approximately

```
ASR(c) ~ sum_q c_q * JB_q
```

so the two reference points are

| | score |
|---|---|
| `deterministic` | `min_q JB_q` |
| any mixture | a convex combination of the `JB_q`, hence **>= `min_q JB_q`** |

**A mixture can never win on the static average.** It is a convex combination of numbers
that is bounded below by their minimum. The claim the framework makes is therefore not
"mixing is safer" but

```
adaptation penalty for committing to one config  >  dilution penalty of mixing
```

and the **dilution penalty is exactly the spread of `JB_q` across the menu**. That single
observation is what makes menu design an experimental-validity issue rather than a matter
of taste:

- A **capability/safety gradient** maximises the spread. It is the most expensive possible
  menu to randomise over — you hand the deterministic baseline a large head start and then
  ask adaptation to make it up.
- An **iso-safety** menu drives the spread toward zero. Randomisation becomes nearly free,
  and any adaptation penalty at all is enough to win.

## Why a gradient also breaks the mechanism, not just the margin

`MinimaxSelector` mixes only when no member dominates another columnwise in `J`. A gradient
is a *total order*: members are ranked, so near-domination is the norm. The LP then returns
a vertex, `H(c) = 0`, and `coverage_entropy` reports that the selector chose a router rather
than a mixture — the falsification the metric exists to catch — for a config-design reason
rather than a scientific one.

What the thesis needs is the opposite structure: an **anti-chain** in the partial order of
blind-spot sets. Equal marginal safety, non-nested failure sets.

## The design rule

> **Members differ by substitution, never by addition.**

Every member carries exactly *one* screening mechanism. The reason is mechanical: if `w` is
a refusal-only wrapper, then adding it can shrink a configuration's blind set but never grow
it, so

```
blind(q + w)  subset-of  blind(q)     always
```

A menu containing both `q` and `q + w` therefore contains a **guaranteed domination pair**.
Those two members cannot differ strategically; they differ only in where they sit on the
safety/helpfulness trade. That is a gradient rung wearing a portfolio costume.

The original menu had three such pairs, e.g. `qwen3_4b+[qwen3guard]` against
`qwen3_4b+[keyword_filter, qwen3guard]` — `keyword_filter` is refusal-only, so the nesting
holds for every prompt set, no measurement required. This is why `keyword_filter` and
`safety_prompt` no longer appear as add-ons.

## The five axes

One axis varied at a time, so the Jaccard heatmap in the audit is interpretable.

| Axis | Members | Why the blind spots differ | Cost |
|---|---|---|---|
| **stage** | `qwen3guard_in` vs `qwen3guard_out` | An obfuscated request that reads benign going in is caught coming back, and vice versa. Mechanically disjoint at equal strength. | **free** |
| **provenance** | `qwen3guard_*` vs `shieldgemma_*` | Different training data and policy taxonomy. | free |
| **boundary** | `qwen3guard_in` vs `qwen3guard_in_strict` | Same mechanism, `Controversial` band blocked or not. Trades helpfulness for coverage on the ambiguous band. | free |
| **alignment** | `qwen3_4b` vs `qwen3_4b_saferl` at fixed screening | A real difference in the model's own refusal boundary, at the **same capability tier**. | one extra model |
| **class** | `smoothllm` (no classifier) | Randomized smoothing defeats brittle suffix/GCG-style attacks and does little against clean semantic jailbreaks — the opposite profile from a guard. Also perturbs served text per request, so the target is non-stationary *within* a member. | free |

**The stage axis is the cheapest large win.** `_GUARD_CACHE` in
`src/proteus/menu/wrappers/guards.py:34` is keyed on `hf_name`, while the wrapper instance
cache in `wrappers/__init__.py:15` is keyed on `wrapper_id`. So two wrapper YAMLs with
different ids pointing at the same `hf_name` are two distinct menu members sharing **one
copy of the weights** — zero extra VRAM. Every guard previously ran at `stage: both`, which
collapsed this axis entirely.

**Model diversity belongs within a capability tier.** Across tiers it *is* the gradient:
`qwen3-8b` guarded versus `qwen2.5-3b` guarded is the same strategy at two competence
levels, and one nearly dominates the other. Both are absent from the served menu.

## The audit

`scripts/audit_menu.py` reads `profile.json` and decides whether Phase 2 can be
informative. No GPU, no weights, seconds on a login node. Exit code 1 on any failure, so it
gates a launcher:

```bash
python scripts/audit_menu.py --menu full --seed 2 && bash run_experiments.sh
```

| Check | Failure it catches | Repair |
|---|---|---|
| 0 completeness | a member never probed — `jailbreak_rate` defaults to `0.0` and `help_array` to `1.0`, so it looks perfectly safe **and** perfectly helpful, and wins the deterministic baseline on missing data | re-probe |
| 1 iso-safety | `JB_q` spread — the dilution penalty itself | drop off-tier models |
| 2 anti-chain | `blind(q) subset-of blind(q')` — `q'` contributes nothing to safety | cut it, or substitute instead of add |
| 3 complementarity | pairwise Jaccard of blind sets; high everywhere = strategically redundant | swap in a different stage/provenance |
| 4 floor | prompts every member fails: an irreducible ASR floor (`ErrorKind.MENU_INADEQUACY`, decided in advance) | needs a genuinely new mechanism |
| 5 headroom | best deterministic vs. best mixture | see below |

### Check 5 needs two numbers, not one

`J` is binary, so `max_x J(q, x) = 1.0` for any configuration failing even one prompt.
Comparing that against the LP value would flatter the mixture for free. The two
granularities point in **opposite** directions and both belong in the report:

| | deterministic | mixture | who wins |
|---|---|---|---|
| **worst case** (what the LP optimises) | `min_q max_x J(q,x)` — `1.0` unless some `q` blocks everything | `min_c max_x c·J[:,x]` (LP objective) | the mixture: a deterministic defense has a prompt that always works, a mixture has none |
| **average case** (what Phase 2 measures, static attacker) | `min_q JB_q` | `sum_q c_q JB_q` | the deterministic config, always |

So the audit reports the mixture's worst-case *gain* and its average-case *handicap*
side by side. The handicap is the bar Phase 2 has to clear: adaptation must cost the
deterministic member more ASR than dilution costs the mixture. The audit bounds both
sides of that inequality before you pay for the experiment.

Verified on synthetic profiles: perfectly complementary disjoint blind sets over `|Q| = 8`
give LP value `1/8 = 0.125`, worst-case gain `0.875`, dilution `0.000`, and `H(c*) = log 8
= 2.079` nats. A gradient profile with a nested pair collapses to `H(c*) = 0` with a
`0.300` handicap.

## The contrast arm

The gradient menu is preserved verbatim as `configs/menus/gradient.yaml`. It is not dead
config — running the same selectors over `full` and `gradient` **measures the dilution
price directly**, which is the sharpest available version of RQ4 ("does portfolio diversity
matter?"). Probe it alongside:

```bash
python scripts/audit_menu.py --menu full     --seed 2
python scripts/audit_menu.py --menu gradient --seed 2
```

## Honest caveat

Iso-safety *and* strong complementarity is an empirical bet that may not pay. Off-the-shelf
guards are trained on overlapping data and their false negatives do correlate — the original
menu's own comment says so, as its reason for not *stacking* them (that argument does not
apply to *mixing* over them, which is how whatever decorrelation exists gets exploited).

If probing returns Jaccard overlaps around 0.8 everywhere, the honest result is: **mechanism
diversity at fixed strength is smaller than hoped, and here is the number.** That is a
publishable negative and precisely what the coverage-entropy metric was built to surface.
The **stage** and **boundary** axes are the most likely to survive, because they move the
decision boundary mechanically rather than hoping two independent training runs disagree.

Two members are most at risk of failing check 1: `qwen3_4b_saferl+smoothllm` (no classifier
at all) and `qwen3_4b+qwen3guard_in_strict` (the strict dial may cost more Help than it buys
in safety). If they fail, cut or re-pair them rather than proceeding.

---

## The frontier question: why a gradient is correct *there*

Everything above optimises one number, ASR, and treats `tau` as a side constraint that
barely binds. Change the objective to the **risk–helpfulness frontier** and the conclusion
inverts: the gradient stops being the worst menu and becomes the only one that works.

### The hull argument

Against a static attacker both payoffs are linear in `c`:

```
ASR(c)  = sum_q c_q * JB_q          Help(c) = sum_q c_q * Help_q
```

So the achievable set is the **convex hull** of the `|Q|` points `(Help_q, JB_q)`. A point
mass is stuck at a vertex of that hull; a mixture reaches anywhere inside it. Under the
floor `Help >= tau`:

| | is restricted to |
|---|---|
| `deterministic` | `min JB_q` over `{q : Help_q >= tau}` — a **vertex** |
| a mixture | `min sum_q c_q JB_q` over `{c : c · help >= tau}` — the **hull** |

Worked, with `A = (Help 0.9, JB 0.8)` and `B = (Help 0.4, JB 0.1)` at `tau = 0.7`:
`deterministic` can only take `A`, scoring **0.80**. The mixture `c = (0.6, 0.4)` sits at
`Help = 0.70` exactly and scores **0.52**. The mixture strictly wins.

**This needs no adaptation penalty.** It holds against a *static* attacker, purely from
frontier interpolation — which is exactly what the iso-safety menu cannot deliver, since
there the whole argument rests on the attacker paying for non-stationarity. Whenever `tau`
falls strictly between two adjacent `Help_q` on the hull, the deterministic baseline must
overshoot to the next vertex up and pays the entire gap.

The two designs are therefore answering different questions, and neither supersedes the
other:

| | `full` (iso-safety) | `frontier` (capability-tiered) |
|---|---|---|
| objective | ASR at fixed, barely-binding `tau` | the (Help, ASR) frontier, `tau` binding |
| menu shape | anti-chain: equal safety, non-nested blind spots | monotone chain in **both** payoffs |
| why mixing wins | dilution ~ 0, so any adaptation penalty suffices | interpolation reaches non-vertex points |
| needs an adaptive attacker | **yes** | no |
| spread of `JB_q` | must be small (it is the price) | large by construction, and paid for in Help |

### Help must be quality-weighted, or the gradient is invisible

`Help(q)` was originally `1 - refusal_rate` on benign traffic — a pure **over-refusal**
measure. Under it, a 1.5B model that answers benign questions willingly but badly scores
`Help = 1.0`, identical to a 14B model that answers them correctly. The capability axis
simply does not exist, and a capability-tiered menu collapses into "how aggressively does
each guard over-refuse".

`scripts/probe_menu.py` now computes

```
Help(q) = mean over benign x of:   0.0            if refused
                                   quality / 5    otherwise
```

where `quality` is the 1–5 answer-quality grade from the Flow-Judge helpfulness rubric
(`src/proteus/judges/flow_judge.py`). Refusals stay in the denominator, so over-refusal
still costs — a configuration cannot raise its score by refusing more. Judges that do not
score quality fall back to the old definition, which keeps the `keyword` judge, the `dev`
menu, and every pre-existing `profile.json` working. **That fallback is silent**, so a
`frontier` probe run with the wrong judge produces a flat, meaningless Help column rather
than an error.

### Reading the audit on a gradient menu

`scripts/audit_menu.py` was written to enforce the iso-safety design, so on `frontier`
**checks 1 (spread) and 2 (anti-chain) are expected to fail** — that is the menu working as
intended, not a defect. Relax the threshold and read check 5:

```bash
python scripts/audit_menu.py --menu frontier --seed 42 --tau <chosen> --max-spread 1.0
```

Check 5 already computes the right quantity: `dilution = mixture_average -
deterministic_average`, both under the `tau` constraint. On `full` it is **positive** — the
handicap Phase 2 must overcome. On `frontier` it should be **negative**, and that negative
number *is* the result, available before spending an allocation on Phase 2.

### One caveat that will shape the reading

Only `minimax` can exploit this menu. Per `docs/selectors.md` it is the sole selector
treating `tau` as a constraint on the *mixture* (`c · help >= tau`); `uniform`, `validation`
and `reasoner` all go through `Selector._feasible_mask()`, which restricts the **support** to
individually-feasible members. At a `tau` above the small model's `Help`, that mask deletes
the safe members outright, so those three collapse onto the capable end and look like
`deterministic`. This is a genuine property for `deterministic` — a point mass cannot
interpolate — but an **implementation artifact** for `uniform`/`validation`. Do not report
it as evidence that only clever randomisation helps.
