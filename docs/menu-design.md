# Menu design: why the menu shape decides the result

The menu is not a list of things to try. Its structure determines, before any Phase 2
compute is spent, whether the central claim *can* come out true. This note records the
argument, the design rule that follows, and the audit that enforces it.

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
