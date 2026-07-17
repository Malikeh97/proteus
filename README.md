<div align="center">

<img src="figures/logo.png" alt="Proteus logo" width="120">

# Proteus: Shape-shifting defense via Stackelberg game-theoretic mechanism selection against adaptive LLM attacks



</div>

A deployment is not one model behind one filter. It is a *menu* of models and defense mechanisms, and the safety question is which combination to serve, against an attacker who adapts to whatever you chose.

Proteus formulates that as a Stackelberg security game. The defender commits a **coverage** `c ∈ Δ(Q)` over a resource menu `Q = M × 2^W`; a configuration is drawn per request; the attacker sees only responses and re-optimizes against what they reveal. No payoff matrix is supplied. Attack success, helpfulness, and the residual incentive to deviate are all measured by execution.

> Fun fact: In Homer's *Odyssey* (Book IV), Proteus can foresee the future and answer any question. To guard his knowledge he changes form  (e.g., lion, serpent, leopard, wave, fire), and reveals the truth only after relentless pursuit.

---

## Contents

- [Quickstart](#quickstart) · [Setup](#setup) · [Running experiments](#running-experiments)
- [How it fits together](#how-it-fits-together) · [Extending](#extending-the-framework)
- [Components](#components) · [Output layout](#output-layout) · [Status](#implementation-status)

---

## Quickstart

On a `klogin*` login node, from the project root:

```bash
mkdir -p logs && sbatch setup/create_env_killarney_uv.sh
source setup/start_env.sh

python scripts/inspect_menu.py --menu dev --verbose        # no weights loaded
bash run_probe.sh                                          # dev smoke test
bash run_experiments.sh                                    # dev smoke test
bash run_analysis.sh
```

`run_probe.sh` and `run_experiments.sh` ship with only the `dev` blocks uncommented. Confirm those go green, then uncomment the real experiment blocks.

---

## Setup

### 1. Environment (once)

The environment is built **as a GPU job**, so `torch` and `bitsandbytes` resolve against the node's CUDA rather than the login node's:

```bash
mkdir -p logs
sbatch setup/create_env_killarney_uv.sh
# Logs: logs/<jobid>_create_env_killarney.out
```

This runs `uv sync` against the committed `uv.lock`. Everything afterwards activates the venv automatically via `source setup/start_env.sh`.

For the GRPO attacker (a stub today), add the extra: `uv sync --extra rl`.

### 2. HuggingFace token

Gated models — `Llama-3.1-8B-Instruct` (the judge) and `ShieldGemma` — need a token. Create `.env` in the project root:

```bash
echo 'HF_TOKEN=hf_your_token_here' > .env
```

`.env` is gitignored. `start_env.sh` sources it automatically. Request access to the gated repos on HuggingFace first, or the judge will fail to load at Phase 1.

### 3. What `start_env.sh` does

Sourced by every launcher and every `sbatch` wrapper. It loads `cuda/12.6` and `gcc arrow/19.0.1 python/3.11`, activates `.venv`, points the HF/torch caches at `$SCRATCH`, sources `.env`, and defines two shell functions:

| Function | Purpose |
|---|---|
| `submit <job_name> <command>` | `sbatch` wrapper. Resource flags live in `setup/submit_killarney.sbatch`; the job name and log path are passed at submit time. |
| `should_skip_job <job_name>` | True if the job is queued, running, or completed in the last 2 days. |

`should_skip_job` is what makes every launcher **idempotent**: re-running `bash run_experiments.sh` submits only what is missing. Combined with `--resume` (records are appended one per trial), a preempted 23-hour job resubmits and picks up mid-run.

Results go to `$PROTEUS_OUTPUT_DIR`, which defaults to `$SCRATCH/proteus`.

### Cluster settings

Killarney only, as configured. Adding a cluster means one hostname branch in `start_env.sh` plus a `setup/submit_<cluster>.sbatch`.

| | Killarney (Vector) |
|---|---|
| Login host | `klogin*` |
| Partition | `gpubase_l40s_b3` |
| Account | `aip-craffel` |
| GPU | `--gres=gpu:l40s:1` (48 GB) |
| Resources | `--cpus-per-task=8`, `--mem=128GB`, `--time=23:00:00` |
| `$SCRATCH` | `/home/$USER/scratch/$USER` |

**Why one GPU is enough.** Every model in the menu loads at 4-bit `nf4`, and models are cached process-wide — a model appearing in `2^|W|` configurations is loaded once. The full menu (`|Q| = 28`) plus judge and attacker is ~25 GB of a 48 GB L40S:

| Component | | VRAM |
|---|---|---|
| Qwen3-8B / Qwen3-4B / Qwen2.5-3B / Qwen3-4B-SafeRL | served menu | ~12.0 GB |
| Qwen3Guard-4B / ShieldGemma-2b / PIGuard | wrappers | ~ 4.7 GB |
| Llama-3.1-8B-Instruct | judge | ~ 5.0 GB |
| Olmo-3-7B-Instruct | attacker | ~ 4.0 GB |
| | **total** | **~25.7 GB** |

---

## Running experiments

Four phases. Phases 1–2 need a GPU and are submitted with `submit`; phases 3–4 are pure post-processing and run on the login node.

| Phase | Script | Launcher | GPU | Produces |
|---|---|---|---|---|
| 0 | `inspect_menu.py` | — | no | The menu `Q`, printed. No weights loaded. |
| 1 | `probe_menu.py` | `run_probe.sh` | yes | `J(q,x)` and `Help(q)` per configuration → `profile.json` |
| 2 | `run_game.py` | `run_experiments.sh` | yes | Committed coverages + adaptive attack records |
| 3 | `evaluate.py` | `run_analysis.sh` | no | `metrics.csv` |
| 4 | `plot_results.py` | `run_analysis.sh` | no | `frontier / gap / entropy / cost.pdf` |

Phase 1 is mandatory: every selector but `uniform` optimizes over the payoffs it measures. It also gives you **every single-configuration baseline for free** — undefended models, each wrapper alone, the naive stack of all wrappers, and the always-de-escalate policy are all just menu members.

### The launcher convention

`run_probe.sh` and `run_experiments.sh` are walls of `submit` lines grouped by research question, nearly all commented out. **Uncomment what you want, then run.** The config stays fixed and all variation is CLI overrides, so a run's identity is legible from its job name:

```bash
submit "ex_rq1_mm_pair_tau0.7_s1997" "$RQ1 --selector minimax --tau 0.7 --seeds 1997"
#       └── job name ──────────────┘         └── overrides ─────────────────────┘
```

Adding a condition is copying a line and changing one override.

### Typical workflow

```bash
source setup/start_env.sh

# Phase 1 — uncomment the FULL MENU block in run_probe.sh first
bash run_probe.sh
squeue -u $USER                      # wait for these to finish

# Phase 2 — uncomment the RQ1 block in run_experiments.sh
bash run_experiments.sh
squeue -u $USER

# Phases 3-4 — login node, no GPU
bash run_analysis.sh
```

### What each research question needs

| RQ | Question | Vary | Blocks |
|---|---|---|---|
| 1 | Does randomizing beat the best deterministic config? | `--selector` × `--tau` | `RQ1` |
| 2 | Does randomizing narrow the static↔adaptive gap? | `--attacker static` vs `pair` | `RQ2` |
| 3 | How do coverage-selection methods compare? | `--selector` | `RQ1` (same runs) |
| 4 | Does portfolio diversity matter? | `--menu` | `RQ4` |
| 5 | Does it raise the attacker's compute budget? | — (from RQ1 runs) | `RQ1` |
| 6 | How close to equilibrium? Robust to unseen attacks? | `--held-out-attacker` | `RQ6` |

RQ1 and RQ3 are the same runs read two ways. RQ5 is a metric over RQ1's records, not a separate experiment.

### Running one thing by hand

```bash
python scripts/run_game.py \
    --experiment configs/experiments/paper/rq1_randomized_vs_deterministic.yaml \
    --selector minimax --tau 0.8 --seeds 1997 \
    --output-dir $PROTEUS_OUTPUT_DIR --resume
```

Every script's `--help` carries copy-pasteable examples.

---

## How it fits together

```
                 configs/menus/full.yaml
                          │
                    ResourceMenu ─── Q = M × 2^W ── Configuration q = (M_i, S)
                          │                              │  serve(x):
   profile.json ──────────┤                              │    input wrappers → model → output wrappers
   (Phase 1: J, Help)     │                              │
                          ▼
                      Selector ───────► Coverage  c: X × H → Δ(Q)
              uniform / validation /        │
              minimax / [reasoner] /        │
              [purple] / deterministic      ▼
                                       Deployment ── q ~ c(x) ── serve ── Judge ── Outcome
                                            │                                       {Jailbreak, Safe, Refused}
                                            ▼
                    ┌──────────────── run_round ────────────────┐
                    │  for x in prompts:                        │
                    │      q ~ c_t(x)      ← redrawn per request│
                    │      y = q.serve(x)                       │
                    │      z = judge(x, y)                      │
                    │      x = attacker.refine(x, y, z)         │
                    └───────────────────┬───────────────────────┘
                                        ▼
                          results.jsonl → metrics → plots
```

**The one invariant worth protecting.** `Attacker.refine(prompt, response, judgment, step)` receives the response and nothing else — never the coverage `c_t`, never the drawn configuration `q_t`, never which wrapper fired. That signature *is* the projection `ρ` from the formal setup: an attacker at an API issues prompts and reads outputs. Widening it silently converts the game into the leaked-commitment relaxation and turns every reported ASR from a measurement into an upper bound. See `src/proteus/attackers/base.py`.

A configuration is drawn **per request**, not per round: the attacker's second probe of the same prompt may land on a different member of the support. That is what makes the deployed defense hard to identify.

---

## Extending the framework

Every plug-in point is a registry. Adding a component is one file plus a decorator plus a YAML — nothing else in the tree changes.

<details open>
<summary><b>Add a model</b> — YAML only</summary>

```yaml
# configs/models/my_model.yaml
model_id: "my-model"          # hyphenated; appears in qids and output paths
backend: "huggingface"
hf_name: "org/MyModel-7B"
params_b: 7.0                 # for FLOP accounting
quantization: "4bit"
generation: {max_new_tokens: 512, temperature: 0.7, top_p: 0.9, do_sample: true}
```

Then add the stem `my_model` to a menu's `models:` list. Config stems use underscores; `model_id` uses hyphens.
</details>

<details>
<summary><b>Add a defense mechanism</b> — one class</summary>

```python
# src/proteus/menu/wrappers/mine.py
from proteus.menu.wrappers import WRAPPERS
from proteus.menu.wrappers.base import Verdict, Wrapper

@WRAPPERS.register("my_filter")
class MyFilter(Wrapper):
    def on_input(self, prompt: str) -> Verdict:
        if bad(prompt):
            return Verdict.refuse("my_filter:input")
        return Verdict.passthrough(prompt)

    # on_output is optional — both hooks default to pass-through.
```

Import it in `wrappers/__init__.py`, then write `configs/wrappers/my_filter.yaml` with `type: "my_filter"`. Wrappers may allow, **transform** (return different text), or refuse — which covers classifiers, prompt-level interventions, and smoothing alike.
</details>

<details>
<summary><b>Add a coverage-selection strategy</b></summary>

```python
# src/proteus/coverage/selectors/mine.py
from proteus.coverage.coverage import StaticCoverage
from proteus.coverage.selectors import SELECTORS
from proteus.coverage.selectors.base import Selector

@SELECTORS.register("my_selector")
class MySelector(Selector):
    def select(self, history=None, round_idx=0):
        profile = self._require_profile()        # measured J and Help
        w = my_solve(profile, self._tau, self._c_min)
        return StaticCoverage(self._menu, w, self.selector_id)
```

You get `self._menu`, `self._profile`, `self._tau`, `self._c_min`, and the defender's `history` (which retains `c_s`, `q_s`, `y_s`). Return `PromptConditionalCoverage` instead if you condition on `x`.
</details>

<details>
<summary><b>Add an attacker</b></summary>

```python
# src/proteus/attackers/mine.py
from proteus.attackers import ATTACKERS
from proteus.attackers.base import Attacker

@ATTACKERS.register("my_attack")
class MyAttacker(Attacker):
    def reset(self, coverage_id=None):        # called once per round
        super().reset(coverage_id)

    def initialize(self, base_prompt: str) -> str:
        return base_prompt

    def refine(self, prompt, response, judgment, step) -> str:
        return mutate(prompt, response)       # response only — see the invariant above
```
</details>

<details>
<summary><b>Add a benchmark or judge</b></summary>

Same shape: `@BENCHMARKS.register("name")` implementing `load(n, seed) -> list[Prompt]`, or `@JUDGES.register("name")` implementing `judge(prompt, response) -> int`.
</details>

To see everything currently registered:

```bash
python scripts/inspect_menu.py --list-registries
```

---

## Components

### Menus (`configs/menus/`)

| Menu | Size | Purpose |
|---|---|---|
| `dev` | 4 | Smoke test. One model, two free wrappers, no guards to load. |
| `full` | 28 | The paper menu. 4 models × wrapper subsets of size ≤ 2. |
| `small` | 8 | The fallback menu. 2 models, 2 wrappers. |
| `models_only` | 4 | Ablation: model axis alone, no mechanisms. |
| `single_model` | 8 | Ablation: mechanism axis alone, one fixed model. |

`max_wrappers` caps the wrapper-subset size to keep the menu tractable — the stack-everything corner is a baseline rather than a menu member. Drop the cap to include it.

### Selectors (`configs/selectors/`)

| Selector | Conditions on `x`? | Needs profile? | Notes |
|---|---|---|---|
| `deterministic` | no | yes | Best single config — a point mass. **The RQ1 baseline.** |
| `uniform` | no | for `tau` | Uniform over the feasible menu. |
| `validation` | no | yes | `w_q ∝ (1 − JB_q)^α` on the fit split. |
| `minimax` | no | yes | The classical SSG solve as an LP, over measured payoffs. |
| `reasoner` | **yes** | yes | *Stub.* Risk-scores each prompt and tilts the coverage. |
| `purple` | no | yes | *Stub.* Anticipatory per-config RRT. |

### Attackers (`configs/attackers/`)

| Attacker | Adaptive? | Role |
|---|---|---|
| `static` | no | Base prompt only. The static half of RQ2. |
| `template` | no | Cycles a template bank. Adapts its index, not to responses. |
| `pair` | **yes** | PAIR-style rewriting. **The v1 in-loop attacker.** |
| `grpo` | yes | *Stub.* RL prompt policy, retrained per coverage. |
| `search` | yes | *Stub.* Held-out beam search with LLM mutation. |

### Metrics

`asr` · `asr_at_budget` · `helpfulness` · `coverage_entropy` · `support_size` · `flops_to_first_jailbreak` · `flops_to_target_asr` · `equilibrium_gap` · `pareto_front` · `classify_regime`

Coverage entropy is the measurement most capable of falsifying the framework: `H(c) = 0` means the selector chose a router, not a mixture, and the central claim does not apply.

---

## Output layout

```
$PROTEUS_OUTPUT_DIR/
├── {menu}/
│   ├── profile/{seed}/
│   │   ├── probe.jsonl                                  Phase 1 trials
│   │   └── profile.json                                 J(q,x), Help(q)
│   └── {selector}/{attacker}/tau{tau}/{seed}/
│       ├── coverages.json                               committed c_t per round
│       ├── round{t}/results.jsonl                       Phase 2 trials
│       └── held_out/results.jsonl                       held-out evaluation
└── analysis/
    ├── metrics.csv
    ├── summary.txt
    └── plots/{frontier,gap,entropy,cost}.pdf
```

Every step of every trial is stored, so `asr_at_budget` and the `tau` sweep are derivable post hoc — the budget sweep needs no re-running.

---

## Implementation status

Working: the menu, coverage, and game layers; `uniform` / `validation` / `minimax` / `deterministic` selectors; `static` / `template` / `pair` attackers; the LLM judge; all metrics; all four phases.

Registered stubs, each raising `NotImplementedError` with an implementation plan in its docstring:

| Stub | File |
|---|---|
| Purple Agent (Algorithm 1) | `src/proteus/coverage/selectors/purple.py` |
| Reasoner reweighting | `src/proteus/coverage/selectors/reasoner.py` |
| GRPO attacker | `src/proteus/attackers/grpo.py` |
| Held-out search attacker | `src/proteus/attackers/search.py` |

They are registered, so `--selector purple` resolves and fails loudly with the plan rather than silently doing something else.

**Model availability is unverified.** `Qwen/Qwen3-4B-SafeRL`, `Qwen/Qwen3Guard-Gen-4B`, `SaFoLab-WISC/PIGuard`, and `allenai/Olmo-3-7B-Instruct` are named from the proposal; confirm the exact HF repo ids resolve before a long run. Only `hf_name` in `configs/models/` and `configs/wrappers/` needs changing.

Guard output parsing (`_is_unsafe` in `src/proteus/menu/wrappers/guards.py`) is string-matching against each guard's documented verdict format. Spot-check it against real outputs before trusting a probe — a guard that never fires and a guard whose verdict you are misparsing look identical in the profile.

---

## Tests

```bash
pytest -q          # 12 smoke tests, no GPU, no network
```

They check the wiring, not the science: that `q` short-circuits the model when an input wrapper refuses, that payoffs are zero-sum, that a point mass has zero entropy, that `minimax` mixes when blind spots are complementary and collapses to a vertex when one configuration dominates.
```
