"""Pydantic schemas + YAML loaders.

Composition is by *stem reference*: an experiment names `menu: full`, which
resolves to `configs/menus/full.yaml`; that menu names `models: [qwen3_8b]`,
which resolves to `configs/models/qwen3_8b.yaml`. No interpolation, no Hydra.

Every schema carries an `extra`/`params` dict so a new component can take
options without editing this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONFIGS_DIR = Path("configs")


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class ModelConfig(BaseModel):
    model_id: str  # hyphenated; appears in qids and output paths
    backend: Literal["huggingface"] = "huggingface"
    hf_name: str
    hf_tokenizer_id: str | None = None  # override when tokenizer != hf_name
    params_b: float | None = None  # billions, for FLOP accounting
    model_type: Literal["base", "instruct"] = "instruct"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    # Compute dtype. float16 is the historical default for the served menu; set
    # bfloat16 for models whose card specifies it (Phi-3 overflows in fp16).
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "cuda"
    # Off by default: every model on the menu has a native transformers
    # implementation, and a repo's vendored modeling_*.py is pinned to whatever
    # transformers was current when it was uploaded. Flow-Judge's auto_map points
    # at microsoft/Phi-3.5-mini-instruct's 2024 code, which calls the long-removed
    # DynamicCache.seen_tokens and dies at the first generate(). Only enable for a
    # genuinely novel architecture transformers does not yet ship.
    trust_remote_code: bool = False
    enable_thinking: bool = False  # Qwen3 chat-template switch
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    extra: dict[str, Any] = Field(default_factory=dict)


class WrapperConfig(BaseModel):
    """A defense mechanism w in the library W."""

    wrapper_id: str
    type: str  # key in the WRAPPERS registry
    stage: Literal["input", "output", "both"] = "both"
    hf_name: str | None = None
    hf_tokenizer_id: str | None = None
    params_b: float | None = None
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    device: str = "cuda"
    trust_remote_code: bool = False  # see ModelConfig.trust_remote_code
    threshold: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)


class MenuEntry(BaseModel):
    """One curated configuration: a model stem + an explicit wrapper-set."""

    model: str  # stem ref into configs/models/
    wrappers: list[str] = Field(default_factory=list)  # stem refs into configs/wrappers/


class MenuConfig(BaseModel):
    """The resource menu. Either a cross-product (M x 2^W) or a curated list."""

    menu_id: str

    # Cross-product mode: every model crossed with subsets of the shared wrapper set.
    models: list[str] = Field(default_factory=list)
    wrappers: list[str] = Field(default_factory=list)
    # all: every subset of W. singletons: {} and each {w}. none: bare models only.
    subsets: Literal["all", "singletons", "none"] = "all"
    max_wrappers: int | None = None  # cap |S| to keep |Q| tractable
    exclude: list[str] = Field(default_factory=list)  # qids to drop

    # Curated mode: explicit per-model wrapper sets. Mutually exclusive with `models`.
    configs: list[MenuEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_mode(self) -> "MenuConfig":
        if self.configs and self.models:
            raise ValueError(
                f"Menu '{self.menu_id}': set either 'configs' (curated) or "
                f"'models' (cross-product), not both."
            )
        if not self.configs and not self.models:
            raise ValueError(f"Menu '{self.menu_id}': needs 'models' or 'configs'.")
        return self


class AttackerConfig(BaseModel):
    attacker_id: str
    type: str  # key in the ATTACKERS registry
    attacker_model: str | None = None  # stem ref into configs/models/
    max_steps: int = 5
    params: dict[str, Any] = Field(default_factory=dict)


class SelectorConfig(BaseModel):
    selector_id: str
    type: str  # key in the SELECTORS registry
    params: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    judge_id: str
    type: str = "llm"  # key in the JUDGES registry
    model: str | None = None  # stem ref into configs/models/
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str

    # Components, all stem references.
    menu: str
    selector: str
    attacker: str
    judge: str
    held_out_attacker: str | None = None  # evaluates c*, never selects it

    # Data
    attack_benchmark: str = "harmbench"
    benign_benchmark: str = "xstest"
    n_attack_prompts: int = 50
    n_benign_prompts: int = 50
    fit_frac: float = 0.5  # prompts that fit a coverage are disjoint from those scoring it

    # Game
    rounds: int = 3
    attack_budget: int = 5  # refinement steps per prompt per round
    tau: float = 0.7  # helpfulness floor
    c_min: float = 0.0  # minimum mass per configuration

    seeds: list[int] = Field(default_factory=lambda: [1997])
    extra: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str | Path, config_class: type) -> Any:
    with open(path) as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def _load_named(kind: str, name: str, cls: type, configs_dir: str | Path) -> Any:
    path = Path(configs_dir) / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (Path(configs_dir) / kind).glob("*.yaml"))
        raise FileNotFoundError(
            f"No {kind} config '{name}' at {path}. Available: {', '.join(available) or '(none)'}"
        )
    return load_config(path, cls)


def load_model_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> ModelConfig:
    return _load_named("models", name, ModelConfig, configs_dir)


def load_wrapper_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> WrapperConfig:
    return _load_named("wrappers", name, WrapperConfig, configs_dir)


def load_menu_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> MenuConfig:
    return _load_named("menus", name, MenuConfig, configs_dir)


def load_attacker_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> AttackerConfig:
    return _load_named("attackers", name, AttackerConfig, configs_dir)


def load_selector_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> SelectorConfig:
    return _load_named("selectors", name, SelectorConfig, configs_dir)


def load_judge_config(name: str, configs_dir: str | Path = CONFIGS_DIR) -> JudgeConfig:
    return _load_named("judges", name, JudgeConfig, configs_dir)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)
