from __future__ import annotations

from pathlib import Path

from proteus.judges.base import Judge, RefusalDetector
from proteus.utils.config import CONFIGS_DIR, JudgeConfig, load_judge_config
from proteus.utils.registry import Registry

JUDGES: Registry[Judge] = Registry("judge")

# Registration side effects; must follow JUDGES.
from proteus.judges.flow_judge import FlowJudge  # noqa: E402
from proteus.judges.llm_judge import (  # noqa: E402
    KeywordJudge,
    KeywordRefusalDetector,
    LLMJudge,
)


def load_judge(config: JudgeConfig, configs_dir: str | Path = CONFIGS_DIR) -> Judge:
    cls = JUDGES.get(config.type)
    try:
        return cls(config, configs_dir=configs_dir)
    except TypeError:
        return cls(config)


def load_judge_by_name(name: str, configs_dir: str | Path = CONFIGS_DIR) -> Judge:
    return load_judge(load_judge_config(name, configs_dir), configs_dir)


__all__ = [
    "JUDGES",
    "FlowJudge",
    "Judge",
    "KeywordJudge",
    "KeywordRefusalDetector",
    "LLMJudge",
    "RefusalDetector",
    "load_judge",
    "load_judge_by_name",
]
