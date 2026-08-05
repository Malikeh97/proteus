from __future__ import annotations

from pathlib import Path

from proteus.attackers.base import Attacker
from proteus.utils.config import CONFIGS_DIR, AttackerConfig, load_attacker_config
from proteus.utils.registry import Registry

ATTACKERS: Registry[Attacker] = Registry("attacker")

# Registration side effects; must follow ATTACKERS.
from proteus.attackers import gcg, grpo, pair, search, static  # noqa: E402,F401


def load_attacker(config: AttackerConfig, configs_dir: str | Path = CONFIGS_DIR) -> Attacker:
    cls = ATTACKERS.get(config.type)
    try:
        return cls(config, configs_dir=configs_dir)
    except TypeError:
        return cls(config)


def load_attacker_by_name(name: str, configs_dir: str | Path = CONFIGS_DIR) -> Attacker:
    return load_attacker(load_attacker_config(name, configs_dir), configs_dir)


__all__ = ["ATTACKERS", "Attacker", "load_attacker", "load_attacker_by_name"]
