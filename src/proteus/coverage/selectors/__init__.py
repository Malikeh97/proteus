from __future__ import annotations

from pathlib import Path

from proteus.coverage.profile import MenuProfile
from proteus.coverage.selectors.base import Selector
from proteus.menu.menu import ResourceMenu
from proteus.utils.config import CONFIGS_DIR, SelectorConfig, load_selector_config
from proteus.utils.registry import Registry

SELECTORS: Registry[Selector] = Registry("selector")

# Registration side effects; must follow SELECTORS.
from proteus.coverage.selectors import minimax, purple, reasoner, simple  # noqa: E402,F401


def load_selector(
    config: SelectorConfig,
    menu: ResourceMenu,
    profile: MenuProfile | None = None,
    tau: float = 0.0,
    c_min: float = 0.0,
) -> Selector:
    return SELECTORS.create(config.type, config, menu, profile, tau, c_min)


def load_selector_by_name(
    name: str,
    menu: ResourceMenu,
    profile: MenuProfile | None = None,
    tau: float = 0.0,
    c_min: float = 0.0,
    configs_dir: str | Path = CONFIGS_DIR,
) -> Selector:
    return load_selector(load_selector_config(name, configs_dir), menu, profile, tau, c_min)


__all__ = ["SELECTORS", "Selector", "load_selector", "load_selector_by_name"]
