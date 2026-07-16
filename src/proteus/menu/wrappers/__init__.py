from __future__ import annotations

from proteus.menu.wrappers.base import Verdict, Wrapper
from proteus.utils.config import WrapperConfig
from proteus.utils.registry import Registry

WRAPPERS: Registry[Wrapper] = Registry("wrapper")

# Imported for their registration side effects. Keep at the bottom: the modules
# import WRAPPERS from here.
from proteus.menu.wrappers import guards, prompting  # noqa: E402,F401

_CACHE: dict[str, Wrapper] = {}


def load_wrapper(config: WrapperConfig) -> Wrapper:
    """Wrappers are shared across every configuration that includes them."""
    if config.wrapper_id not in _CACHE:
        _CACHE[config.wrapper_id] = WRAPPERS.create(config.type, config)
    return _CACHE[config.wrapper_id]


def clear_cache() -> None:
    _CACHE.clear()


__all__ = ["WRAPPERS", "Verdict", "Wrapper", "clear_cache", "load_wrapper"]
