"""Model registry + a process-wide cache.

The cache matters: one game run touches every configuration in the menu, and a
model appears in 2^|W| of them. Loading it once is the difference between the
menu fitting on one L40S and not.
"""

from __future__ import annotations

from proteus.menu.models.base import BaseModel, Generation
from proteus.menu.models.hf_model import HFModel
from proteus.utils.config import ModelConfig
from proteus.utils.registry import Registry

MODELS: Registry[BaseModel] = Registry("model backend")
MODELS.register("huggingface")(HFModel)

_CACHE: dict[str, BaseModel] = {}


def load_model(config: ModelConfig) -> BaseModel:
    if config.model_id not in _CACHE:
        _CACHE[config.model_id] = MODELS.create(config.backend, config)
    return _CACHE[config.model_id]


def clear_cache() -> None:
    """Drop cached models and free their VRAM."""
    _CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


__all__ = ["MODELS", "BaseModel", "Generation", "HFModel", "clear_cache", "load_model"]
