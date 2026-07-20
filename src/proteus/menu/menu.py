"""The resource menu Q = M x 2^W.

Configurations are enumerated eagerly (they are just descriptors) but the models
and wrappers behind them load lazily and are shared, so |Q| = 28 costs the VRAM
of |M| + |W| models, not 28 of them.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from proteus.menu.configuration import Configuration, make_qid
from proteus.menu.models import load_model
from proteus.menu.wrappers import load_wrapper
from proteus.utils.config import (
    CONFIGS_DIR,
    MenuConfig,
    load_menu_config,
    load_model_config,
    load_wrapper_config,
)
from proteus.utils.logging import get_logger

logger = get_logger("menu")


def _wrapper_subsets(names: list[str], mode: str, max_wrappers: int | None) -> list[tuple[str, ...]]:
    if mode == "none":
        return [()]
    if mode == "singletons":
        return [()] + [(w,) for w in names]
    cap = len(names) if max_wrappers is None else min(max_wrappers, len(names))
    return [s for k in range(cap + 1) for s in combinations(names, k)]


class ResourceMenu:
    def __init__(self, config: MenuConfig, configs_dir: str | Path = CONFIGS_DIR) -> None:
        self._config = config
        self._configs_dir = Path(configs_dir)
        self._model_stems: dict[str, str] = {}  # model_id -> stem
        self._instances: dict[str, Configuration] = {}

        self._model_configs = {}
        self._wrapper_configs = {}
        self._spec: dict[str, tuple[str, tuple[str, ...]]] = {}

        if config.configs:
            # Curated: an explicit list of (model, wrapper-set) configurations. Each
            # model may carry a different wrapper set; only referenced models/wrappers
            # are loaded. Wrapper ids are stored in YAML (serve-pipeline) order, while
            # the qid uses make_qid's sorted order -- as Configuration.qid does.
            for entry in config.configs:
                mc = load_model_config(entry.model, self._configs_dir)
                self._model_configs[mc.model_id] = mc
                self._model_stems[mc.model_id] = entry.model

                wrapper_ids: list[str] = []
                for w_stem in entry.wrappers:
                    wc = load_wrapper_config(w_stem, self._configs_dir)
                    self._wrapper_configs[wc.wrapper_id] = wc
                    wrapper_ids.append(wc.wrapper_id)

                qid = make_qid(mc.model_id, wrapper_ids)
                if qid in config.exclude:
                    continue
                if qid in self._spec:
                    logger.warning(
                        f"Menu '{config.menu_id}': duplicate config {qid}, keeping first"
                    )
                    continue
                self._spec[qid] = (mc.model_id, tuple(wrapper_ids))
        else:
            # Cross-product: every model crossed with subsets of the shared wrapper set.
            for stem in config.models:
                mc = load_model_config(stem, self._configs_dir)
                self._model_configs[mc.model_id] = mc
                self._model_stems[mc.model_id] = stem

            for stem in config.wrappers:
                wc = load_wrapper_config(stem, self._configs_dir)
                self._wrapper_configs[wc.wrapper_id] = wc

            subsets = _wrapper_subsets(
                list(self._wrapper_configs), config.subsets, config.max_wrappers
            )
            for model_id in self._model_configs:
                for subset in subsets:
                    qid = make_qid(model_id, list(subset))
                    if qid in config.exclude:
                        continue
                    self._spec[qid] = (model_id, subset)

        logger.info(
            f"Menu '{config.menu_id}': |M|={len(self._model_configs)} "
            f"|W|={len(self._wrapper_configs)} |Q|={len(self._spec)}"
        )

    @classmethod
    def from_name(cls, name: str, configs_dir: str | Path = CONFIGS_DIR) -> ResourceMenu:
        return cls(load_menu_config(name, configs_dir), configs_dir)

    @property
    def menu_id(self) -> str:
        return self._config.menu_id

    @property
    def qids(self) -> list[str]:
        """Canonical ordering. Every coverage vector is indexed by this."""
        return sorted(self._spec)

    def __len__(self) -> int:
        return len(self._spec)

    def index(self, qid: str) -> int:
        return self.qids.index(qid)

    def describe(self, qid: str) -> dict:
        model_id, subset = self._spec[qid]
        mc = self._model_configs[model_id]
        return {
            "qid": qid,
            "model_id": model_id,
            "hf_name": mc.hf_name,
            "params_b": mc.params_b,
            "wrappers": list(subset),
        }

    def get(self, qid: str) -> Configuration:
        """Instantiate on first request; models/wrappers come from shared caches."""
        if qid not in self._instances:
            if qid not in self._spec:
                raise ValueError(f"Unknown qid '{qid}'. Menu has: {', '.join(self.qids)}")
            model_id, subset = self._spec[qid]
            model = load_model(self._model_configs[model_id])
            wrappers = [load_wrapper(self._wrapper_configs[w]) for w in subset]
            self._instances[qid] = Configuration(model, wrappers)
        return self._instances[qid]

    def configurations(self) -> list[Configuration]:
        return [self.get(q) for q in self.qids]
