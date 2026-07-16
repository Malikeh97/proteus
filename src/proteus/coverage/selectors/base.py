from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from proteus.coverage.coverage import Coverage
from proteus.coverage.profile import MenuProfile
from proteus.menu.menu import ResourceMenu
from proteus.utils.config import SelectorConfig


class Selector(ABC):
    """Produces the coverage the defender commits at the start of a round.

    Everything a selector may condition on is passed in: the menu, the measured
    profile, the helpfulness floor tau, the mass floor c_min, and the defender's
    history (which retains c_s, q_s, y_s -- unlike the attacker's).
    """

    def __init__(
        self,
        config: SelectorConfig,
        menu: ResourceMenu,
        profile: MenuProfile | None = None,
        tau: float = 0.0,
        c_min: float = 0.0,
    ) -> None:
        self._config = config
        self._menu = menu
        self._profile = profile
        self._tau = tau
        self._c_min = c_min

    @property
    def selector_id(self) -> str:
        return self._config.selector_id

    @property
    def params(self) -> dict:
        return self._config.params

    def _require_profile(self) -> MenuProfile:
        if self._profile is None:
            raise ValueError(
                f"Selector '{self.selector_id}' needs a menu profile. "
                "Run scripts/probe_menu.py first and pass --profile."
            )
        return self._profile

    def _feasible_mask(self) -> np.ndarray:
        """Configurations individually clearing the helpfulness floor.

        Feasibility is a constraint on the mixture (c . help >= tau), not on each
        q; restricting the support to individually-feasible configurations is a
        conservative sufficient condition, and the one uniform/validation use.
        """
        if self._profile is None or self._tau <= 0:
            return np.ones(len(self._menu), dtype=bool)
        mask = self._profile.help_array(self._menu.qids) >= self._tau
        return mask if mask.any() else np.ones(len(self._menu), dtype=bool)

    @abstractmethod
    def select(self, history: Any = None, round_idx: int = 0) -> Coverage: ...
