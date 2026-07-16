"""Smoke tests: no GPU, no network. Check the wiring, not the science.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.coverage.coverage import StaticCoverage
from proteus.coverage.profile import MenuProfile
from proteus.coverage.selectors import load_selector_by_name
from proteus.game.outcome import Outcome, attacker_utility, defender_utility
from proteus.menu.configuration import Configuration, make_qid
from proteus.menu.menu import ResourceMenu
from proteus.menu.models.base import BaseModel, Generation
from proteus.menu.wrappers.base import Verdict, Wrapper
from proteus.metrics.safety import coverage_entropy, support_size
from proteus.utils.config import ModelConfig, WrapperConfig


class FakeModel(BaseModel):
    def __init__(self, model_id="fake", reply="sure, here you go"):
        super().__init__(ModelConfig(model_id=model_id, hf_name="fake/fake", params_b=1.0))
        self._reply = reply

    def generate(self, prompt: str, **kwargs) -> Generation:
        return Generation(text=self._reply, prompt_tokens=10, completion_tokens=5)


class BlockAll(Wrapper):
    def on_input(self, prompt: str) -> Verdict:
        return Verdict.refuse("block_all:input")


def test_qid_is_order_independent():
    assert make_qid("m", ["b", "a"]) == make_qid("m", ["a", "b"])
    assert make_qid("m", []) == "m"


def test_configuration_serves():
    c = Configuration(FakeModel())
    r = c.serve("hi")
    assert not r.refused
    assert r.response == "sure, here you go"


def test_input_wrapper_short_circuits_the_model():
    cfg = WrapperConfig(wrapper_id="block_all", type="x", stage="input")
    c = Configuration(FakeModel(), [BlockAll(cfg)])
    r = c.serve("hi")
    assert r.refused
    assert r.refused_by == "block_all:input"
    assert r.completion_tokens == 0  # the model never ran


def test_menu_enumerates_the_product():
    menu = ResourceMenu.from_name("dev")
    assert len(menu) == 4  # 1 model x 2^2 wrappers
    assert menu.qids == sorted(menu.qids)


def test_static_coverage_normalizes_and_samples():
    menu = ResourceMenu.from_name("dev")
    cov = StaticCoverage(menu, np.array([1.0, 1.0, 0.0, 0.0]), "test")
    p = cov.probs("x")
    assert p.sum() == pytest.approx(1.0)
    assert len(cov.support()) == 2
    assert cov.entropy() == pytest.approx(np.log(2))


def test_point_mass_has_zero_entropy():
    menu = ResourceMenu.from_name("dev")
    cov = StaticCoverage(menu, np.array([1.0, 0.0, 0.0, 0.0]), "point")
    assert cov.entropy() == pytest.approx(0.0)


def test_rejects_degenerate_weights():
    menu = ResourceMenu.from_name("dev")
    with pytest.raises(ValueError):
        StaticCoverage(menu, np.zeros(len(menu)), "zero")
    with pytest.raises(ValueError):
        StaticCoverage(menu, np.array([1.0, 2.0]), "wrong_length")


def _toy_profile(menu: ResourceMenu) -> MenuProfile:
    """Complementary blind spots: q0 fails on x0, q1 fails on x1. No single
    configuration dominates, so a mixture should strictly beat any vertex."""
    qids = menu.qids
    prof = MenuProfile(menu_id=menu.menu_id, qids=qids)
    prof.j_matrix = {
        qids[0]: {"x0": 1, "x1": 0},
        qids[1]: {"x0": 0, "x1": 1},
        qids[2]: {"x0": 1, "x1": 1},
        qids[3]: {"x0": 1, "x1": 1},
    }
    prof.help_rate = {q: 0.9 for q in qids}
    return prof


def test_minimax_mixes_when_blind_spots_are_complementary():
    menu = ResourceMenu.from_name("dev")
    sel = load_selector_by_name("minimax", menu, _toy_profile(menu), tau=0.5)
    cov = sel.select()
    p = cov.probs("")
    # Every vertex loses a target outright (max = 1); the 50/50 mix caps both at 0.5.
    assert cov.entropy() > 0
    assert p[0] == pytest.approx(0.5, abs=0.05)
    assert p[1] == pytest.approx(0.5, abs=0.05)


def test_deterministic_selector_returns_a_point_mass():
    menu = ResourceMenu.from_name("dev")
    sel = load_selector_by_name("deterministic", menu, _toy_profile(menu), tau=0.5)
    cov = sel.select()
    assert cov.entropy() == pytest.approx(0.0)
    assert len(cov.support()) == 1


def test_infeasible_tau_falls_back_rather_than_crashing():
    menu = ResourceMenu.from_name("dev")
    prof = _toy_profile(menu)
    sel = load_selector_by_name("uniform", menu, prof, tau=0.99)  # no q clears it
    assert sel.select().probs("").sum() == pytest.approx(1.0)


def test_payoffs_are_zero_sum():
    for o in Outcome:
        assert attacker_utility(o) + defender_utility(o) == 0
    assert attacker_utility(Outcome.JAILBREAK) == 1
    assert attacker_utility(Outcome.SAFE) == attacker_utility(Outcome.REFUSED) == 0


def test_entropy_and_support_agree_on_point_masses():
    assert coverage_entropy({"a": 1.0}) == pytest.approx(0.0)
    assert support_size({"a": 1.0, "b": 0.0}) == 1
