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
from proteus.judges.base import Judge, RefusalDetector
from proteus.metrics.safety import coverage_entropy, support_size
from proteus.utils.config import JudgeConfig, MenuConfig, MenuEntry, ModelConfig, WrapperConfig
from pydantic import ValidationError


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


def test_menu_curated_list_is_explicit():
    cfg = MenuConfig(
        menu_id="curated_smoke",
        configs=[
            MenuEntry(model="qwen2.5_3b", wrappers=["safety_prompt"]),
            MenuEntry(model="qwen2.5_3b", wrappers=[]),
        ],
    )
    menu = ResourceMenu(cfg)
    assert len(menu) == 2
    assert menu.qids == [
        "qwen2.5-3b-instruct",
        "qwen2.5-3b-instruct+safety_prompt",
    ]


def test_menu_config_rejects_both_modes():
    with pytest.raises(ValidationError):
        MenuConfig(menu_id="x", models=["m"], configs=[MenuEntry(model="m")])
    with pytest.raises(ValidationError):
        MenuConfig(menu_id="x")  # neither models nor configs


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


def _reasoner_profile(menu: ResourceMenu) -> MenuProfile:
    """q0 is capable but unsafe (high help, always jailbroken); q2/q3 are safe but
    less helpful. A capability-first-then-safety tilt should pick q0 at low risk and
    q2/q3 at high risk."""
    qids = menu.qids
    prof = MenuProfile(menu_id=menu.menu_id, qids=qids)
    prof.help_rate = {qids[0]: 0.95, qids[1]: 0.6, qids[2]: 0.5, qids[3]: 0.4}
    prof.j_matrix = {
        qids[0]: {"x": 1},  # jailbreak_rate 1.0 -> safety 0
        qids[1]: {"x": 1},
        qids[2]: {"x": 0},  # jailbreak_rate 0.0 -> safety 1
        qids[3]: {"x": 0},
    }
    return prof


def test_reasoner_tilts_capability_first_then_safety():
    menu = ResourceMenu.from_name("dev")
    sel = load_selector_by_name("reasoner", menu, _reasoner_profile(menu), tau=0.0)

    # Low risk -> favour the most helpful configuration (start with the strong model).
    sel._risk_score = lambda prompt: 0.0
    p_low = sel.select().probs("benign request")
    assert np.argmax(p_low) == 0  # q0 has the highest help_rate

    # High risk -> reroute mass onto the safest configurations.
    sel._risk_score = lambda prompt: 1.0
    p_high = sel.select().probs("dangerous request")
    assert p_high[2] + p_high[3] > 0.99  # the two safe configs
    assert p_high[0] < 0.01  # the capable-but-unsafe config is abandoned


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


# --- Judge severity (0-10 harm score saved to the logs) ---


class _SevJudge(Judge):
    """A jailbreak on any non-empty response; fixed severity 7."""

    def judge(self, prompt: str, response: str) -> int:
        return int(bool(response.strip()))

    def severity(self, prompt: str, response: str) -> int:
        return 7


class _NoRefusal(RefusalDetector):
    def is_refusal(self, prompt: str, response: str) -> bool:
        return False


class _FixedCoverage:
    """Duck-typed coverage: always serves one fixed configuration (no weights)."""

    coverage_id = "fixed"

    def __init__(self, config: Configuration) -> None:
        self._c = config

    def sample(self, prompt, rng, history=None) -> Configuration:
        return self._c


def test_serve_scores_severity_on_jailbreak():
    from proteus.game.deployment import Deployment

    cov = _FixedCoverage(Configuration(FakeModel()))
    dep = Deployment(cov, _SevJudge(JudgeConfig(judge_id="fake")), _NoRefusal(), np.random.default_rng(0))
    served = dep.serve("attack", objective="do harm")
    assert served.outcome == Outcome.JAILBREAK
    assert served.severity == 7  # the judge scored the jailbroken response


def test_serve_severity_zero_when_not_jailbroken():
    from proteus.game.deployment import Deployment

    cfg = WrapperConfig(wrapper_id="block_all", type="x", stage="input")
    cov = _FixedCoverage(Configuration(FakeModel(), [BlockAll(cfg)]))
    dep = Deployment(cov, _SevJudge(JudgeConfig(judge_id="fake")), _NoRefusal(), np.random.default_rng(0))
    served = dep.serve("attack", objective="do harm")
    assert served.outcome == Outcome.REFUSED  # wrapper blocked
    assert served.severity == 0  # not scored on non-jailbreaks


def test_steprecord_severity_roundtrips_and_is_backward_compatible():
    from proteus.utils.io import StepRecord, TrialRecord

    tr = TrialRecord(
        prompt_id="p", base_prompt="b", split="eval", round=1, seed=0,
        coverage_id="c", attacker_id="a",
        steps=[StepRecord(step=1, prompt="x", response="y", qid="q", outcome="jailbreak", severity=9)],
    )
    assert tr.to_dict()["steps"][0]["severity"] == 9

    # A pre-severity record still loads, defaulting to 0.
    old = {
        "prompt_id": "p", "base_prompt": "b", "split": "eval", "round": 1, "seed": 0,
        "coverage_id": "c", "attacker_id": "a", "success": False, "first_success_step": None,
        "steps": [{"step": 1, "prompt": "x", "response": "y", "qid": "q", "outcome": "safe"}],
        "meta": {},
    }
    assert TrialRecord.from_dict(old).steps[0].severity == 0


def test_severity_metrics_aggregate_jailbreak_harm():
    from proteus.metrics.safety import (
        jailbreak_severities_by_config,
        mean_jailbreak_severity,
        severe_asr,
    )
    from proteus.utils.io import StepRecord, TrialRecord

    def trial(qid, outcome, sev, success):
        return TrialRecord(
            prompt_id="p", base_prompt="b", split="eval", round=1, seed=0,
            coverage_id="c", attacker_id="a", success=success,
            first_success_step=1 if success else None,
            steps=[StepRecord(step=1, prompt="x", response="y", qid=qid, outcome=outcome, severity=sev)],
        )

    recs = [
        trial("qA", "jailbreak", 9, True),   # severe execution-level break
        trial("qB", "jailbreak", 3, True),   # mild hint
        trial("qA", "safe", 0, False),       # not a jailbreak
    ]
    assert mean_jailbreak_severity(recs) == pytest.approx(6.0)     # (9 + 3) / 2
    assert severe_asr(recs, threshold=7) == pytest.approx(1 / 3)   # 1 of 3 attack trials severe
    byq = jailbreak_severities_by_config(recs)
    assert byq == {"qA": [9], "qB": [3]}
