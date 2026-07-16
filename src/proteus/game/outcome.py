"""The outcome map o_T(x, q) and the payoffs it induces.

Sec. 3: outcomes are {Jailbreak, Safe, Refused}; payoffs are binary and
zero-sum. Safe and Refused carry identical payoff and so collapse into the
"covered" case utility-wise -- they are kept distinct only because Refused is
what the helpfulness constraint counts on benign traffic.
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    JAILBREAK = "jailbreak"
    SAFE = "safe"
    REFUSED = "refused"


def attacker_utility(outcome: Outcome) -> int:
    """u_{1,T}: +1 on jailbreak, 0 otherwise."""
    return 1 if outcome == Outcome.JAILBREAK else 0


def defender_utility(outcome: Outcome) -> int:
    """u_{2,T}: -1 on jailbreak, 0 otherwise. Zero-sum with the attacker."""
    return -1 if outcome == Outcome.JAILBREAK else 0


def jailbreak_indicator(outcome: Outcome) -> int:
    """J(q, x) in {0, 1}. x is covered by q iff J(q, x) = 0."""
    return int(outcome == Outcome.JAILBREAK)
