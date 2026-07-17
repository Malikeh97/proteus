"""Observer interface for the game loop.

The round loop emits typed events (round start, coverage committed, per-request
step, trial end, round end) built entirely from the dataclasses that already
flow through the game -- `Coverage`, `ServedRound`, `TrialRecord`, `Prompt`,
`Outcome`. A renderer subscribes to watch the game unfold; the default is a
no-op, so a headless run is byte-for-byte identical to having no observer at all.

This module deliberately imports no UI library. The `rich` renderer lives in
`live_ui.py` and is only imported when `--live` is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Import for typing only. `runner` imports this module, so importing `Prompt`
    # at runtime here would close the cycle; the forward-ref keeps it open.
    from proteus.coverage.coverage import Coverage
    from proteus.game.deployment import ServedRound
    from proteus.game.runner import Prompt
    from proteus.utils.io import TrialRecord


@runtime_checkable
class GameObserver(Protocol):
    """Side-channel view of the game. Every method is best-effort and must never
    influence control flow, disk writes, or resume behaviour."""

    def on_run_start(
        self,
        *,
        menu_id: str,
        selector: str,
        attacker: str,
        tau: float,
        rounds: int,
        seed: int,
    ) -> None: ...

    def on_round_start(self, round_idx: int, total_rounds: int) -> None: ...

    def on_coverage_committed(self, coverage: "Coverage") -> None: ...

    def on_trial_start(self, prompt: "Prompt", kind: str) -> None: ...

    def on_step(self, step: int, budget: int, served: "ServedRound") -> None: ...

    def on_trial_end(self, record: "TrialRecord") -> None: ...

    def on_round_end(
        self, round_idx: int, asr: float, help_rate: float, coverage: "Coverage"
    ) -> None: ...

    def on_run_end(self) -> None: ...


class NullObserver:
    """The default. Every hook is a no-op, so the non-live path is unchanged.

    Also a context manager, so `with observer:` works uniformly whether the
    observer is this no-op or the live renderer."""

    def on_run_start(self, **kwargs) -> None:
        pass

    def on_round_start(self, *args) -> None:
        pass

    def on_coverage_committed(self, *args) -> None:
        pass

    def on_trial_start(self, *args) -> None:
        pass

    def on_step(self, *args) -> None:
        pass

    def on_trial_end(self, *args) -> None:
        pass

    def on_round_end(self, *args) -> None:
        pass

    def on_run_end(self) -> None:
        pass

    def __enter__(self) -> "NullObserver":
        return self

    def __exit__(self, *exc) -> bool:
        return False


NULL_OBSERVER = NullObserver()
