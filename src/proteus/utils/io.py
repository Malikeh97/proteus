"""Records and JSONL I/O.

Records are appended one per trial so a preempted 23h job loses at most one
prompt; `load_completed_ids` is what makes `--resume` work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    """One request: a draw q ~ c(x), the response it generated, the adjudication."""

    step: int
    prompt: str
    response: str
    qid: str  # which configuration was drawn -- defender history only
    outcome: str  # jailbreak | safe | refused
    refused_by: str | None = None  # wrapper_id, or None if the model self-refused
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wrapper_tokens: int = 0
    attacker_tokens: int = 0


@dataclass
class TrialRecord:
    prompt_id: str
    base_prompt: str
    split: str  # fit | eval | benign
    round: int
    seed: int
    coverage_id: str
    attacker_id: str
    success: bool = False
    first_success_step: int | None = None
    steps: list[StepRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrialRecord:
        steps = [StepRecord(**s) for s in d.pop("steps", [])]
        return cls(steps=steps, **d)

    def refused(self) -> bool:
        """Whether the served interaction ended in a refusal (drives Help(c))."""
        return bool(self.steps) and self.steps[-1].outcome == "refused"


def append_jsonl(record: TrialRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[TrialRecord]:
    if not Path(path).exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(TrialRecord.from_dict(json.loads(line)))
    return records


def load_completed_ids(path: Path) -> set[str]:
    return {r.prompt_id for r in read_jsonl(path)}


def write_json(obj: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
