"""Records and JSONL I/O.

Records are appended one per trial so a preempted 23h job loses at most one
prompt; `load_completed_ids` is what makes `--resume` work.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from proteus.utils.logging import get_logger

logger = get_logger("io")


@dataclass
class StepRecord:
    """One request: a draw q ~ c(x), the response it generated, the adjudication."""

    step: int
    prompt: str
    response: str
    qid: str  # which configuration was drawn -- defender history only
    outcome: str  # jailbreak | safe | refused
    refused_by: str | None = None  # wrapper_id, or None if the model self-refused
    severity: int = 0  # 0-10 harm score, judge-scored on jailbreaks (0 otherwise)
    quality: int = 0  # 1-5 answer quality, judge-scored on served benign traffic (0 otherwise)
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
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TrialRecord.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                # A preempted job can die mid-write, and since probe shards are read
                # by other jobs while they are still being appended to, the last line
                # may be half-written. Dropping it costs one trial; raising would cost
                # the whole file.
                logger.warning("Skipping unparseable line %d of %s", i, path)
    return records


def load_completed_ids(path: Path) -> set[str]:
    return {r.prompt_id for r in read_jsonl(path)}


def write_json(obj: Any, path: Path) -> None:
    """Atomic: written to a temp file in the same directory and renamed, so a
    reader never sees a half-written file and two concurrent writers cannot
    interleave. Phase 1 relies on this -- every probe job attempts the profile
    merge, so several may write profile.json at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
