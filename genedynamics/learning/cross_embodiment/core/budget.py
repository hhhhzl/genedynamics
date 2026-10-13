"""Host-side target-interaction ledger; count attempts before executing them."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile


@dataclass
class ProbeBudget:
    limit: int
    path: str | None = None
    attempts: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if isinstance(self.limit,bool) or not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError("budget must be a nonnegative integer")
        if self.path is not None and Path(self.path).exists():
            saved=json.loads(Path(self.path).read_text())
            if saved.get("schema_version")!=1 or saved.get("limit")!=self.limit:
                raise ValueError("budget ledger schema/limit mismatch")
            self.attempts=saved["attempts"]
            if len(self.attempts)>self.limit or any(
                a.get("purpose") not in {"interface","fit","validation","support","controller"}
                or a.get("outcome") not in {"pending","success","failed","stopped"} for a in self.attempts):
                raise ValueError("invalid budget ledger")

    def _persist(self):
        if self.path is None:
            return
        path=Path(self.path); path.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=path.parent,prefix=".ledger-")
        try:
            with os.fdopen(fd,"w") as stream:
                json.dump(self.snapshot(),stream,indent=2,allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temp,path)
        finally:
            if os.path.exists(temp): os.unlink(temp)

    def snapshot(self):
        return {"schema_version":1,"limit":self.limit,"attempts":[dict(a) for a in self.attempts]}

    def reserve(self, purpose: str) -> int:
        if purpose not in {"interface", "fit", "validation", "support", "controller"}:
            raise ValueError("undeclared target interaction purpose")
        if len(self.attempts) >= self.limit:
            raise RuntimeError("target probe budget exhausted")
        self.attempts.append({"purpose": purpose, "outcome": "pending"})
        self._persist()  # durable BEFORE the hidden environment is called
        return len(self.attempts) - 1

    def complete(self, attempt: int, outcome: str):
        if not isinstance(attempt, int) or not 0 <= attempt < len(self.attempts):
            raise ValueError("unknown probe attempt")
        if outcome not in {"success", "failed", "stopped"}:
            raise ValueError("invalid probe outcome")
        if self.attempts[attempt]["outcome"] != "pending":
            raise ValueError("attempt already completed")
        self.attempts[attempt]["outcome"] = outcome
        self._persist()
