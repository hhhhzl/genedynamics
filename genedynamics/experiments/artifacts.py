"""Bounded host artifact writes shared by research workflows.

No array runtime is imported here. A policy is propagated through the environment
to subprocesses; a filesystem lock serializes writers across those processes.
The policy reserves space for temporary and published generations together.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

POLICY_ENV = "GENEDYNAMICS_ARTIFACT_POLICY"


def canonical_json(value):
    """Compare contracts in their persisted JSON form (tuples become lists)."""
    return json.loads(json.dumps(value, allow_nan=False))


class ResourceExhausted(RuntimeError):
    """Stop submitting work; existing artifacts and probe charges stay intact."""


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*")
               if p.is_file() and not p.is_symlink())


@dataclass(frozen=True)
class ArtifactPolicy:
    root: str
    max_artifact_bytes: int = 10 * 1024**3
    min_free_bytes: int = 20 * 1024**3

    def __post_init__(self):
        if self.max_artifact_bytes < 1 or self.min_free_bytes < 0:
            raise ValueError("artifact limits must be positive/nonnegative")
        object.__setattr__(self, "root", str(Path(self.root).resolve()))

    def check(self, anticipated_bytes=0):
        root = Path(self.root)
        root.mkdir(parents=True, exist_ok=True)
        used, free = directory_bytes(root), shutil.disk_usage(root).free
        if used + anticipated_bytes > self.max_artifact_bytes:
            raise ResourceExhausted(f"artifact quota: {used} + {anticipated_bytes} > {self.max_artifact_bytes}")
        if free - anticipated_bytes < self.min_free_bytes:
            raise ResourceExhausted(f"free-space reserve: {free} - {anticipated_bytes} < {self.min_free_bytes}")
        return {"used_bytes": used, "free_bytes": free, **asdict(self)}

    @contextmanager
    def write(self, path, anticipated_bytes):
        root, path = Path(self.root), Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"artifact write outside policy root: {path}")
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".artifact.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self.check(anticipated_bytes)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def active_policy():
    source = os.environ.get(POLICY_ENV)
    return ArtifactPolicy(**json.loads(Path(source).read_text())) if source else None


def artifact_write(path, anticipated_bytes):
    policy = active_policy()
    return policy.write(path, anticipated_bytes) if policy else nullcontext()


def atomic_json(path, value, *, enforce_policy=True):
    path = Path(path)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    guard = artifact_write(path, len(payload) + 4096) if enforce_policy else nullcontext()
    with guard:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".json-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
