"""Capabilities are explicit; registration alone does not qualify a backend."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionCapabilities:
    frontend: str
    physics_engine: str
    tensor_runtime: str
    device: str
    vectorized: bool = True
    subset_reset: bool = False
    snapshot: str = "unsupported"
    terminal_observation: bool = True
    contact_readout: bool = False
    qualification: str = "experimental"

    def __post_init__(self):
        if self.snapshot not in {"unsupported", "approximate", "exact"}:
            raise ValueError("snapshot must be unsupported, approximate, or exact")
        if self.qualification not in {"experimental", "verified"}:
            raise ValueError("qualification must be experimental or verified")

    def require(self, **requirements):
        for key, expected in requirements.items():
            if not hasattr(self, key) or getattr(self, key) != expected:
                raise ValueError(f"Execution backend does not satisfy {key}={expected!r}")

    def as_mapping(self):
        return asdict(self)
