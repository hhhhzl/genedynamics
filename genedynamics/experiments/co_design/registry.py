"""
Baseline Registry: register and discover co-design baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .baseline import BaselineProtocol

_REGISTRY: Dict[str, BaselineProtocol] = {}


def register_baseline(baseline: BaselineProtocol, *, name: Optional[str] = None) -> None:
    """Register a baseline."""
    key = name if name is not None else baseline.name
    _REGISTRY[key] = baseline


def get_baseline(name: str) -> BaselineProtocol:
    """Get baseline by name. Raises KeyError if not found."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Baseline '{name}' not found. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_baselines() -> List[str]:
    """List registered baseline names."""
    return list(_REGISTRY.keys())


def has_baseline(name: str) -> bool:
    """Check if baseline is registered."""
    return name in _REGISTRY


class BaselineRegistry:
    """Fluent API for baseline registration."""

    def __init__(self) -> None:
        self._pending: Dict[str, BaselineProtocol] = {}

    def register(self, baseline: BaselineProtocol, *, name: Optional[str] = None) -> "BaselineRegistry":
        key = name if name is not None else baseline.name
        self._pending[key] = baseline
        return self

    def freeze(self) -> None:
        for key, bl in self._pending.items():
            register_baseline(bl, name=key)
        self._pending.clear()
