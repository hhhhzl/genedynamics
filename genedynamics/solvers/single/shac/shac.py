"""SHACSolver — high-level entry point for Short-Horizon Actor-Critic.

Industrial pattern matches MRMFMBD: thin solver shell that selects a backend
implementation by name. The only backend shipped is ``jax`` (jax_mpm); the
SHAC paper's PyTorch reference lives under
``third_party/diffrl`` and is consulted for algorithm details only.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from .protocols import SHACConfig
from .backends import SHACBackendJax


_BACKENDS = {"jax": SHACBackendJax}


class SHACSolver:
    """Stateful holder around a SHAC backend.

    Use:
        solver = SHACSolver(scene, mpm_cfg, cfg)
        result = solver.train()        # writes phi, history, final_return

    Construction is cheap. ``train()`` blocks for the full n_episodes loop.
    """

    def __init__(
        self,
        scene,
        mpm_cfg,
        cfg: SHACConfig,
        *,
        backend: str = "jax",
        morphology=None,
        friction: float = 0.5,
        terrain_height=None,
    ):
        if backend not in _BACKENDS:
            raise ValueError(
                f"SHAC backend {backend!r} not registered; "
                f"available: {sorted(_BACKENDS)}"
            )
        self.cfg = cfg
        self.backend_name = backend
        self._impl = _BACKENDS[backend](
            scene=scene,
            mpm_cfg=mpm_cfg,
            cfg=cfg,
            morphology=morphology,
            friction=friction,
            terrain_height=terrain_height,
        )

    def train(self) -> Dict[str, Any]:
        """Run the full SHAC loop. Returns dict with phi, history, final_return."""
        return self._impl.train()
