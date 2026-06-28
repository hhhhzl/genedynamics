"""DiffuseBotSolver — high-level entry point (thin shell, mirrors SHACSolver).

Physics-augmented generative diffusion co-design (Wang et al., NeurIPS 2023),
reproduced against /workspace/DiffuseBot. Thin shell that selects a backend by
name; the only backend shipped is ``jax`` (differentiable jax_mpm).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .protocols import DiffuseBotConfig
from .backends import DiffuseBotBackendJax


_BACKENDS = {"jax": DiffuseBotBackendJax}


class DiffuseBotSolver:
    """Stateful holder around a DiffuseBot backend.

    Use:
        solver = DiffuseBotSolver(scene, mpm_cfg, cfg, morph_decoder=dec)
        result = solver.solve()        # {x, phi, return_, history}
    """

    def __init__(
        self,
        scene,
        mpm_cfg,
        cfg: DiffuseBotConfig,
        *,
        backend: str = "jax",
        morphology=None,
        morph_decoder=None,
        friction: float = 0.5,
        terrain_height=None,
    ):
        if backend not in _BACKENDS:
            raise ValueError(
                f"DiffuseBot backend {backend!r} not registered; available: {sorted(_BACKENDS)}"
            )
        self.cfg = cfg
        self.backend_name = backend
        self._impl = _BACKENDS[backend](
            scene=scene,
            mpm_cfg=mpm_cfg,
            cfg=cfg,
            morphology=morphology,
            morph_decoder=morph_decoder,
            friction=friction,
            terrain_height=terrain_height,
        )

    def solve(self) -> Dict[str, Any]:
        return self._impl.solve()
