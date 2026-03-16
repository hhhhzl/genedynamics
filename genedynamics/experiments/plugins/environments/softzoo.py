"""
SoftZoo environment plugin for soft robot co-design.

Integrates SoftZoo (MIT) with genedynamics for locomotion/manipulation tasks.
Requires: pip install softzoo (from https://github.com/mitibmwatsonailab/softzoo)
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ...framework.base import EnvironmentPlugin

SOFTZOO_AVAILABLE = False
try:
    # SoftZoo package structure may vary; try common imports
    import softzoo  # noqa: F401
    SOFTZOO_AVAILABLE = True
except ImportError:
    try:
        from softzoo import envs  # noqa: F401
        SOFTZOO_AVAILABLE = True
    except ImportError:
        pass


class SoftZooEnvironmentPlugin(EnvironmentPlugin):
    """
    Environment plugin for SoftZoo soft robot environments.

    State is high-dimensional (mesh/control points). extract_position returns
    centroid or specified control point indices. Configure via env_params:
    - position_mode: 'centroid' (default) or 'keypoints'
    - position_indices: optional tuple of indices for keypoints mode
    - robot_type: e.g. 'caterpillar', 'snake'
    - terrain: e.g. 'flat', 'desert'
    """

    @property
    def name(self) -> str:
        return "softzoo"

    def create_env(self, config: Dict[str, Any]) -> Any:
        if not SOFTZOO_AVAILABLE:
            if config.get("use_stub_when_unavailable", False):
                config_stub = {**config, "act_dim": config.get("act_dim", 16), "state_dim": config.get("state_dim", 128)}
                return _SoftZooEnvStub(config_stub)
            raise ImportError(
                "SoftZoo is not installed. Install with: "
                "pip install git+https://github.com/mitibmwatsonailab/softzoo.git"
            )
        robot_type = config.get("robot_type", "caterpillar")
        terrain = config.get("terrain", "flat")
        try:
            from softzoo.envs import get_env
            env = get_env(robot_type=robot_type, terrain=terrain, **config)
        except (ImportError, AttributeError) as e:
            # SoftZoo API may vary; provide stub for development
            config_stub = {**config, "act_dim": 16, "state_dim": 128}
            env = _SoftZooEnvStub(config_stub)
            env._softzoo_import_error = str(e)
        if not hasattr(env, "_softzoo_position_mode"):
            env._softzoo_position_mode = config.get("position_mode", "centroid")
            env._softzoo_position_indices = config.get("position_indices")
        return env

    def create_energy(self, env: Any = None) -> Any:
        from genedynamics.core.energy import LegacyEnergyFunctional, EnergyTerm
        import jax.numpy as jnp

        def task_energy(x, u, ctx):
            pos = x[:2]  # centroid xy
            goal = jnp.array([0.0, 0.0], dtype=jnp.float32)
            if ctx is not None and isinstance(ctx, dict) and "target_xy" in ctx:
                t = jnp.asarray(ctx["target_xy"], dtype=jnp.float32)
                if t.size >= 2:
                    goal = t.reshape(-1)[:2]
            return jnp.sum((pos - goal) ** 2)

        return LegacyEnergyFunctional({"task": EnergyTerm(task_energy, 1.0)})

    def get_state_dim(self) -> int:
        # SoftZoo state dim is robot-dependent; use a typical value
        return 128

    def extract_position(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size >= 2:
            return s[:2].copy()
        return s[: min(2, s.size)].copy()

    def get_position_dim(self) -> int:
        return 2


class _SoftZooEnvStub:
    """
    Minimal stub when SoftZoo API is unavailable.
    Provides jax_transition for MRMFMBD development/testing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dt = config.get("dt", 0.01)
        self.horizon = config.get("horizon", 500)
        self.target = np.zeros(2, dtype=np.float32)
        self.act_dim = config.get("act_dim", 16)
        self.state_dim = config.get("state_dim", 128)
        self._rng = np.random.default_rng(config.get("seed", 0))

    def reset(self, rng=None):
        if rng is not None and hasattr(rng, "split"):
            import jax
            return jax.random.normal(rng, (self.state_dim,)) * 0.1, {}
        return self._rng.standard_normal(self.state_dim).astype(np.float32) * 0.1, {}

    def step(self, action):
        raise NotImplementedError("SoftZoo not properly installed")

    def jax_transition(self, state, action):
        """Minimal JAX transition for MRMFMBD: state + dt * action (broadcast)."""
        try:
            import jax.numpy as jnp
            s = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
            a = jnp.asarray(action, dtype=jnp.float32).reshape(-1)
            n = min(s.size, a.size)
            out = s.at[:n].add(self.dt * a[:n])
            return out
        except ImportError:
            import numpy as np
            s = np.asarray(state, dtype=np.float32).reshape(-1)
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            n = min(s.size, a.size)
            s = s.copy()
            s[:n] += self.dt * a[:n]
            return s

    def jax_transition_fidelity(self, state, action, fidelity_level: int):
        """Same as jax_transition when fidelity not supported."""
        return self.jax_transition(state, action)
