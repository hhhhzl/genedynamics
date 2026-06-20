"""Soft-robot co-design posed as a (state, action) problem for the GENERIC solver framework.

This is the bridge that makes CO-DESIGN compatible with the trajectory-optimizer
(action, state) abstraction, so ANY registered solver (CEM / CMA-ES / MBD /
MRMFMBD) optimizes theta = (x_morphology, phi_controller) through the standard
``Solver(dynamics, energy).solve(x0, horizon)`` contract — with no per-algorithm
co-design reimplementation and no separate ``baselines/`` layer.

Why a reparameterization is needed (the crux):
  trajectory solvers sample a ZERO-centered, symmetric action (mean 0, |a|<=limit),
  but the design theta has per-dim, ASYMMETRIC bounds and a non-zero prior mean
  (x in [x_lo,x_hi] ~ x_mean; phi in [phi_lo,phi_hi]). So the "action" here is a
  zero-centered vector ``a``; the design is ``theta = mean + scale * a`` clipped to
  bounds. The solver stays untouched; the co-design env owns the reparameterization.

Shape of the problem:
  horizon = 1  — the single action IS the whole design vector theta;
  dynamics  — ``jax_transition(state, a)`` = (decode/mirror morphology) + full MPM
              episode under the controller phi -> a 1-vector state carrying the reward;
  energy    — cost = -reward (so the solver, which maximizes -cost, maximizes reward).

For voxel-occupancy morphology (e.g. crawling_ground) ``a -> theta -> rollout_return``
is pure JAX, so ``jax_transition`` is jit/vmap-able and plugs straight into the solver
backends' batched candidate evaluation with ZERO solver changes. Mesh-robotized bodies
(non-JAX) would instead go through the numpy ``step`` path / an evaluator. Multi-regime
and multi-fidelity attach as: env returns C rewards -> energy marginalizes; a
fidelity-parametrized dynamics -> the solver's fidelity ladder. The morphology prior
``log p0(theta)`` is an MBD-only action prior. Those are additive; this module wires
the core single-regime, single-fidelity bridge that the general solvers consume.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

from genedynamics.core.dynamics import DynamicsModel
from genedynamics.core.energy import LegacyEnergyFunctional
from genedynamics.core.types import Trajectory
from genedynamics.envs.external.jax_mpm.scene import rollout_return


class SoftRobotCoDesignDynamics(DynamicsModel):
    """Horizon-1 co-design "dynamics": action a -> design theta -> MPM reward.

    Exposes ``jax_transition(state, a)`` (pure JAX, jit/vmap-able) consumed by the
    solver backends, plus a numpy ``step`` fallback. ``act_dim`` is the design dim D.
    """

    def __init__(
        self,
        scene: Any,
        cfg: Any,
        *,
        x_opt_dim: int,
        phi_dim: int,
        theta_mean: np.ndarray,
        theta_scale: np.ndarray,
        theta_lo: np.ndarray,
        theta_hi: np.ndarray,
        friction: float = 0.5,
        num_env_steps: int = 200,
        z_sym: bool = False,
        voxel_dims: Optional[Tuple[int, int, int]] = None,
        morph_decoder: Any = None,
    ):
        self.scene = scene
        self.cfg = cfg
        # A2 shape-latent decoder: when set, the x-block is a latent w and
        # rollout-time occupancy is g(w) — so a GENERAL solver (CEM/CMA-ES) on
        # the env bridge searches the SAME 32-d latent as MRMFMBD/DiffuseBot
        # (fair Table-1 comparison), not the raw 1352-d occupancy.
        self._morph_decoder = morph_decoder
        self.x_opt_dim = int(x_opt_dim)
        self.phi_dim = int(phi_dim)
        self.act_dim = int(x_opt_dim + phi_dim)          # D — read by the solver backend
        self.state_dim = 1                                # 1-state carries the reward
        self.friction = float(friction)
        self.num_env_steps = int(num_env_steps)
        self._mean = jnp.asarray(theta_mean, jnp.float32)
        self._scale = jnp.asarray(theta_scale, jnp.float32)
        self._lo = jnp.asarray(theta_lo, jnp.float32)
        self._hi = jnp.asarray(theta_hi, jnp.float32)
        self.z_sym = bool(z_sym)
        self.voxel_dims = tuple(voxel_dims) if voxel_dims is not None else None

    def action_to_theta(self, a):
        """Reparameterize the zero-centered action into the bounded design vector."""
        return jnp.clip(self._mean + self._scale * jnp.asarray(a), self._lo, self._hi)

    def _x_full(self, x_opt):
        if self._morph_decoder is not None:
            return jnp.asarray(self._morph_decoder.decode(x_opt))   # w -> occupancy g(w)
        if self.z_sym:
            vx, vy, vz = self.voxel_dims
            h = x_opt.reshape(vx, vy, vz // 2)
            return jnp.concatenate([h, jnp.flip(h, axis=-1)], axis=-1).reshape(-1)
        return x_opt

    def jax_transition(self, state, action):
        """(state, a) -> 1-state carrying the MPM reward. Pure JAX; jit/vmap-able."""
        theta = self.action_to_theta(action)
        x_full = self._x_full(theta[: self.x_opt_dim])
        phi = theta[self.x_opt_dim:]
        r, _disp, _com = rollout_return(
            x_full, phi, jnp.asarray(self.friction, jnp.float32),
            self.scene, self.cfg, self.num_env_steps,
        )
        return jnp.reshape(r, (1,))

    # DynamicsModel abstract API ------------------------------------------------
    def step(self, x, u):
        return np.asarray(self.jax_transition(jnp.asarray(x, jnp.float32),
                                              jnp.asarray(u, jnp.float32)))

    def rollout(self, x0, actions: List[Any]) -> Trajectory:
        states = [np.asarray(x0, dtype=np.float32)]
        for u in actions:
            states.append(self.step(states[-1], u))
        return Trajectory(states=states, actions=list(actions))


def codesign_energy() -> LegacyEnergyFunctional:
    """Energy = -reward (reward is carried in state[0]); solver maximizes -cost = reward."""
    return LegacyEnergyFunctional({"neg_reward": lambda x, u, info=None: -x[0]})


def build_codesign_problem(
    scene: Any,
    cfg: Any,
    *,
    x_opt_dim: int,
    phi_dim: int,
    x_lo: float,
    x_hi: float,
    x_mean: float,
    phi_lo: float,
    phi_hi: float,
    phi_mean: float = 0.0,
    friction: float = 0.5,
    num_env_steps: int = 200,
    z_sym: bool = False,
    voxel_dims: Optional[Tuple[int, int, int]] = None,
    morph_decoder: Any = None,
    morph_init: Optional[np.ndarray] = None,
):
    """Build (dynamics, energy, x0) for a soft-robot co-design problem.

    Any registered solver can then solve it: ``Solver(dynamics, energy, backend)
    .solve(x0, horizon=1)``. Returns the dynamics (which exposes ``action_to_theta``
    so callers can map the solver's best action back to the design theta).
    """
    x_mean_block = np.full(x_opt_dim, x_mean, dtype=np.float32)
    if morph_init is not None:
        _mi = np.asarray(morph_init, np.float32).ravel()
        if _mi.shape[0] == x_opt_dim:
            x_mean_block = _mi   # prior-seeded init (e.g. encoded TripoSG latent w0)
    theta_mean = np.concatenate([x_mean_block, np.full(phi_dim, phi_mean)]).astype(np.float32)
    theta_scale = np.concatenate([
        np.full(x_opt_dim, (x_hi - x_lo) / 2.0),
        np.full(phi_dim, (phi_hi - phi_lo) / 2.0),
    ]).astype(np.float32)
    theta_lo = np.concatenate([np.full(x_opt_dim, x_lo), np.full(phi_dim, phi_lo)]).astype(np.float32)
    theta_hi = np.concatenate([np.full(x_opt_dim, x_hi), np.full(phi_dim, phi_hi)]).astype(np.float32)
    dyn = SoftRobotCoDesignDynamics(
        scene, cfg, x_opt_dim=x_opt_dim, phi_dim=phi_dim,
        theta_mean=theta_mean, theta_scale=theta_scale, theta_lo=theta_lo, theta_hi=theta_hi,
        friction=friction, num_env_steps=num_env_steps, z_sym=z_sym, voxel_dims=voxel_dims,
        morph_decoder=morph_decoder,
    )
    return dyn, codesign_energy(), np.zeros(1, dtype=np.float32)
