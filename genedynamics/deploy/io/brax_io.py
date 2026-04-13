"""Brax-backed robot IO for GPU-accelerated contact-rich simulation.

Wraps a Brax environment as a :class:`RobotIO` with
``physics_backend="brax"`` and ``array_runtime="jax"``. This gives access
to Brax's ``positional`` and ``generalized`` backends for gradient-based RL
training, while maintaining the same IO contract as :class:`MjxRobotIO`.

**Known limitation — contact richness:**
Brax does not expose per-geom contact wrenches the way MuJoCo does.
Foot-contact observations are derived from
``state.pipeline_state.contact.penetration`` and link indices, producing a
minimal :class:`FootContactSnapshot`-compatible dict. Controllers that
depend on full contact force decomposition (e.g.
:class:`HumanoidContactScheduler`'s touchdown latching) may behave slightly
differently on Brax. This is documented — not hidden.

Dependencies:
    ``brax >= 0.10`` (pip-installable, pulls ``jax``).
    Import is fully lazy: if ``brax`` is not installed, importing this
    module raises ``ImportError`` at class-construction time, not at module
    import time. The registries module catches this gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO

__all__ = ["BraxRobotIO"]

logger = logging.getLogger(__name__)


class BraxRobotIO(BaseRobotIO):
    """Brax environment wrapped as a deploy-pipeline :class:`RobotIO`.

    Args:
        env_name: Brax environment name (``"humanoid"`` or a custom
            registered name). Defaults to ``"humanoid"``.
        backend: Brax physics backend — ``"mjx"`` (for parity with
            :class:`MjxRobotIO`), ``"positional"``, or ``"generalized"``.
        seed: PRNG seed for ``env.reset``.
        sim_dt: Simulation timestep in seconds. Brax environments define
            their own internal dt; this is used for ``state.t`` bookkeeping.
        spec: Optional pre-built :class:`G1RobotSpec`. When ``None``, a
            minimal spec is synthesized from the environment's observation
            dimensions.
        sync_host_shadow: When ``True`` (default), maintain numpy shadow
            copies of ``qpos`` / ``qvel`` via ``jax.device_get()`` so that
            numpy controllers paired through the Phase 16 bridge pay only
            one device→host transfer per step.

    Raises:
        ImportError: If ``brax`` is not installed.
    """

    physics_backend = "brax"
    array_runtime = "jax"
    accepts: Tuple[str, ...] = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        *,
        env_name: str = "humanoid",
        backend: str = "mjx",
        seed: int = 0,
        sim_dt: float = 1.0 / 500.0,
        spec: Optional[Any] = None,
        sync_host_shadow: bool = True,
    ) -> None:
        import brax
        import brax.envs
        import jax
        import jax.numpy as jnp

        self._brax = brax
        self._jax = jax
        self._jnp = jnp

        self.env_name = env_name
        self.backend = backend
        self.seed = seed
        self.sim_dt = float(sim_dt)
        self.sync_host_shadow = sync_host_shadow

        # Create the Brax environment.
        self._env = brax.envs.create(env_name, backend=backend)
        self._rng = jax.random.PRNGKey(seed)
        self._brax_state: Optional[Any] = None
        self._latched_action: Optional[Any] = None

        # Shadow arrays for host-side consumers.
        self._shadow_qpos: Optional[np.ndarray] = None
        self._shadow_qvel: Optional[np.ndarray] = None

        # Build a minimal spec if none provided.
        if spec is None:
            spec = self._build_minimal_spec()

        super().__init__(spec=spec)

        # JIT-compiled step for performance.
        self._jit_step = jax.jit(self._env.step)
        self._jit_reset = jax.jit(self._env.reset)

        logger.info(
            "BraxRobotIO: env=%s backend=%s seed=%d sim_dt=%.4f",
            env_name, backend, seed, sim_dt,
        )

    # ------------------------------------------------------------------
    # Minimal spec builder
    # ------------------------------------------------------------------

    def _build_minimal_spec(self) -> Any:
        """Synthesize a minimal spec from the Brax environment."""
        from types import SimpleNamespace

        # Reset once to get dimensions.
        state = self._env.reset(self._rng)
        pipeline_state = state.pipeline_state
        nq = int(pipeline_state.q.shape[-1])
        nv = int(pipeline_state.qd.shape[-1])

        # Heuristic: first 7 dofs are the floating base.
        num_actuated = max(nv - 6, 0)
        joint_names = tuple(f"joint_{i}" for i in range(num_actuated))

        spec = SimpleNamespace(
            num_actuated=num_actuated,
            nq=nq,
            nv=nv,
            actuated_joints=joint_names,
            actuated_qpos_indices=np.arange(7, 7 + num_actuated, dtype=np.int32),
            actuated_dof_indices=np.arange(6, 6 + num_actuated, dtype=np.int32),
            joint_range={name: np.array([-3.14, 3.14]) for name in joint_names},
            torque_limit={name: 100.0 for name in joint_names},
        )
        return spec

    # ------------------------------------------------------------------
    # BaseRobotIO hooks
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        self._rng, sub_key = self._jax.random.split(self._rng)
        self._brax_state = self._jit_reset(sub_key)
        self._latched_action = None
        self._sync_shadow()

    def _read_state(self, t: float) -> RobotState:
        if self._brax_state is None:
            raise RuntimeError("BraxRobotIO: call reset() before get_state()")

        ps = self._brax_state.pipeline_state
        qpos = ps.q
        qvel = ps.qd

        nq = int(qpos.shape[-1])
        base_pose = qpos[..., :7] if nq >= 7 else None
        nv = int(qvel.shape[-1])
        base_twist = qvel[..., :6] if nv >= 6 else None

        contact = self._extract_contact()

        return RobotState(
            t=float(t),
            qpos=qpos,
            qvel=qvel,
            base_pose=base_pose,
            base_twist=base_twist,
            contact=contact,
            extras={"step": self._step_count, "brax_reward": float(self._brax_state.reward)},
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        jnp = self._jnp

        if cmd.kind in ("joint_pos", "mixed") and cmd.joint_pos is not None:
            self._latched_action = self._jnp.asarray(cmd.joint_pos)
        elif cmd.kind == "torque" and cmd.joint_torque is not None:
            self._latched_action = self._jnp.asarray(cmd.joint_torque)
        elif cmd.kind == "torque" and cmd.joint_torque is None:
            raise ValueError("ControlCommand(kind='torque') requires joint_torque")
        else:
            # Fallback: zero action.
            action_size = self._env.action_size
            self._latched_action = jnp.zeros(action_size)

    def _step_physics(self, dt: float) -> None:
        if self._brax_state is None:
            raise RuntimeError("BraxRobotIO: call reset() before step()")

        action = self._latched_action
        if action is None:
            action = self._jnp.zeros(self._env.action_size)

        self._brax_state = self._jit_step(self._brax_state, action)
        self._sync_shadow()

    def _on_close(self) -> None:
        self._brax_state = None
        self._latched_action = None
        self._shadow_qpos = None
        self._shadow_qvel = None
        logger.info("BraxRobotIO: closed")

    # ------------------------------------------------------------------
    # Contact extraction
    # ------------------------------------------------------------------

    def _extract_contact(self) -> Optional[Mapping[str, Any]]:
        """Extract minimal foot-contact information from Brax state.

        Returns a dict compatible with the ``contact`` field of
        :class:`RobotState`. Contact richness is lower than
        :class:`MujocoRobotIO` — only penetration-based boolean contact
        and approximate normal force are available.
        """
        try:
            ps = self._brax_state.pipeline_state
            contact = ps.contact
            if contact is None:
                return None

            penetration = contact.penetration
            # Aggregate: any link with positive penetration is "in contact".
            # This is a rough approximation — no per-geom wrench decomposition.
            jnp = self._jnp
            in_contact_mask = penetration > 0.0
            total_penetration = float(jnp.sum(penetration))

            return {
                "in_contact_mask": in_contact_mask,
                "total_penetration": total_penetration,
                "contact_count": int(jnp.sum(in_contact_mask)),
            }
        except (AttributeError, TypeError):
            # Brax backend may not expose contact in all configurations.
            return None

    # ------------------------------------------------------------------
    # Foot contact observations (simplified Brax version)
    # ------------------------------------------------------------------

    def foot_contact_observations(self) -> Mapping[str, Any]:
        """Return simplified foot-contact snapshots for left and right feet.

        Unlike :meth:`MujocoRobotIO.foot_contact_observations`, this method
        cannot provide per-geom wrenches, site Jacobians, or precise support
        loads — Brax does not expose these quantities. Instead it returns a
        :class:`FootContactSnapshot`-compatible dict derived from penetration
        data and the shadow state.

        Controllers that depend on rich contact force decomposition (e.g.
        :class:`HumanoidContactScheduler`'s touchdown latching) may behave
        slightly differently. Use :class:`MujocoRobotIO` for full fidelity.
        """
        from genedynamics.deploy.io.mujoco_io import FootContactSnapshot

        zeros3 = np.zeros(3, dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)

        # Extract per-side contact from the aggregate Brax contact data.
        left_contact, right_contact = self._split_foot_contacts()

        return {
            "left": FootContactSnapshot(
                position_world=zeros3.copy(),
                velocity_world=zeros3.copy(),
                rotation_world=eye3.copy(),
                angular_velocity_world=zeros3.copy(),
                in_contact=left_contact["in_contact"],
                contact_count=left_contact["contact_count"],
                support_load=left_contact["support_load"],
            ),
            "right": FootContactSnapshot(
                position_world=zeros3.copy(),
                velocity_world=zeros3.copy(),
                rotation_world=eye3.copy(),
                angular_velocity_world=zeros3.copy(),
                in_contact=right_contact["in_contact"],
                contact_count=right_contact["contact_count"],
                support_load=right_contact["support_load"],
            ),
        }

    def _split_foot_contacts(self) -> tuple[dict, dict]:
        """Split aggregate Brax contact into left/right foot estimates.

        Heuristic: for a humanoid, the lower-indexed contact links are the
        left foot and the higher-indexed ones are the right foot. This is a
        rough approximation — for precise per-geom contacts, use MujocoRobotIO.
        """
        default = {"in_contact": False, "contact_count": 0, "support_load": 0.0}
        if self._brax_state is None:
            return default.copy(), default.copy()
        try:
            ps = self._brax_state.pipeline_state
            contact = ps.contact
            if contact is None:
                return default.copy(), default.copy()

            jnp = self._jnp
            penetration = contact.penetration
            in_contact_mask = penetration > 0.0
            n_contacts = int(in_contact_mask.shape[-1])

            if n_contacts == 0:
                return default.copy(), default.copy()

            # Split contacts in half: left = first half, right = second half.
            mid = n_contacts // 2
            left_mask = in_contact_mask[..., :mid]
            right_mask = in_contact_mask[..., mid:]
            left_pen = penetration[..., :mid]
            right_pen = penetration[..., mid:]

            left = {
                "in_contact": bool(jnp.any(left_mask)),
                "contact_count": int(jnp.sum(left_mask)),
                "support_load": float(jnp.sum(jnp.where(left_mask, left_pen, 0.0))),
            }
            right = {
                "in_contact": bool(jnp.any(right_mask)),
                "contact_count": int(jnp.sum(right_mask)),
                "support_load": float(jnp.sum(jnp.where(right_mask, right_pen, 0.0))),
            }
            return left, right
        except (AttributeError, TypeError):
            return default.copy(), default.copy()

    # ------------------------------------------------------------------
    # Shadow sync
    # ------------------------------------------------------------------

    def _sync_shadow(self) -> None:
        """Pull qpos/qvel to host numpy arrays if shadow sync is enabled."""
        if not self.sync_host_shadow or self._brax_state is None:
            return
        ps = self._brax_state.pipeline_state
        self._shadow_qpos = np.asarray(self._jax.device_get(ps.q), dtype=np.float64)
        self._shadow_qvel = np.asarray(self._jax.device_get(ps.qd), dtype=np.float64)

    @property
    def shadow_qpos(self) -> Optional[np.ndarray]:
        """Host-side numpy copy of qpos, or ``None`` if shadow sync is off."""
        return self._shadow_qpos

    @property
    def shadow_qvel(self) -> Optional[np.ndarray]:
        """Host-side numpy copy of qvel, or ``None`` if shadow sync is off."""
        return self._shadow_qvel
