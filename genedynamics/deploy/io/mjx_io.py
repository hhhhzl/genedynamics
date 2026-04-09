"""MJX (MuJoCo XLA) robot IO.

GPU-friendly counterpart to :class:`MujocoRobotIO`. State arrays come out as
``jax.Array`` instances, making this the right pairing for JAX-runtime
controllers (RL policies, future MJX-native WBC).

Compatibility note for **host-runtime controllers** paired with this IO
(notably the WBC controller running ``runtime="numpy"``): :class:`MjxRobotIO`
keeps a shadow ``mujoco.MjData`` synchronized with the device ``mjx.Data``
just before each physics query. The cost is one host↔device transfer per
query, which is acceptable at 200 Hz control. The shadow data is *not*
authoritative — :meth:`step` always advances the device data.

When a JAX-native controller is paired with this IO (e.g. an
``RLController`` with ``runtime="jax"``), no shadow sync occurs and the
pipeline runs entirely on device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO
from genedynamics.deploy.io.mujoco_io import FootContactSnapshot
from genedynamics.robots.g1 import G1RobotSpec, g1_scene_path

__all__ = ["MjxRobotIO"]


class MjxRobotIO(BaseRobotIO):
    """``mjx.Model``/``mjx.Data``-backed implementation of :class:`RobotIO`.

    Args:
        model_xml_path: Optional MJCF override.
        sim_dt: Internal MJX timestep.
        keyframe_name: MJCF key to use as the reset pose.
        spec: Optional pre-built :class:`G1RobotSpec`.
        sync_host_shadow: When ``True`` (default), maintain a synchronized
            ``mujoco.MjData`` so physics queries return numpy arrays. Set to
            ``False`` to skip the sync overhead when paired with a JAX-native
            controller.
    """

    physics_backend = "mjx"
    array_runtime = "jax"
    accepts = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        model_xml_path: Optional[str | Path] = None,
        *,
        sim_dt: float = 0.002,
        keyframe_name: str = "stand",
        spec: Optional[G1RobotSpec] = None,
        sync_host_shadow: bool = True,
    ) -> None:
        import mujoco
        from mujoco import mjx
        import jax
        import jax.numpy as jnp

        self._mujoco = mujoco
        self._mjx = mjx
        self._jax = jax
        self._jnp = jnp

        self.model_path: str = g1_scene_path(model_xml_path, prefer_mjx=True)
        self.sim_dt = float(sim_dt)
        self.keyframe_name = keyframe_name
        self.sync_host_shadow = sync_host_shadow

        cpu_model = mujoco.MjModel.from_xml_path(self.model_path)
        cpu_model.opt.timestep = self.sim_dt
        # MJX requires CG or Newton solvers (PGS unsupported)
        if cpu_model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            cpu_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.cpu_model = cpu_model
        self.model = mjx.put_model(cpu_model)
        self.data = mjx.make_data(self.model)

        # JAX honors x64 only when jax_enable_x64 is set; otherwise falls back
        # to float32. We pick the dtype the runtime is configured for so the
        # MJX path doesn't generate dtype-truncation warnings on every call.
        self._jdtype = jnp.zeros(()).dtype

        # Host-side shadow used for physics queries when sync_host_shadow=True.
        self._shadow_data = mujoco.MjData(cpu_model) if sync_host_shadow else None

        spec = spec or G1RobotSpec.from_mujoco_model(cpu_model, model_xml_path=self.model_path)
        super().__init__(spec=spec)

        self._actuated_qpos_idx = spec.actuated_qpos_indices
        self._actuated_dof_idx = spec.actuated_dof_indices

        # JIT-compiled step closure: avoids recompilation on every step.
        self._jit_step = jax.jit(lambda m, d: mjx.step(m, d))

    # ------------------------------------------------------------------
    # BaseRobotIO hooks
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        mjx, mujoco = self._mjx, self._mujoco
        cpu_data = mujoco.MjData(self.cpu_model)
        kid = mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.cpu_model, cpu_data, kid)
        else:
            mujoco.mj_resetData(self.cpu_model, cpu_data)
        mujoco.mj_forward(self.cpu_model, cpu_data)
        self.data = mjx.put_data(self.cpu_model, cpu_data)
        if self._shadow_data is not None:
            self._shadow_data = cpu_data
            self._shadow_dirty = False
        else:
            self._shadow_dirty = False

    def _read_state(self, t: float) -> RobotState:
        # State arrays come out as jax.Array (the runtime invariant). Callers
        # running ``runtime="numpy"`` convert via jax.device_get when they need
        # host arrays.
        qpos = self.data.qpos
        qvel = self.data.qvel
        nq = self.cpu_model.nq
        base_pose = qpos[:7] if nq >= 7 else None
        base_twist = qvel[:6] if self.cpu_model.nv >= 6 else None
        return RobotState(
            t=float(t),
            qpos=qpos,
            qvel=qvel,
            base_pose=base_pose,
            base_twist=base_twist,
            extras={"step": self._step_count},
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        jnp = self._jnp
        nu = self.cpu_model.nu
        dtype = self._jdtype

        ctrl = jnp.zeros(nu, dtype=dtype)
        if cmd.kind in ("joint_pos", "mixed") and cmd.joint_pos is not None:
            jp = jnp.asarray(cmd.joint_pos, dtype=dtype).reshape(-1)
            ctrl = ctrl.at[: jp.size].set(jp[:nu])
        # Torque feedforward path: apply via qfrc_applied on the data object.
        qfrc = jnp.zeros(self.cpu_model.nv, dtype=dtype)
        if cmd.joint_torque is not None:
            tau = jnp.asarray(cmd.joint_torque, dtype=dtype).reshape(-1)
            idx = jnp.asarray(self._actuated_dof_idx[: tau.size])
            qfrc = qfrc.at[idx].set(tau[: idx.size])
        if cmd.kind == "torque" and cmd.joint_torque is None:
            raise ValueError("ControlCommand(kind='torque') requires joint_torque")
        self.data = self.data.replace(ctrl=ctrl, qfrc_applied=qfrc)
        self._shadow_dirty = True

    def _step_physics(self, dt: float) -> None:
        substeps = max(1, int(round(dt / self.sim_dt)))
        for _ in range(substeps):
            self.data = self._jit_step(self.model, self.data)
        self._shadow_dirty = True

    def _on_close(self) -> None:
        self.data = None
        self.model = None
        self._shadow_data = None

    # ------------------------------------------------------------------
    # Physics queries — return numpy via host shadow sync
    # ------------------------------------------------------------------

    def _ensure_shadow_synced(self) -> None:
        if self._shadow_data is None:
            raise RuntimeError(
                "MjxRobotIO was constructed with sync_host_shadow=False; "
                "physics queries are unavailable. Pair this IO with a "
                "JAX-native controller or set sync_host_shadow=True."
            )
        if not self._shadow_dirty:
            return
        # Pull device → host. mjx → mujoco data conversion.
        sd = self._shadow_data
        sd.qpos[:] = np.asarray(self._jax.device_get(self.data.qpos), dtype=np.float64)
        sd.qvel[:] = np.asarray(self._jax.device_get(self.data.qvel), dtype=np.float64)
        sd.ctrl[:] = np.asarray(self._jax.device_get(self.data.ctrl), dtype=np.float64)
        self._mujoco.mj_forward(self.cpu_model, sd)
        self._shadow_dirty = False

    def mass_matrix(self) -> np.ndarray:
        self._ensure_shadow_synced()
        M = np.zeros((self.cpu_model.nv, self.cpu_model.nv), dtype=np.float64)
        self._mujoco.mj_fullM(self.cpu_model, M, self._shadow_data.qM)
        return M

    def bias(self) -> np.ndarray:
        self._ensure_shadow_synced()
        return np.asarray(self._shadow_data.qfrc_bias, dtype=np.float64).copy()

    def site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_shadow_synced()
        sid = self._site_id(site_name)
        pos = np.asarray(self._shadow_data.site_xpos[sid], dtype=np.float64).copy()
        R = np.asarray(self._shadow_data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
        return pos, R

    def site_jacobian(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_shadow_synced()
        sid = self._site_id(site_name)
        jacp = np.zeros((3, self.cpu_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.cpu_model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.cpu_model, self._shadow_data, jacp, jacr, sid)
        return jacp, jacr

    def body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_shadow_synced()
        bid = self._body_id(body_name)
        pos = np.asarray(self._shadow_data.xpos[bid], dtype=np.float64).copy()
        quat = np.asarray(self._shadow_data.xquat[bid], dtype=np.float64).copy()
        return pos, quat

    def subtree_com_jacobian(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_shadow_synced()
        bid = self._body_id(body_name)
        jacp = np.zeros((3, self.cpu_model.nv), dtype=np.float64)
        self._mujoco.mj_jacSubtreeCom(self.cpu_model, self._shadow_data, jacp, bid)
        com = np.asarray(self._shadow_data.subtree_com[bid], dtype=np.float64).copy()
        return jacp, com

    def foot_contact_observations(self) -> Mapping[str, FootContactSnapshot]:
        # Reuse the MujocoRobotIO logic against the shadow data. Implemented
        # inline rather than via inheritance because the shadow's contact
        # array is updated by mj_forward (no collision pass), so we delegate
        # to mj_step on the cpu shadow whenever this is requested.
        self._ensure_shadow_synced()
        # mj_step on the shadow once to populate contacts at the current state
        # without advancing time perceptibly: snapshot, step, restore.
        sd = self._shadow_data
        snap_qpos = sd.qpos.copy()
        snap_qvel = sd.qvel.copy()
        snap_ctrl = sd.ctrl.copy()
        try:
            self._mujoco.mj_step(self.cpu_model, sd)
            return {
                "left": self._foot_contact_observation("left"),
                "right": self._foot_contact_observation("right"),
            }
        finally:
            sd.qpos[:] = snap_qpos
            sd.qvel[:] = snap_qvel
            sd.ctrl[:] = snap_ctrl
            self._mujoco.mj_forward(self.cpu_model, sd)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _site_id(self, name: str) -> int:
        sid = self.spec.site_id.get(name)
        if sid is None:
            raise KeyError(f"Site not found in G1 model: {name}")
        return int(sid)

    def _body_id(self, name: str) -> int:
        bid = self.spec.body_id.get(name)
        if bid is None:
            raise KeyError(f"Body not found in G1 model: {name}")
        return int(bid)

    def _foot_contact_observation(self, side: str) -> FootContactSnapshot:
        from genedynamics.robots.g1.spec import G1_FOOT_BODY_NAMES, G1_FOOT_SITE_NAMES

        site_name = G1_FOOT_SITE_NAMES[0 if side == "left" else 1]
        body_name = G1_FOOT_BODY_NAMES[0 if side == "left" else 1]

        sid = self._site_id(site_name)
        sd = self._shadow_data
        jacp = np.zeros((3, self.cpu_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.cpu_model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.cpu_model, sd, jacp, jacr, sid)
        pos = np.asarray(sd.site_xpos[sid], dtype=np.float64).copy()
        R = np.asarray(sd.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
        vel = jacp @ sd.qvel
        ang_vel = jacr @ sd.qvel

        bid = self.spec.body_id.get(body_name)
        in_contact = False
        contact_count = 0
        support_load = 0.0
        if bid is not None:
            for ci in range(int(sd.ncon)):
                contact = sd.contact[ci]
                body1 = int(self.cpu_model.geom_bodyid[int(contact.geom1)])
                body2 = int(self.cpu_model.geom_bodyid[int(contact.geom2)])
                if bid not in (body1, body2):
                    continue
                in_contact = True
                contact_count += 1
                wrench = np.zeros(6, dtype=np.float64)
                self._mujoco.mj_contactForce(self.cpu_model, sd, ci, wrench)
                support_load += max(float(wrench[0]), 0.0)

        return FootContactSnapshot(
            position_world=pos,
            velocity_world=np.asarray(vel, dtype=np.float64).copy(),
            rotation_world=R,
            angular_velocity_world=np.asarray(ang_vel, dtype=np.float64).copy(),
            in_contact=in_contact,
            contact_count=contact_count,
            support_load=support_load,
        )
