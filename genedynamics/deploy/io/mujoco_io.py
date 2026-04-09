"""NumPy MuJoCo robot IO.

Wraps a ``mujoco.MjModel`` + ``mujoco.MjData`` pair behind the
:class:`RobotIO` protocol. State arrays come out as :class:`numpy.ndarray`,
making this the right pairing for controllers running on the host
(``runtime="numpy"``): WBC + osqp, joint-space PD, sport-mode mock.

Beyond the protocol, this class exposes a small set of **physics query**
methods that the WBC controller and footstep planners need. They are
deliberately scoped to "pull state out of MuJoCo" and contain no controller
logic. Each query reuses the IO's already-stepped ``MjData`` and never
mutates it; controllers can call them freely between :meth:`get_state` and
:meth:`send_control`.

Physics queries (numpy arrays, no JAX involvement):

* :meth:`mass_matrix`            — full ``(nv, nv)`` joint-space inertia
* :meth:`bias`                   — coriolis + gravity + applied
* :meth:`site_pose`              — ``(pos, R)`` of a named MuJoCo site
* :meth:`site_jacobian`          — translational + rotational ``(6, nv)``
* :meth:`subtree_com_jacobian`   — CoM Jacobian and current CoM
* :meth:`body_pose`              — ``(pos, quat_wxyz)`` of a named body
* :meth:`foot_contact_observations` — both feet, used by footstep planners
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO
from genedynamics.robots.g1 import G1RobotSpec, g1_scene_path
from genedynamics.robots.g1.spec import G1_FOOT_BODY_NAMES, G1_FOOT_SITE_NAMES

__all__ = ["MujocoRobotIO", "FootContactSnapshot"]


# A lightweight container instead of importing the existing
# `FootContactObservation` so this module stays decoupled from the legacy
# `followers/humanoid/task_spec.py`. The WBC controller adapts as needed.
from dataclasses import dataclass


@dataclass
class FootContactSnapshot:
    position_world: np.ndarray
    velocity_world: np.ndarray
    rotation_world: np.ndarray
    angular_velocity_world: np.ndarray
    in_contact: bool
    contact_count: int = 0
    support_load: float = 0.0


class MujocoRobotIO(BaseRobotIO):
    """``mujoco.MjModel``/``MjData``-backed implementation of :class:`RobotIO`.

    Args:
        model_xml_path: Optional MJCF override. Defaults to the resolved
            G1 scene path. The IO is currently G1-shaped (uses
            :class:`G1RobotSpec`); generalizing to other humanoids is a
            matter of swapping the spec class.
        sim_dt: Internal timestep written into ``model.opt.timestep``.
            :meth:`step` accepts a higher-level ``dt`` and substeps as needed.
        keyframe_name: MJCF key to use as the reset pose. Defaults to
            ``"stand"`` (matches the G1 menagerie scene).
        spec: Optional pre-built :class:`G1RobotSpec`. When omitted the spec
            is constructed by introspecting the loaded model.
    """

    physics_backend = "mujoco"
    array_runtime = "numpy"
    accepts = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        model_xml_path: Optional[str | Path] = None,
        *,
        sim_dt: float = 0.002,
        keyframe_name: str = "stand",
        spec: Optional[G1RobotSpec] = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model_path: str = g1_scene_path(model_xml_path)
        self.sim_dt = float(sim_dt)
        self.keyframe_name = keyframe_name

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        spec = spec or G1RobotSpec.from_mujoco_model(self.model, model_xml_path=self.model_path)
        super().__init__(spec=spec)

        self._actuated_qpos_idx = spec.actuated_qpos_indices
        self._actuated_dof_idx = spec.actuated_dof_indices

    # ------------------------------------------------------------------
    # BaseRobotIO hooks
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        kid = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name
        )
        if kid >= 0:
            self._mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

    def _read_state(self, t: float) -> RobotState:
        nq = self.model.nq
        qpos = np.asarray(self.data.qpos[:nq], dtype=np.float64).copy()
        qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()

        base_pose: Optional[np.ndarray] = None
        base_twist: Optional[np.ndarray] = None
        if nq >= 7:
            base_pose = qpos[:7].copy()  # x, y, z, qw, qx, qy, qz
        if self.model.nv >= 6:
            base_twist = qvel[:6].copy()

        return RobotState(
            t=float(t),
            qpos=qpos,
            qvel=qvel,
            base_pose=base_pose,
            base_twist=base_twist,
            extras={"step": self._step_count},
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        nu = self.model.nu
        ctrl = np.zeros(nu, dtype=np.float64)
        qfrc = np.zeros(self.model.nv, dtype=np.float64)

        if cmd.kind == "joint_pos":
            if cmd.joint_pos is None:
                raise ValueError("ControlCommand(kind='joint_pos') requires joint_pos")
            jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
            ctrl[: jp.size] = jp[:nu]
            if cmd.joint_torque is not None:
                tau = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
                qfrc[self._actuated_dof_idx[: tau.size]] = tau[: self._actuated_dof_idx.size]
        elif cmd.kind == "torque":
            if cmd.joint_torque is None:
                raise ValueError("ControlCommand(kind='torque') requires joint_torque")
            tau = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
            qfrc[self._actuated_dof_idx[: tau.size]] = tau[: self._actuated_dof_idx.size]
        elif cmd.kind == "mixed":
            # Lower body comes from loco_cmd (interpreted by a leg controller),
            # upper body from joint_pos. We delegate the lower-body translation
            # to the controller; the IO only writes joint_pos and (optional) tau.
            if cmd.joint_pos is not None:
                jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
                ctrl[: jp.size] = jp[:nu]
            if cmd.joint_torque is not None:
                tau = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
                qfrc[self._actuated_dof_idx[: tau.size]] = tau[: self._actuated_dof_idx.size]
        else:
            raise ValueError(f"MujocoRobotIO does not handle ControlCommand.kind={cmd.kind!r}")

        self.data.ctrl[:] = ctrl
        self.data.qfrc_applied[:] = qfrc

    def _step_physics(self, dt: float) -> None:
        substeps = max(1, int(round(dt / self.sim_dt)))
        for _ in range(substeps):
            self._mujoco.mj_step(self.model, self.data)

    def _on_close(self) -> None:
        # MjModel / MjData own native memory but mujoco-py manages release on
        # garbage collection — explicit close is a no-op.
        pass

    # ------------------------------------------------------------------
    # Reset overrides
    # ------------------------------------------------------------------

    def reset_to(
        self,
        *,
        base_xyz: Optional[Sequence[float]] = None,
        base_quat_wxyz: Optional[Sequence[float]] = None,
        actuated_qpos: Optional[Sequence[float]] = None,
    ) -> RobotState:
        """Reset to the keyframe pose, optionally overriding base / joints.

        Provided as a convenience for trajectory-aligned resets — the
        previous ``HumanoidMujocoPipeline._align_root_to_plan_frame`` logic
        moves here.
        """
        state = self.reset()
        if base_xyz is None and base_quat_wxyz is None and actuated_qpos is None:
            return state
        if self.model.nq >= 3 and base_xyz is not None:
            self.data.qpos[0:3] = np.asarray(base_xyz, dtype=np.float64).reshape(3)
        if self.model.nq >= 7 and base_quat_wxyz is not None:
            self.data.qpos[3:7] = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
        if actuated_qpos is not None:
            self.data.qpos[:] = self.spec.apply_actuated_qpos(self.data.qpos, actuated_qpos)
        self._mujoco.mj_forward(self.model, self.data)
        return self._read_state(t=self._t)

    # ------------------------------------------------------------------
    # Physics queries (no controller logic, just MuJoCo readbacks)
    # ------------------------------------------------------------------

    def mass_matrix(self) -> np.ndarray:
        """Full joint-space inertia matrix ``M(q)``, shape ``(nv, nv)``."""
        M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        self._mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def bias(self) -> np.ndarray:
        """Bias term ``c(q, qd) + g(q) - F_ext``, shape ``(nv,)``."""
        return np.asarray(self.data.qfrc_bias, dtype=np.float64).copy()

    def site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(position, rotation_matrix)`` for a named site."""
        sid = self._site_id(site_name)
        pos = np.asarray(self.data.site_xpos[sid], dtype=np.float64).copy()
        R = np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
        return pos, R

    def site_jacobian(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(jacp, jacr)``, each shape ``(3, nv)``."""
        sid = self._site_id(site_name)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
        return jacp, jacr

    def body_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(position, quaternion_wxyz)`` for a named body."""
        bid = self._body_id(body_name)
        pos = np.asarray(self.data.xpos[bid], dtype=np.float64).copy()
        quat = np.asarray(self.data.xquat[bid], dtype=np.float64).copy()
        return pos, quat

    def subtree_com_jacobian(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(jacp, com_world)`` for the subtree rooted at ``body_name``."""
        bid = self._body_id(body_name)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._mujoco.mj_jacSubtreeCom(self.model, self.data, jacp, bid)
        com = np.asarray(self.data.subtree_com[bid], dtype=np.float64).copy()
        return jacp, com

    def foot_contact_observations(self) -> Mapping[str, FootContactSnapshot]:
        """Return per-foot contact snapshots keyed by ``"left"`` / ``"right"``.

        Computes contact wrenches via :func:`mujoco.mj_contactForce` for any
        contact involving the corresponding foot body.
        """
        return {
            "left": self._foot_contact_observation("left"),
            "right": self._foot_contact_observation("right"),
        }

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
        if side == "left":
            site_name, body_name = G1_FOOT_SITE_NAMES[0], G1_FOOT_BODY_NAMES[0]
        elif side == "right":
            site_name, body_name = G1_FOOT_SITE_NAMES[1], G1_FOOT_BODY_NAMES[1]
        else:
            raise ValueError(f"Foot side must be 'left' or 'right', got {side!r}")

        jacp, jacr = self.site_jacobian(site_name)
        pos, R = self.site_pose(site_name)
        vel = jacp @ self.data.qvel
        ang_vel = jacr @ self.data.qvel

        bid = self.spec.body_id.get(body_name)
        in_contact = False
        contact_count = 0
        support_load = 0.0
        if bid is not None:
            for ci in range(int(self.data.ncon)):
                contact = self.data.contact[ci]
                body1 = int(self.model.geom_bodyid[int(contact.geom1)])
                body2 = int(self.model.geom_bodyid[int(contact.geom2)])
                if bid not in (body1, body2):
                    continue
                in_contact = True
                contact_count += 1
                wrench = np.zeros(6, dtype=np.float64)
                self._mujoco.mj_contactForce(self.model, self.data, ci, wrench)
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
