"""Unitree G1 robot model.

Implements the :class:`RobotModel` protocol from :mod:`genedynamics.envs.robots.base`
on top of MuJoCo. The model is stateless from the caller's perspective: every
public method takes joint positions (and where relevant a base pose) and
returns the requested quantity. Internally a private ``mujoco.MjData`` scratch
is reused to avoid per-call allocation.

The actuated joint vector follows :data:`G1_ACTUATED_JOINTS` (legs → waist →
arms, 29 entries). For floating-base operations the base pose is supplied
explicitly via the ``base_pos`` and ``base_quat`` keyword arguments; if
omitted the base is pinned at the origin in identity orientation, which is
the convention used by the leg-IK helper.

Example::

    from genedynamics.envs.robots.g1 import G1RobotModel

    g1 = G1RobotModel()
    pos, quat = g1.forward_kinematics(g1.spec.stand_ctrl, link_name="pelvis")

The same instance also exposes the underlying :class:`G1RobotSpec` via
``g1.spec`` for callers that need joint indices, limits, or torque bounds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from genedynamics.envs.robots.base import RobotModelMixin
from genedynamics.robots.g1.assets import g1_scene_path
from genedynamics.robots.g1.spec import G1_ACTUATED_JOINTS, G1RobotSpec

__all__ = ["G1RobotModel"]


_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class G1RobotModel(RobotModelMixin):
    """MuJoCo-backed kinematics / dynamics model for the Unitree G1.

    Args:
        model_xml_path: Optional explicit MJCF path. Defaults to the resolved
            G1 scene XML.
        default_link: Default end-effector link name returned by FK / jacobian
            when no link name is supplied. Defaults to ``"pelvis"``.
    """

    def __init__(
        self,
        model_xml_path: Optional[str] = None,
        *,
        default_link: str = "pelvis",
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        path = g1_scene_path(model_xml_path)
        self._model = mujoco.MjModel.from_xml_path(path)
        self._data = mujoco.MjData(self._model)
        self.spec = G1RobotSpec.from_mujoco_model(self._model, model_xml_path=path)
        self.default_link = default_link

        # Cached index arrays (avoid recomputation in inner loops).
        self._actuated_qpos_idx = self.spec.actuated_qpos_indices
        self._actuated_dof_idx = self.spec.actuated_dof_indices

        # RobotModel protocol metadata.
        super().__init__(
            n_dof=self.spec.num_actuated,
            joint_names=list(self.spec.actuated_joints),
            link_names=self._collect_body_names(),
        )

    # ------------------------------------------------------------------
    # Public protocol surface
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The underlying ``mujoco.MjModel`` (read-only convention)."""
        return self._model

    @property
    def nq(self) -> int:
        return int(self._model.nq)

    @property
    def nv(self) -> int:
        return int(self._model.nv)

    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None,
        *,
        base_pos: Optional[np.ndarray] = None,
        base_quat_wxyz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the world pose of ``link_name``.

        Args:
            joint_positions: Actuated joint vector, shape ``(num_actuated,)``.
            link_name: Body name. Defaults to ``self.default_link``.
            base_pos: Base position in world frame, shape ``(3,)``. Defaults
                to the origin.
            base_quat_wxyz: Base orientation as a unit quaternion in
                ``(w, x, y, z)`` order. Defaults to identity.

        Returns:
            ``(position, quaternion_wxyz)`` of the requested link, both numpy
            arrays of dtype ``float64``.
        """
        self._write_state(joint_positions, base_pos, base_quat_wxyz, qvel=None)
        self._mujoco.mj_kinematics(self._model, self._data)
        bid = self._body_id(link_name)
        pos = np.asarray(self._data.xpos[bid], dtype=np.float64).copy()
        quat = np.asarray(self._data.xquat[bid], dtype=np.float64).copy()
        return pos, quat

    def jacobian(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None,
        *,
        base_pos: Optional[np.ndarray] = None,
        base_quat_wxyz: Optional[np.ndarray] = None,
        actuated_only: bool = True,
    ) -> np.ndarray:
        """Spatial Jacobian of ``link_name``.

        Args:
            joint_positions: Actuated joint vector, shape ``(num_actuated,)``.
            link_name: Body name. Defaults to ``self.default_link``.
            actuated_only: If ``True`` returns the ``(6, num_actuated)`` slice
                aligned with ``self.spec.actuated_dof_indices``. If ``False``
                returns the full ``(6, nv)`` Jacobian (useful when working
                with the floating base).

        Returns:
            Stacked translational + rotational Jacobian, shape ``(6, n)``.
        """
        self._write_state(joint_positions, base_pos, base_quat_wxyz, qvel=None)
        self._mujoco.mj_kinematics(self._model, self._data)
        self._mujoco.mj_comPos(self._model, self._data)
        bid = self._body_id(link_name)
        jacp = np.zeros((3, self.nv), dtype=np.float64)
        jacr = np.zeros((3, self.nv), dtype=np.float64)
        self._mujoco.mj_jacBody(self._model, self._data, jacp, jacr, bid)
        full = np.vstack([jacp, jacr])
        if actuated_only:
            return full[:, self._actuated_dof_idx].copy()
        return full

    def forward_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        *,
        base_pos: Optional[np.ndarray] = None,
        base_quat_wxyz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward dynamics: ``tau → qdd`` (full ``nv`` vector)."""
        self._write_state(joint_positions, base_pos, base_quat_wxyz, qvel=joint_velocities)
        ctrl = np.zeros(self._model.nu, dtype=np.float64)
        for i, name in enumerate(self.spec.actuated_joints):
            aid = self.spec.actuator_id.get(name)
            if aid is not None:
                ctrl[aid] = float(joint_torques[i])
        self._data.ctrl[:] = ctrl
        self._mujoco.mj_forward(self._model, self._data)
        return np.asarray(self._data.qacc, dtype=np.float64).copy()

    def inverse_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        *,
        base_pos: Optional[np.ndarray] = None,
        base_quat_wxyz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Inverse dynamics: ``(q, qd, qdd) → tau`` for actuated DoFs."""
        self._write_state(joint_positions, base_pos, base_quat_wxyz, qvel=joint_velocities)
        qacc_full = self._broadcast_actuated_qvel(joint_accelerations)
        self._data.qacc[:] = qacc_full
        self._mujoco.mj_inverse(self._model, self._data)
        tau_full = np.asarray(self._data.qfrc_inverse, dtype=np.float64)
        return tau_full[self._actuated_dof_idx].copy()

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        *,
        link_name: Optional[str] = None,
        base_pos: Optional[np.ndarray] = None,
        base_quat_wxyz: Optional[np.ndarray] = None,
        max_iter: int = 50,
        tol: float = 1e-4,
        damping: float = 1e-2,
        step_clip: float = 0.3,
    ) -> np.ndarray:
        """Damped least-squares IK for a single body.

        This is a generic Newton-style IK suitable for end-effector targeting
        on the upper body. For high-frequency leg control prefer a closed-form
        leg IK; in this codebase that responsibility lives in the spark RL
        policy weights (see ``SparkRLLocoClient``).

        Returns:
            Actuated joint vector, shape ``(num_actuated,)``.
        """
        q = (
            np.asarray(initial_guess, dtype=np.float64).copy()
            if initial_guess is not None
            else self.spec.stand_ctrl.copy()
        )
        target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_R = (
            _quat_to_rot(np.asarray(target_orientation, dtype=np.float64))
            if target_orientation is not None
            else None
        )

        for _ in range(max_iter):
            pos, quat = self.forward_kinematics(
                q, link_name=link_name, base_pos=base_pos, base_quat_wxyz=base_quat_wxyz
            )
            err_pos = target_pos - pos
            if target_R is not None:
                err_rot = _orientation_error(target_R, _quat_to_rot(quat))
                err = np.concatenate([err_pos, err_rot])
            else:
                err = err_pos

            if np.linalg.norm(err) < tol:
                break

            J = self.jacobian(
                q, link_name=link_name, base_pos=base_pos, base_quat_wxyz=base_quat_wxyz
            )
            if target_R is None:
                J = J[:3]
            JJT = J @ J.T + damping**2 * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJT, err)
            dq = np.clip(dq, -step_clip, step_clip)
            q = q + dq
            q = self.spec.clip_to_joint_limits(q)

        return q

    # ------------------------------------------------------------------
    # Closed-form-ish leg IK (used by Phase 5 sport-mode walker)
    # ------------------------------------------------------------------

    def solve_leg_ik(
        self,
        side: str,
        foot_pos_pelvis: np.ndarray,
        *,
        initial_guess: Optional[np.ndarray] = None,
        max_iter: int = 25,
        tol: float = 5e-4,
        damping: float = 1e-2,
        step_clip: float = 0.25,
    ) -> np.ndarray:
        """Damped least-squares IK restricted to one G1 leg.

        Solves for the 6 actuated joint angles of the requested leg such that
        the corresponding foot site reaches ``foot_pos_pelvis`` (a position
        in the **pelvis frame**, with the pelvis pinned at the world origin
        in identity orientation). All other actuated joints are held at the
        spec's stand pose.

        This is the leg-only specialization of :meth:`inverse_kinematics`;
        targeting the foot via the leg subchain converges in <25 iterations
        for any reachable target and is fast enough for 200 Hz control.

        Args:
            side: ``"left"`` or ``"right"``.
            foot_pos_pelvis: Desired foot position in the pelvis frame, shape ``(3,)``.
            initial_guess: Optional starting guess for the 6 leg angles. Defaults
                to the leg slice of ``spec.stand_ctrl``.
            max_iter, tol, damping, step_clip: Newton iteration controls.

        Returns:
            6-vector of joint angles ordered like ``spec.left_leg_joints`` /
            ``spec.right_leg_joints``.
        """
        if side == "left":
            leg_joints = self.spec.left_leg_joints
            foot_site = "left_foot"
        elif side == "right":
            leg_joints = self.spec.right_leg_joints
            foot_site = "right_foot"
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        leg_actuated_idx = np.fromiter(
            (self.spec.actuated_joints.index(j) for j in leg_joints),
            dtype=np.int64,
            count=len(leg_joints),
        )
        leg_dof_idx = np.fromiter(
            (self.spec.joint_dof_index[j] for j in leg_joints),
            dtype=np.int64,
            count=len(leg_joints),
        )

        q_full = self.spec.stand_ctrl.copy()
        if initial_guess is not None:
            q_full[leg_actuated_idx] = np.asarray(initial_guess, dtype=np.float64).reshape(-1)

        target = np.asarray(foot_pos_pelvis, dtype=np.float64).reshape(3)
        site_id = int(self.spec.site_id[foot_site])

        for _ in range(max_iter):
            self._write_state(q_full, base_pos=None, base_quat_wxyz=None, qvel=None)
            self._mujoco.mj_kinematics(self._model, self._data)
            self._mujoco.mj_comPos(self._model, self._data)

            foot_world = np.asarray(self._data.site_xpos[site_id], dtype=np.float64)
            err = target - foot_world  # pelvis is pinned at origin → world == pelvis frame
            if float(np.linalg.norm(err)) < tol:
                break

            jacp = np.zeros((3, self.nv), dtype=np.float64)
            self._mujoco.mj_jacSite(self._model, self._data, jacp, None, site_id)
            J_leg = jacp[:, leg_dof_idx]  # (3, 6)

            JJT = J_leg @ J_leg.T + (damping ** 2) * np.eye(3, dtype=np.float64)
            dq = J_leg.T @ np.linalg.solve(JJT, err)
            dq = np.clip(dq, -step_clip, step_clip)
            q_full[leg_actuated_idx] = q_full[leg_actuated_idx] + dq

            # Joint limit clip on the leg subset only.
            for i, name in enumerate(leg_joints):
                lo, hi = self.spec.joint_range[name]
                q_full[leg_actuated_idx[i]] = float(np.clip(q_full[leg_actuated_idx[i]], lo, hi))

        return q_full[leg_actuated_idx].copy()

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.empty(self.spec.num_actuated, dtype=np.float64)
        upper = np.empty(self.spec.num_actuated, dtype=np.float64)
        for i, name in enumerate(self.spec.actuated_joints):
            lo, hi = self.spec.joint_range[name]
            lower[i] = lo
            upper[i] = hi
        return lower, upper

    def get_velocity_limits(self) -> np.ndarray:
        # G1 MJCF does not declare per-joint velocity limits; expose +inf and
        # let downstream callers (controllers / safety filters) impose their
        # own bounds. Override here when concrete numbers become available.
        return np.full(self.spec.num_actuated, np.inf, dtype=np.float64)

    def get_torque_limits(self) -> np.ndarray:
        return self.spec.torque_limit_vector()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _body_id(self, link_name: Optional[str]) -> int:
        name = link_name or self.default_link
        bid = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"G1 body not found: {name}")
        return int(bid)

    def _collect_body_names(self) -> list[str]:
        names = []
        for bid in range(self._model.nbody):
            adr = int(self._model.name_bodyadr[bid])
            end = self._model.names.find(b"\x00", adr)
            names.append(self._model.names[adr:end].decode())
        return names

    def _write_state(
        self,
        q_actuated: np.ndarray,
        base_pos: Optional[np.ndarray],
        base_quat_wxyz: Optional[np.ndarray],
        qvel: Optional[np.ndarray],
    ) -> None:
        q_actuated = np.asarray(q_actuated, dtype=np.float64).reshape(-1)
        if q_actuated.size != self.spec.num_actuated:
            raise ValueError(
                f"Expected {self.spec.num_actuated} actuated joint positions, got {q_actuated.size}"
            )

        qpos = np.zeros(self.nq, dtype=np.float64)
        if self.nq >= 7:
            qpos[0:3] = base_pos if base_pos is not None else 0.0
            qpos[3:7] = base_quat_wxyz if base_quat_wxyz is not None else _IDENTITY_QUAT_WXYZ
        qpos[self._actuated_qpos_idx] = q_actuated
        self._data.qpos[:] = qpos

        qv = np.zeros(self.nv, dtype=np.float64)
        if qvel is not None:
            qv = self._broadcast_actuated_qvel(qvel)
        self._data.qvel[:] = qv

    def _broadcast_actuated_qvel(self, q_actuated: np.ndarray) -> np.ndarray:
        full = np.zeros(self.nv, dtype=np.float64)
        v = np.asarray(q_actuated, dtype=np.float64).reshape(-1)
        if v.size == self.nv:
            return v.astype(np.float64, copy=True)
        if v.size != self.spec.num_actuated:
            raise ValueError(
                f"Expected {self.spec.num_actuated} or {self.nv} entries, got {v.size}"
            )
        full[self._actuated_dof_idx] = v
        return full


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (kept private to avoid leaking utility soup)
# ---------------------------------------------------------------------------


def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Axis-angle orientation error (target ⊖ current) in the world frame."""
    R_err = R_target @ R_current.T
    # Skew of the antisymmetric part gives 2·sin(theta)·axis.
    skew = 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=np.float64,
    )
    return skew
