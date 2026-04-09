"""G1 sport-mode RL loco client — loads ``g1_motion.pt`` and walks G1.

The bundled policy (``assets/g1_motion.pt``, 145 kB) is a TorchScript-saved
PPO policy trained on the standard Unitree RL Gym G1 walking task.  This
client adapts it to our :class:`LocoClient` protocol.

The policy builds a 47-dim observation vector and is queried at 50 Hz.
The advantage of fitting it under the existing :class:`SportModeController`
(rather than as a top-level controller) is that the controller's upper-body
PD merge logic (``HumanoidTaskSpec.joint_hints``) keeps working unchanged —
only the 12 leg joints are produced by the RL policy.

Observation layout (frozen by the trained policy)::

    [ omega(3),                              # base angular velocity
      gravity_in_base(3),                    # gravity vector projected into base frame
      cmd * cmd_scale(3),                    # (vx, vy, yaw_rate) * [2, 2, 0.25]
      (qj - default)(12),                    # leg joint pos relative to default
      qj_dot * dof_vel_scale(12),            # leg joint vel * 0.05
      last_action(12),                       # action emitted last tick
      sin(phase), cos(phase)(2),             # period = 0.8 s clock
    ]                                        # = 47

Action mapping (output → leg joint targets)::

    q_target_leg = default_pos_lower_body + action * action_scale  # action_scale = 0.25

The leg joint order matches the canonical
:attr:`G1RobotSpec.left_leg_joints + right_leg_joints`, so no permutation
is needed.

The policy file is loaded lazily on first ``reset()`` so the module imports
without ``torch`` installed.  The default path is
``assets/g1_motion.pt`` (bundled next to this file); override via the
``policy_path`` constructor arg or the ``GENEDYNAMICS_G1_MOTION_PT``
environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import LocoCommand, RobotState
from genedynamics.envs.robots.g1 import G1RobotModel
from genedynamics.robots.g1.spec import (
    G1_LEFT_LEG_JOINTS,
    G1_RIGHT_LEG_JOINTS,
)

__all__ = ["SparkRLLocoClient", "DEFAULT_G1_MOTION_PT"]

# ---------------------------------------------------------------------------
# Default policy path — bundled next to this file in assets/.
# Override via env var GENEDYNAMICS_G1_MOTION_PT or the policy_path= arg.
# ---------------------------------------------------------------------------

#: Bundled TorchScript policy (145 kB). Resolve relative to this file so the
#: path stays correct regardless of the working directory.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_G1_MOTION_PT: Path = Path(
    os.environ.get("GENEDYNAMICS_G1_MOTION_PT", str(_ASSETS_DIR / "g1_motion.pt"))
)

# Back-compat alias (remove once all callers are updated).
DEFAULT_SPARK_MOTION_PT = DEFAULT_G1_MOTION_PT


# Frozen training-time constants from
# spark/module/spark_robot/spark_robot/g1/config/g1_sport_mode_dynamic_1_config.py
# and spark/.../g1_mujoco_sport_mode_agent.py:
DEFAULT_POS_LOWER_BODY = np.array(
    [
        # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        # right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    ],
    dtype=np.float64,
)

CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float64)
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.25
ACTION_SCALE = 0.25
PHASE_PERIOD_S = 0.8
NUM_LEG_ACTIONS = 12
OBS_DIM = 9 + 3 * NUM_LEG_ACTIONS + 2  # = 47


class SparkRLLocoClient:
    """LocoClient that delegates leg control to spark's pretrained RL policy.

    Args:
        robot: A :class:`G1RobotModel`. Used solely for the spec — the
            client never asks the robot to do FK/IK; the policy emits joint
            angles directly.
        policy_path: Optional override for the ``.pt`` file. Defaults to
            ``assets/g1_motion.pt`` (bundled next to this file), or the
            ``GENEDYNAMICS_G1_MOTION_PT`` env var if set.
        nominal_step_period: Phase clock period in seconds. ``0.8`` matches
            the value baked into the policy at training time.
        cmd_clip: Per-channel clamp on the velocity command before scaling
            (matches spark's ``np.clip(sport_cmd, -0.3, 0.3)``).
    """

    nominal_step_period: float

    def __init__(
        self,
        robot: G1RobotModel,
        *,
        policy_path: Optional[str | Path] = None,
        nominal_step_period: float = PHASE_PERIOD_S,
        cmd_clip: float = 0.3,
    ) -> None:
        self.robot = robot
        self.spec = robot.spec
        self.policy_path = Path(policy_path) if policy_path is not None else DEFAULT_G1_MOTION_PT
        self.nominal_step_period = float(nominal_step_period)
        self.cmd_clip = float(cmd_clip)

        # Pre-compute joint name → qpos / dof index for the 12 leg joints.
        self._leg_names: tuple[str, ...] = tuple(G1_LEFT_LEG_JOINTS) + tuple(G1_RIGHT_LEG_JOINTS)
        self._leg_qpos_idx = np.array(
            [self.spec.joint_qpos_index[name] for name in self._leg_names],
            dtype=np.int64,
        )
        self._leg_dof_idx = np.array(
            [self.spec.joint_dof_index[name] for name in self._leg_names],
            dtype=np.int64,
        )

        # Lazy-loaded policy + last-action memory.
        self._policy = None
        self._torch = None
        self._last_action = np.zeros(NUM_LEG_ACTIONS, dtype=np.float64)
        self._t_phase: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # LocoClient protocol
    # ------------------------------------------------------------------

    def reset(self, pelvis_world: np.ndarray, pelvis_yaw: float = 0.0) -> None:
        # pelvis_world / pelvis_yaw are accepted for protocol symmetry.
        # The policy lives in joint-space; world-frame anchors don't apply.
        del pelvis_world, pelvis_yaw

        if self._policy is None:
            self._load_policy()
        self._last_action[:] = 0.0
        self._t_phase = 0.0
        self._initialized = True

    def step(
        self,
        cmd: LocoCommand,
        dt: float,
        pelvis_world: np.ndarray,
        pelvis_yaw: float,
        state: Optional[RobotState] = None,
    ) -> Mapping[str, float]:
        del pelvis_world, pelvis_yaw  # the policy reads everything from state
        if not self._initialized:
            raise RuntimeError("SparkRLLocoClient.step called before reset")
        if state is None:
            raise RuntimeError(
                "SparkRLLocoClient requires the RobotState — make sure the "
                "SportModeController is up to date and forwards `state=` to "
                "loco_client.step()."
            )
        if state.qpos is None or state.qvel is None:
            raise RuntimeError("SparkRLLocoClient requires state.qpos / state.qvel")

        # 1. Pull the 12 leg joint positions and velocities from state.
        qpos = np.asarray(state.qpos, dtype=np.float64)
        qvel = np.asarray(state.qvel, dtype=np.float64)
        qj = qpos[self._leg_qpos_idx]
        dqj = qvel[self._leg_dof_idx]

        # 2. Base angular velocity (omega) — qvel[3:6] for the floating base.
        omega = qvel[3:6].copy()

        # 3. Gravity vector projected into the base frame.
        if qpos.size >= 7:
            quat_wxyz = qpos[3:7]
            gravity_in_base = _gravity_in_base(quat_wxyz)
        else:
            gravity_in_base = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # 4. Sport command — clip then scale, exactly like spark.
        sport_cmd = np.array([cmd.vx, cmd.vy, cmd.yaw_rate], dtype=np.float64)
        sport_cmd = np.clip(sport_cmd, -self.cmd_clip, +self.cmd_clip)

        # 5. Phase clock — wall-time based.
        self._t_phase += float(dt)
        phase = (self._t_phase % self.nominal_step_period) / self.nominal_step_period
        sin_phase = float(np.sin(2.0 * np.pi * phase))
        cos_phase = float(np.cos(2.0 * np.pi * phase))

        # 6. Build the 47-dim observation in spark's exact layout.
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:3] = (omega * ANG_VEL_SCALE).astype(np.float32)
        obs[3:6] = gravity_in_base.astype(np.float32)
        obs[6:9] = (sport_cmd * CMD_SCALE).astype(np.float32)
        obs[9 : 9 + NUM_LEG_ACTIONS] = (
            (qj - DEFAULT_POS_LOWER_BODY) * DOF_POS_SCALE
        ).astype(np.float32)
        obs[9 + NUM_LEG_ACTIONS : 9 + 2 * NUM_LEG_ACTIONS] = (
            dqj * DOF_VEL_SCALE
        ).astype(np.float32)
        obs[9 + 2 * NUM_LEG_ACTIONS : 9 + 3 * NUM_LEG_ACTIONS] = self._last_action.astype(
            np.float32
        )
        obs[-2:] = np.array([sin_phase, cos_phase], dtype=np.float32)

        # 7. Inference.
        torch = self._torch
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self._policy(obs_tensor).detach().cpu().numpy().squeeze(0)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != NUM_LEG_ACTIONS:
            raise RuntimeError(
                f"Spark policy emitted {action.size} actions, expected "
                f"{NUM_LEG_ACTIONS}. Wrong .pt file?"
            )
        self._last_action = action

        # 8. Map action → leg joint targets via the training-time formula.
        q_target = DEFAULT_POS_LOWER_BODY + action * ACTION_SCALE

        return {name: float(q_target[i]) for i, name in enumerate(self._leg_names)}

    @property
    def swing_foot(self) -> Optional[str]:
        """The RL policy doesn't expose its gait phase — return ``None``."""
        return None

    @property
    def phase(self) -> float:
        if self.nominal_step_period <= 0:
            return 0.0
        return float((self._t_phase % self.nominal_step_period) / self.nominal_step_period)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_policy(self) -> None:
        """Lazy import torch and load the TorchScript policy."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SparkRLLocoClient requires PyTorch. Install via:\n"
                "    pip install torch\n"
                "or run inside the genedynamics/dev-cpu Docker image."
            ) from exc
        if not self.policy_path.exists():
            raise FileNotFoundError(
                f"G1 sport-mode policy not found at {self.policy_path}.\n"
                "Options:\n"
                "  1. Set env var GENEDYNAMICS_G1_MOTION_PT=/path/to/g1_motion.pt\n"
                "  2. Pass policy_path= to SparkRLLocoClient()\n"
                "  3. Place g1_motion.pt in "
                f"{_ASSETS_DIR}"
            )
        self._torch = torch
        self._policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self._policy.eval()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _gravity_in_base(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project the world gravity unit vector into the body (base) frame.

    Mirrors spark's :meth:`_send_control_sport_mode` formula. ``quat_wxyz`` is
    in MuJoCo order ``(w, x, y, z)``. The result is the third column of
    ``R_world_to_base`` (i.e. the body-frame components of the world ``-z``
    direction normalized to length 1).
    """
    qw, qx, qy, qz = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float64,
    )
