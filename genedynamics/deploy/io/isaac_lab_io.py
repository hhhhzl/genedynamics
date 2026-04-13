"""Isaac Lab robot IO — torch-runtime GPU simulation backend.

Wraps an Isaac Lab ``ManagerBasedRLEnv`` as a :class:`RobotIO` with
``physics_backend="isaac_lab"`` and ``array_runtime="torch"``. This is the
deploy layer's first torch-runtime IO, unlocking native pairing with
:class:`UnitreeRLGymG1Controller` and any future torch-side WBC — no
cross-runtime conversion overhead.

**Lazy import:** Everything under ``omni.isaac.lab.*`` is imported inside
``__init__`` so that importing this module costs nothing on a machine without
Isaac Lab. The registries module catches the resulting ``ImportError``
gracefully.

**Single-env scope:** ``num_envs=1`` for deploy. Batched multi-env rollouts
are a training concern and are out of scope for the deploy layer.

Dependencies:
    * Isaac Lab (via NVIDIA's Docker image) — optional.
    * Torch >= 2.1 — already required by ``UnitreeRLGymG1Controller``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO

__all__ = ["IsaacLabRobotIO"]

logger = logging.getLogger(__name__)


class IsaacLabRobotIO(BaseRobotIO):
    """Isaac Lab environment wrapped as a deploy-pipeline :class:`RobotIO`.

    Args:
        task_name: Isaac Lab task / environment name (e.g.
            ``"Isaac-Velocity-Flat-G1-v0"``). Defaults to a generic
            stand task.
        device: Torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
        sim_dt: Used for ``state.t`` bookkeeping. Isaac Lab manages its own
            internal physics dt via the task config.
        headless: Run Isaac Sim without a GUI window. Defaults to ``True``.
        spec: Optional pre-built :class:`G1RobotSpec`. When ``None``, a
            minimal spec is synthesized from the environment's articulation
            metadata.
        sync_host_shadow: When ``True`` (default), maintain numpy copies of
            qpos / qvel on CPU so numpy controllers paired through the
            Phase 16 bridge pay only one GPU→CPU transfer per step.

    Raises:
        ImportError: If ``omni.isaac.lab`` is not installed.
    """

    physics_backend = "isaac_lab"
    array_runtime = "torch"
    accepts: Tuple[str, ...] = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        *,
        task_name: str = "Isaac-Velocity-Flat-G1-v0",
        device: str = "cuda:0",
        sim_dt: float = 1.0 / 200.0,
        headless: bool = True,
        spec: Optional[Any] = None,
        sync_host_shadow: bool = True,
    ) -> None:
        import torch

        self._torch = torch

        self.task_name = task_name
        self.device = device
        self.sim_dt = float(sim_dt)
        self.headless = headless
        self.sync_host_shadow = sync_host_shadow

        # Lazy-import Isaac Lab.
        self._env, self._simulation_app = self._create_env(task_name, device, headless)

        # Shadow arrays for host-side consumers.
        self._shadow_qpos: Optional[np.ndarray] = None
        self._shadow_qvel: Optional[np.ndarray] = None

        # Latched action tensor.
        self._latched_action: Optional[Any] = None

        # Joint ordering permutation (populated on first reset if needed).
        self._joint_permutation: Optional[Any] = None

        # Build spec.
        if spec is None:
            spec = self._build_minimal_spec()

        super().__init__(spec=spec)

        logger.info(
            "IsaacLabRobotIO: task=%s device=%s sim_dt=%.4f headless=%s",
            task_name, device, sim_dt, headless,
        )

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_env(
        task_name: str,
        device: str,
        headless: bool,
    ) -> tuple:
        """Create an Isaac Lab ManagerBasedRLEnv with num_envs=1.

        Returns:
            (env, simulation_app) tuple.
        """
        from omni.isaac.lab.app import AppLauncher

        launcher = AppLauncher(headless=headless)
        simulation_app = launcher.app

        import gymnasium as gym
        import omni.isaac.lab_tasks  # noqa: F401 — registers Isaac Lab tasks

        env = gym.make(task_name, num_envs=1, device=device)
        return env, simulation_app

    # ------------------------------------------------------------------
    # Minimal spec builder
    # ------------------------------------------------------------------

    def _build_minimal_spec(self) -> Any:
        """Synthesize a minimal spec from the Isaac Lab environment."""
        from types import SimpleNamespace

        try:
            articulation = self._env.unwrapped.scene.articulations["robot"]
            joint_names = articulation.data.joint_names
            num_actuated = len(joint_names)
            nq = articulation.data.joint_pos.shape[-1] + 7  # + floating base
            nv = articulation.data.joint_vel.shape[-1] + 6
        except (AttributeError, KeyError):
            # Fallback for envs with different structure.
            obs_size = self._env.observation_space.shape[-1] if hasattr(self._env, "observation_space") else 48
            num_actuated = max(obs_size // 3, 1)
            nq = 7 + num_actuated
            nv = 6 + num_actuated
            joint_names = [f"joint_{i}" for i in range(num_actuated)]

        spec = SimpleNamespace(
            num_actuated=num_actuated,
            nq=nq,
            nv=nv,
            actuated_joints=tuple(joint_names),
            actuated_qpos_indices=np.arange(7, 7 + num_actuated, dtype=np.int32),
            actuated_dof_indices=np.arange(6, 6 + num_actuated, dtype=np.int32),
            joint_range={name: np.array([-3.14, 3.14]) for name in joint_names},
            torque_limit={name: 100.0 for name in joint_names},
        )
        return spec

    # ------------------------------------------------------------------
    # Joint ordering
    # ------------------------------------------------------------------

    def _maybe_build_permutation(self) -> None:
        """Build a permutation index if the env's joint order disagrees with spec."""
        if self._joint_permutation is not None:
            return
        try:
            articulation = self._env.unwrapped.scene.articulations["robot"]
            env_joint_names = list(articulation.data.joint_names)
            spec_joint_names = list(self.spec.actuated_joints)

            if env_joint_names == spec_joint_names:
                self._joint_permutation = None
                return

            # Build permutation: spec order → env order.
            perm = []
            missing = []
            for name in spec_joint_names:
                if name in env_joint_names:
                    perm.append(env_joint_names.index(name))
                else:
                    missing.append(name)
            if missing:
                raise RuntimeError(
                    f"IsaacLabRobotIO: {len(missing)} joint(s) in spec not "
                    f"found in the Isaac Lab environment: {missing}. "
                    f"Env joints: {env_joint_names}. "
                    f"Spec joints: {spec_joint_names}. "
                    f"Fix the spec or use a matching environment task."
                )
            self._joint_permutation = self._torch.tensor(
                perm, dtype=self._torch.long, device=self.device
            )
            logger.info("IsaacLabRobotIO: built joint permutation index")
        except (AttributeError, KeyError):
            self._joint_permutation = None

    # ------------------------------------------------------------------
    # BaseRobotIO hooks
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        obs, _ = self._env.reset()
        self._last_obs = obs
        self._latched_action = None
        self._maybe_build_permutation()
        self._sync_shadow()

    def _read_state(self, t: float) -> RobotState:
        torch = self._torch

        try:
            articulation = self._env.unwrapped.scene.articulations["robot"]
            root_state = articulation.data.root_state_w[0]  # (13,): pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
            joint_pos = articulation.data.joint_pos[0]
            joint_vel = articulation.data.joint_vel[0]

            base_pos = root_state[:3]
            base_quat = root_state[3:7]
            base_lin_vel = root_state[7:10]
            base_ang_vel = root_state[10:13]

            qpos = torch.cat([base_pos, base_quat, joint_pos])
            qvel = torch.cat([base_lin_vel, base_ang_vel, joint_vel])
            base_pose = torch.cat([base_pos, base_quat])
            base_twist = torch.cat([base_lin_vel, base_ang_vel])
        except (AttributeError, KeyError):
            # Fallback: construct from observation.
            obs = self._last_obs
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", obs.get("obs", list(obs.values())[0]))
            else:
                obs_tensor = obs
            obs_tensor = obs_tensor[0] if obs_tensor.dim() > 1 else obs_tensor
            n = obs_tensor.shape[0]
            qpos = obs_tensor[:min(n, self.spec.nq)]
            qvel = obs_tensor[min(n, self.spec.nq):min(n, self.spec.nq + self.spec.nv)]
            base_pose = qpos[:7] if qpos.shape[0] >= 7 else None
            base_twist = qvel[:6] if qvel.shape[0] >= 6 else None

        return RobotState(
            t=float(t),
            qpos=qpos,
            qvel=qvel,
            base_pose=base_pose,
            base_twist=base_twist,
            extras={"step": self._step_count},
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        torch = self._torch

        if cmd.kind in ("joint_pos", "mixed") and cmd.joint_pos is not None:
            action = cmd.joint_pos
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            if self._joint_permutation is not None:
                action = action[self._joint_permutation]
            self._latched_action = action.unsqueeze(0)  # (1, nu)
        elif cmd.kind == "torque" and cmd.joint_torque is not None:
            action = cmd.joint_torque
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)
            if self._joint_permutation is not None:
                action = action[self._joint_permutation]
            self._latched_action = action.unsqueeze(0)
        elif cmd.kind == "torque" and cmd.joint_torque is None:
            raise ValueError("ControlCommand(kind='torque') requires joint_torque")
        else:
            action_dim = self._env.action_space.shape[-1]
            self._latched_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=self.device
            )

    def _step_physics(self, dt: float) -> None:
        torch = self._torch

        if self._latched_action is None:
            action_dim = self._env.action_space.shape[-1]
            action = torch.zeros(1, action_dim, dtype=torch.float32, device=self.device)
        else:
            action = self._latched_action

        obs, reward, terminated, truncated, extras = self._env.step(action)
        self._last_obs = obs
        self._sync_shadow()

    def _on_close(self) -> None:
        try:
            self._env.close()
        except Exception as exc:
            logger.debug("IsaacLabRobotIO: env.close() raised: %s", exc)
        try:
            self._simulation_app.close()
        except Exception as exc:
            logger.debug("IsaacLabRobotIO: simulation_app.close() raised: %s", exc)
        self._shadow_qpos = None
        self._shadow_qvel = None
        logger.info("IsaacLabRobotIO: closed")

    # ------------------------------------------------------------------
    # Shadow sync
    # ------------------------------------------------------------------

    def _sync_shadow(self) -> None:
        """Pull qpos/qvel to host numpy arrays if shadow sync is enabled."""
        if not self.sync_host_shadow:
            return
        try:
            articulation = self._env.unwrapped.scene.articulations["robot"]
            joint_pos = articulation.data.joint_pos[0]
            joint_vel = articulation.data.joint_vel[0]
            self._shadow_qpos = joint_pos.detach().cpu().numpy().astype(np.float64)
            self._shadow_qvel = joint_vel.detach().cpu().numpy().astype(np.float64)
        except (AttributeError, KeyError):
            self._shadow_qpos = None
            self._shadow_qvel = None

    @property
    def shadow_qpos(self) -> Optional[np.ndarray]:
        """Host-side numpy copy of joint positions, or ``None``."""
        return self._shadow_qpos

    @property
    def shadow_qvel(self) -> Optional[np.ndarray]:
        """Host-side numpy copy of joint velocities, or ``None``."""
        return self._shadow_qvel
