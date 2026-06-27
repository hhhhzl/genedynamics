"""
MJX (MuJoCo XLA) adapter for full 3D quadrotor environment.

Uses mujoco-mjx for JAX-based, GPU-accelerated physics simulation.
Same interface as DroneFull3DMujocoEnv but with MJX backend.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import tempfile
import os

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

MJX_AVAILABLE = False
MjxPhysicsBackend = None
try:
    from genedynamics.core.backends.adapters.mjx_adapter import MjxPhysicsBackend
    MJX_AVAILABLE = True
except ImportError:
    pass

from genedynamics.envs.domains.drone.physics import DroneFull3DPhysicsEnv
from genedynamics.envs.utils.state_converter import (
    state_12d_to_mujoco,
    mujoco_to_state_12d,
)
from genedynamics.envs.utils.mujoco_model_generator import (
    generate_mujoco_xml_with_obstacles,
    create_base_quadrotor_xml,
)
from genedynamics.envs.obstacles.base import ObstacleManager

Array = np.ndarray


@dataclass
class DroneFull3DMjxEnv(DroneFull3DPhysicsEnv):
    """
    Full 3D quadrotor environment with MJX (MuJoCo XLA) physics backend.

    Uses JAX-based MuJoCo for GPU/TPU simulation. Same interface as
    DroneFull3DMujocoEnv. Supports batch rollout via physics backend.
    """

    model_path: Optional[str] = None
    use_mjx_physics: bool = True
    obstacles: Optional[ObstacleManager] = None

    def __post_init__(self):
        super().__post_init__()
        self.physics_backend = "mjx"
        self.use_physics_backend = self.use_mjx_physics

        if self.use_mjx_physics and MJX_AVAILABLE and MjxPhysicsBackend is not None:
            if self.obstacles is not None and len(self.obstacles) > 0:
                if self.model_path is None:
                    temp_base = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".xml", delete=False
                    )
                    create_base_quadrotor_xml(
                        temp_base.name,
                        mass=self.mass,
                        arm_length=self.arm_length,
                    )
                    base_path = temp_base.name
                    temp_base.close()
                else:
                    base_path = self.model_path
                temp_xml = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".xml", delete=False
                )
                temp_xml.close()
                generate_mujoco_xml_with_obstacles(
                    base_path, self.obstacles, output_path=temp_xml.name
                )
                actual_model_path = temp_xml.name
                self._temp_model_path = temp_xml.name
                if self.model_path is None:
                    os.unlink(base_path)
            else:
                if self.model_path is None:
                    temp_xml = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".xml", delete=False
                    )
                    temp_xml.close()
                    create_base_quadrotor_xml(
                        temp_xml.name,
                        mass=self.mass,
                        arm_length=self.arm_length,
                    )
                    actual_model_path = temp_xml.name
                    self._temp_model_path = temp_xml.name
                else:
                    actual_model_path = self.model_path
                    self._temp_model_path = None

            self._physics_backend_instance = MjxPhysicsBackend(
                model_path=actual_model_path,
                dt=self.dt,
            )
        else:
            self._physics_backend_instance = None
            self._temp_model_path = None

    def _state_to_mujoco(self, state: Array) -> Dict[str, np.ndarray]:
        return state_12d_to_mujoco(state)

    def _mujoco_to_state(self, mujoco_state: Dict[str, np.ndarray]) -> Array:
        return mujoco_to_state_12d(mujoco_state)

    def transition(self, state: Array, action: Array) -> Array:
        if not self.use_mjx_physics or self._physics_backend_instance is None:
            return super().transition(state, action)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        # MBD/2go output [-limit, limit]; map to [0, limit] for thrust
        motor_thrusts = (np.clip(action, -self.control_limit, self.control_limit) + self.control_limit) / 2.0
        mujoco_state = self._state_to_mujoco(state)
        self._physics_backend_instance.set_state(mujoco_state)
        next_mujoco_state = self._physics_backend_instance.step(
            motor_thrusts.astype(np.float64)
        )
        next_state = self._mujoco_to_state(next_mujoco_state)
        return self._project_state(next_state)

    def model_transition(self, state: Array, action: Array) -> Array:
        if not self.use_mjx_physics or self._physics_backend_instance is None:
            return super().model_transition(state, action)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        motor_thrusts = (np.clip(action, -self.control_limit, self.control_limit) + self.control_limit) / 2.0
        mujoco_state = self._state_to_mujoco(state)
        self._physics_backend_instance.set_state(mujoco_state)
        next_mujoco_state = self._physics_backend_instance.step(
            motor_thrusts.astype(np.float64)
        )
        return self._mujoco_to_state(next_mujoco_state)

    def close(self):
        if self._physics_backend_instance is not None:
            self._physics_backend_instance.close()
        if self._temp_model_path and os.path.exists(self._temp_model_path):
            os.unlink(self._temp_model_path)
            self._temp_model_path = None
        super().close()
