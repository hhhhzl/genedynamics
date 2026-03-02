"""
Drone (quadrotor) model implementations.

This module provides quadrotor models with dynamics computation.
Supports different drone configurations and integration with gym-pybullet-drones.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from genedynamics.envs.robots.base import RobotModel, RobotModelMixin


class DroneModel(RobotModelMixin):
    """
    Quadrotor (drone) model.
    
    Implements quadrotor dynamics with:
    - 6 DOF: 3D position + orientation (roll, pitch, yaw)
    - 4 control inputs: motor thrusts
    - Physics-based dynamics
    """
    
    def __init__(
        self,
        mass: float = 0.5,
        Ixx: float = 0.0023,
        Iyy: float = 0.0023,
        Izz: float = 0.0046,
        arm_length: float = 0.17,
        kf: float = 3.16e-10,  # Thrust coefficient
        km: float = 7.94e-12,  # Moment coefficient
        gravity: float = 9.81,
        **kwargs
    ):
        """
        Initialize quadrotor model.
        
        Args:
            mass: Drone mass (kg)
            Ixx, Iyy, Izz: Moments of inertia (kg*m^2)
            arm_length: Distance from center to motor (m)
            kf: Thrust coefficient
            km: Moment coefficient
            gravity: Gravity acceleration (m/s^2)
            **kwargs: Additional parameters
        """
        # Quadrotor has 6 DOF (position + orientation)
        super().__init__(n_dof=6)
        
        self.mass = float(mass)
        self.I = np.array([Ixx, Iyy, Izz], dtype=np.float32)
        self.I_inv = 1.0 / self.I
        self.arm_length = float(arm_length)
        self.kf = float(kf)
        self.km = float(km)
        self.gravity = float(gravity)
        
        # Control input: 4 motor thrusts
        self.n_actuators = 4
        
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        # 12-dimensional state space
        self.state_dim = 12
    
    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics for quadrotor.
        
        For quadrotor, "joint positions" are actually the state [x, y, z, roll, pitch, yaw].
        
        Args:
            joint_positions: State [x, y, z, roll, pitch, yaw] or full state
            link_name: Ignored (quadrotor has single body)
            
        Returns:
            Tuple of (position, orientation)
        """
        state = np.asarray(joint_positions, dtype=np.float32)
        
        if len(state) >= 6:
            # Extract position and orientation
            position = state[:3]
            if len(state) >= 6:
                orientation = state[3:6]  # Roll, pitch, yaw (Euler angles)
            else:
                orientation = np.zeros(3, dtype=np.float32)
        else:
            position = state[:3] if len(state) >= 3 else np.zeros(3, dtype=np.float32)
            orientation = np.zeros(3, dtype=np.float32)
        
        # Convert Euler angles to quaternion
        quaternion = self._euler_to_quaternion(orientation)
        
        return position, quaternion
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z], dtype=np.float32)
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Inverse kinematics for quadrotor.
        
        For quadrotor, this is trivial: position is directly the state.
        Orientation can be set if provided.
        
        Args:
            target_position: Target 3D position, shape (3,)
            target_orientation: Target orientation (quaternion or Euler), optional
            initial_guess: Ignored
            **kwargs: Additional parameters
            
        Returns:
            State [x, y, z, roll, pitch, yaw]
        """
        position = np.asarray(target_position, dtype=np.float32)
        
        if target_orientation is not None:
            orientation = np.asarray(target_orientation, dtype=np.float32)
            if len(orientation) == 4:
                # Quaternion, convert to Euler
                euler = self._quaternion_to_euler(orientation)
            else:
                euler = orientation[:3]
        else:
            euler = np.zeros(3, dtype=np.float32)
        
        return np.concatenate([position, euler]).astype(np.float32)
    
    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def jacobian(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute Jacobian for quadrotor.
        
        For quadrotor, Jacobian relates body velocity to world velocity.
        This is the rotation matrix from body to world frame.
        
        Args:
            joint_positions: State [x, y, z, roll, pitch, yaw]
            link_name: Ignored
            
        Returns:
            Jacobian matrix, shape (6, 6) for spatial velocity
        """
        state = np.asarray(joint_positions, dtype=np.float32)
        
        if len(state) >= 6:
            roll, pitch, yaw = state[3], state[4], state[5]
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        
        # Rotation matrix from body to world
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Spatial Jacobian: [R, 0; 0, R] for position and orientation
        J = np.zeros((6, 6), dtype=np.float32)
        J[:3, :3] = R
        J[3:, 3:] = R
        
        return J
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ], dtype=np.float32)
        
        return R
    
    def forward_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute forward dynamics for quadrotor.
        
        Note: For quadrotor, "joint_torques" are actually motor thrusts.
        This computes accelerations from current state and motor commands.
        
        Args:
            joint_positions: State [x, y, z, roll, pitch, yaw]
            joint_velocities: Velocities [vx, vy, vz, wx, wy, wz]
            joint_torques: Motor thrusts [T1, T2, T3, T4] (normalized 0-1)
            **kwargs: Additional parameters
            
        Returns:
            Accelerations [ax, ay, az, alphax, alphay, alphaz]
        """
        # This is a simplified version
        # Full implementation would use the quadrotor dynamics equations
        raise NotImplementedError(
            "Forward dynamics for quadrotor requires full state and motor model. "
            "Use step() method instead for simulation."
        )
    
    def inverse_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute inverse dynamics for quadrotor.
        
        Computes required motor thrusts to achieve desired accelerations.
        
        Args:
            joint_positions: State [x, y, z, roll, pitch, yaw]
            joint_velocities: Velocities [vx, vy, vz, wx, wy, wz]
            joint_accelerations: Desired accelerations [ax, ay, az, alphax, alphay, alphaz]
            **kwargs: Additional parameters
            
        Returns:
            Motor thrusts [T1, T2, T3, T4] (normalized 0-1)
        """
        # This requires solving the quadrotor control allocation problem
        raise NotImplementedError(
            "Inverse dynamics for quadrotor requires control allocation. "
            "Use a controller instead."
        )
    
    def step(
        self,
        state: np.ndarray,
        motor_thrusts: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        Step quadrotor dynamics forward.
        
        This is the main method for simulating quadrotor dynamics.
        
        Args:
            state: Full state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
            motor_thrusts: Motor thrusts [T1, T2, T3, T4] (normalized 0-1), shape (4,)
            dt: Time step
            
        Returns:
            Next state, shape (12,)
        """
        state = np.asarray(state, dtype=np.float32)
        motor_thrusts = np.asarray(motor_thrusts, dtype=np.float32)
        
        # Extract state components
        pos = state[:3]
        vel = state[3:6]
        euler = state[6:9]
        ang_vel = state[9:12]
        
        # Compute forces and moments from motor thrusts
        # Total thrust
        T_total = np.sum(motor_thrusts) * self.kf
        
        # Moments (simplified)
        # Assuming motors are at: front-left, front-right, back-right, back-left
        T1, T2, T3, T4 = motor_thrusts
        Mx = self.arm_length * self.kf * (T2 - T4)  # Roll moment
        My = self.arm_length * self.kf * (T1 - T3)  # Pitch moment
        Mz = self.km * (T1 - T2 + T3 - T4)  # Yaw moment
        
        # Linear acceleration (in world frame)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Thrust in body frame (upward)
        thrust_body = np.array([0, 0, T_total], dtype=np.float32)
        thrust_world = R @ thrust_body
        
        # Gravity
        gravity_vec = np.array([0, 0, -self.gravity], dtype=np.float32)
        
        # Linear acceleration
        accel = (thrust_world / self.mass) + gravity_vec
        
        # Angular acceleration
        moments = np.array([Mx, My, Mz], dtype=np.float32)
        ang_accel = self.I_inv * moments
        
        # Integrate
        new_vel = vel + accel * dt
        new_pos = pos + vel * dt + 0.5 * accel * dt ** 2
        
        new_ang_vel = ang_vel + ang_accel * dt
        
        # Integrate angular velocity to get new orientation
        # Simplified: use small angle approximation
        new_euler = euler + ang_vel * dt
        
        # Combine new state
        new_state = np.concatenate([
            new_pos,
            new_vel,
            new_euler,
            new_ang_vel
        ]).astype(np.float32)
        
        return new_state
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get position limits (typically unlimited for quadrotor)."""
        # Position limits (can be customized)
        lower = np.array([-np.inf, -np.inf, 0.0, -np.pi, -np.pi/2, -np.pi], dtype=np.float32)
        upper = np.array([np.inf, np.inf, np.inf, np.pi, np.pi/2, np.pi], dtype=np.float32)
        return lower, upper
    
    def get_velocity_limits(self) -> np.ndarray:
        """Get velocity limits."""
        # Typical quadrotor velocity limits
        return np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float32)  # m/s and rad/s
    
    def get_torque_limits(self) -> np.ndarray:
        """Get motor thrust limits (normalized 0-1)."""
        return np.ones(4, dtype=np.float32)  # Max thrust = 1.0
