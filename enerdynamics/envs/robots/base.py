"""
Base robot model interface.

This module defines the RobotModel protocol that all robot models should implement.
Robot models provide kinematics and dynamics computation for robots.
"""

from typing import Protocol, Optional, Tuple, Any, runtime_checkable
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


@runtime_checkable
class RobotModel(Protocol):
    """
    Protocol for robot model interface.
    
    Robot models provide:
    - Kinematics: forward/inverse kinematics, Jacobian computation
    - Dynamics: forward/inverse dynamics
    
    All robot models (manipulators, drones, etc.) should implement this interface.
    
    Attributes:
        n_dof: Number of degrees of freedom (int)
        n_joints: Number of joints (int)
        joint_names: List of joint names (List[str], optional)
        link_names: List of link names (List[str], optional)
    """
    
    n_dof: int
    n_joints: int
    joint_names: Optional[list] = None
    link_names: Optional[list] = None
    
    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics: joint positions -> end-effector pose.
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            link_name: Optional link name (if None, uses end-effector)
            
        Returns:
            Tuple of (position, orientation)
            - position: 3D position, shape (3,)
            - orientation: Orientation (quaternion or rotation matrix), shape (4,) or (3, 3)
        """
        ...
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute inverse kinematics: end-effector pose -> joint positions.
        
        Args:
            target_position: Target 3D position, shape (3,)
            target_orientation: Target orientation (quaternion or rotation matrix), optional
            initial_guess: Initial guess for joint positions, shape (n_dof,)
            **kwargs: Additional IK parameters (tolerance, max_iterations, etc.)
            
        Returns:
            Joint positions, shape (n_dof,)
        """
        ...
    
    def jacobian(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute Jacobian matrix: d(end_effector_velocity) / d(joint_velocity).
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            link_name: Optional link name (if None, uses end-effector)
            
        Returns:
            Jacobian matrix, shape (6, n_dof) for spatial Jacobian
            or shape (3, n_dof) for position-only Jacobian
        """
        ...
    
    def forward_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute forward dynamics: joint torques -> joint accelerations.
        
        M(q) * qdd + C(q, qd) * qd + G(q) = tau
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            joint_velocities: Joint velocities, shape (n_dof,)
            joint_torques: Joint torques, shape (n_dof,)
            **kwargs: Additional parameters (gravity, external forces, etc.)
            
        Returns:
            Joint accelerations, shape (n_dof,)
        """
        ...
    
    def inverse_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute inverse dynamics: joint accelerations -> joint torques.
        
        tau = M(q) * qdd + C(q, qd) * qd + G(q)
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            joint_velocities: Joint velocities, shape (n_dof,)
            joint_accelerations: Joint accelerations, shape (n_dof,)
            **kwargs: Additional parameters (gravity, external forces, etc.)
            
        Returns:
            Joint torques, shape (n_dof,)
        """
        ...
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint position limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits), each shape (n_dof,)
        """
        ...
    
    def get_velocity_limits(self) -> np.ndarray:
        """
        Get joint velocity limits.
        
        Returns:
            Velocity limits, shape (n_dof,)
        """
        ...
    
    def get_torque_limits(self) -> np.ndarray:
        """
        Get joint torque limits.
        
        Returns:
            Torque limits, shape (n_dof,)
        """
        ...


class RobotModelMixin:
    """
    Mixin class providing default implementations for RobotModel.
    
    Robot models can inherit from this to get default behavior,
    then override specific methods as needed.
    """
    
    def __init__(
        self,
        n_dof: int,
        joint_names: Optional[list] = None,
        link_names: Optional[list] = None,
    ):
        """
        Initialize robot model.
        
        Args:
            n_dof: Number of degrees of freedom
            joint_names: Optional list of joint names
            link_names: Optional list of link names
        """
        self.n_dof = n_dof
        self.n_joints = n_dof
        self.joint_names = joint_names or [f"joint_{i}" for i in range(n_dof)]
        self.link_names = link_names or [f"link_{i}" for i in range(n_dof + 1)]
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default joint limits (unlimited).
        
        Subclasses should override with actual limits.
        """
        lower = np.full(self.n_dof, -np.inf, dtype=np.float32)
        upper = np.full(self.n_dof, np.inf, dtype=np.float32)
        return lower, upper
    
    def get_velocity_limits(self) -> np.ndarray:
        """
        Default velocity limits (unlimited).
        
        Subclasses should override with actual limits.
        """
        return np.full(self.n_dof, np.inf, dtype=np.float32)
    
    def get_torque_limits(self) -> np.ndarray:
        """
        Default torque limits (unlimited).
        
        Subclasses should override with actual limits.
        """
        return np.full(self.n_dof, np.inf, dtype=np.float32)
