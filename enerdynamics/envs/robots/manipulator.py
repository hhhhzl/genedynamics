"""
Manipulator (robotic arm) model implementations.

This module provides manipulator models with kinematics and dynamics computation.
Supports loading from URDF files using PyBullet or Pinocchio.
"""

from typing import Optional, Tuple, Any
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from enerdynamics.envs.robots.base import RobotModel, RobotModelMixin


class ManipulatorModel(RobotModelMixin):
    """
    Manipulator (robotic arm) model.
    
    Supports loading from URDF files and computing:
    - Forward/inverse kinematics
    - Jacobian matrices
    - Forward/inverse dynamics
    """
    
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        n_dof: Optional[int] = None,
        backend: str = "pybullet",
        **kwargs
    ):
        """
        Initialize manipulator model.
        
        Args:
            urdf_path: Path to URDF file (optional, can set later)
            n_dof: Number of degrees of freedom (inferred from URDF if not provided)
            backend: Backend to use ("pybullet" or "pinocchio")
            **kwargs: Additional parameters
        """
        self.urdf_path = urdf_path
        self.backend = backend
        
        # Initialize backend
        if backend == "pybullet":
            self._init_pybullet(**kwargs)
        elif backend == "pinocchio":
            self._init_pinocchio(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'pybullet' or 'pinocchio'")
        
        # If n_dof not provided, infer from backend
        if n_dof is None:
            n_dof = self._get_n_dof()
        
        super().__init__(n_dof=n_dof)
        
        # Load URDF if provided
        if urdf_path:
            self.load_urdf(urdf_path)
    
    def _init_pybullet(self, **kwargs):
        """Initialize PyBullet backend."""
        try:
            import pybullet as p
            self.pb = p
            self.pb_client = kwargs.get('physicsClientId', None)
            if self.pb_client is None:
                # Create physics client if not provided
                self.pb_client = self.pb.connect(self.pb.DIRECT)  # Non-graphical
            self._pb_robot_id = None
        except ImportError:
            raise ImportError(
                "PyBullet backend requires pybullet. Install with: pip install pybullet"
            )
    
    def _init_pinocchio(self, **kwargs):
        """Initialize Pinocchio backend."""
        try:
            import pinocchio as pin
            self.pin = pin
            self.model = None
            self.data = None
        except ImportError:
            raise ImportError(
                "Pinocchio backend requires pinocchio. Install with: pip install pin"
            )
    
    def _get_n_dof(self) -> int:
        """Get number of DOF from backend."""
        if self.backend == "pybullet":
            if self._pb_robot_id is not None:
                # Get from loaded robot
                num_joints = self.pb.getNumJoints(self._pb_robot_id)
                # Count only movable joints
                n_dof = 0
                for i in range(num_joints):
                    joint_info = self.pb.getJointInfo(self._pb_robot_id, i)
                    if joint_info[2] != self.pb.JOINT_FIXED:
                        n_dof += 1
                return n_dof
            return 6  # Default
        elif self.backend == "pinocchio":
            if self.model is not None:
                return self.model.nv  # Number of velocity variables
            return 6  # Default
        return 6
    
    def load_urdf(self, urdf_path: str, **kwargs) -> None:
        """
        Load robot from URDF file.
        
        Args:
            urdf_path: Path to URDF file
            **kwargs: Additional loading parameters
        """
        self.urdf_path = urdf_path
        
        if self.backend == "pybullet":
            self._load_urdf_pybullet(urdf_path, **kwargs)
        elif self.backend == "pinocchio":
            self._load_urdf_pinocchio(urdf_path, **kwargs)
        
        # Update n_dof after loading
        self.n_dof = self._get_n_dof()
        self.n_joints = self.n_dof
    
    def _load_urdf_pybullet(self, urdf_path: str, **kwargs):
        """Load URDF using PyBullet."""
        flags = kwargs.get('flags', self.pb.URDF_USE_INERTIA_FROM_FILE)
        self._pb_robot_id = self.pb.loadURDF(
            urdf_path,
            basePosition=kwargs.get('basePosition', [0, 0, 0]),
            baseOrientation=kwargs.get('baseOrientation', [0, 0, 0, 1]),
            flags=flags,
            physicsClientId=self.pb_client
        )
        
        # Extract joint names and limits
        num_joints = self.pb.getNumJoints(self._pb_robot_id)
        joint_names = []
        for i in range(num_joints):
            joint_info = self.pb.getJointInfo(self._pb_robot_id, i)
            if joint_info[2] != self.pb.JOINT_FIXED:
                joint_names.append(joint_info[1].decode('utf-8'))
        
        self.joint_names = joint_names
    
    def _load_urdf_pinocchio(self, urdf_path: str, **kwargs):
        """Load URDF using Pinocchio."""
        self.model = self.pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # Extract joint names
        self.joint_names = [self.model.names[i] for i in range(1, len(self.model.names))]
    
    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics.
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            link_name: Optional link name (if None, uses end-effector)
            
        Returns:
            Tuple of (position, orientation)
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        
        if self.backend == "pybullet":
            return self._forward_kinematics_pybullet(joint_positions, link_name)
        elif self.backend == "pinocchio":
            return self._forward_kinematics_pinocchio(joint_positions, link_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _forward_kinematics_pybullet(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics using PyBullet."""
        if self._pb_robot_id is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Set joint positions
        joint_indices = list(range(self.n_dof))
        self.pb.resetJointStates(
            self._pb_robot_id,
            joint_indices,
            joint_positions,
            physicsClientId=self.pb_client
        )
        
        # Get end-effector pose
        if link_name is None:
            # Use last link
            link_index = self.pb.getNumJoints(self._pb_robot_id) - 1
        else:
            # Find link by name
            link_index = None
            for i in range(self.pb.getNumJoints(self._pb_robot_id)):
                link_info = self.pb.getLinkState(self._pb_robot_id, i)
                if link_info[12].decode('utf-8') == link_name:  # link_name field
                    link_index = i
                    break
            if link_index is None:
                raise ValueError(f"Link '{link_name}' not found")
        
        link_state = self.pb.getLinkState(
            self._pb_robot_id,
            link_index,
            physicsClientId=self.pb_client
        )
        
        position = np.array(link_state[4], dtype=np.float32)  # worldLinkFramePosition
        orientation = np.array(link_state[5], dtype=np.float32)  # worldLinkFrameOrientation (quaternion)
        
        return position, orientation
    
    def _forward_kinematics_pinocchio(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics using Pinocchio."""
        if self.model is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Forward kinematics
        self.pin.forwardKinematics(self.model, self.data, joint_positions)
        self.pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector frame
        if link_name is None:
            # Use last frame
            frame_id = len(self.model.frames) - 1
        else:
            # Find frame by name
            if not self.model.existFrame(link_name):
                raise ValueError(f"Frame '{link_name}' not found")
            frame_id = self.model.getFrameId(link_name)
        
        # Get pose
        placement = self.data.oMf[frame_id]
        position = placement.translation.astype(np.float32)
        
        # Convert rotation matrix to quaternion
        rotation_matrix = placement.rotation
        # Simple conversion (can be improved)
        orientation = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        return position, orientation
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z], dtype=np.float32)
    
    def jacobian(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute Jacobian matrix.
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            link_name: Optional link name (if None, uses end-effector)
            
        Returns:
            Jacobian matrix, shape (6, n_dof) for spatial Jacobian
        """
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        
        if self.backend == "pybullet":
            return self._jacobian_pybullet(joint_positions, link_name)
        elif self.backend == "pinocchio":
            return self._jacobian_pinocchio(joint_positions, link_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _jacobian_pybullet(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> np.ndarray:
        """Compute Jacobian using PyBullet."""
        if self._pb_robot_id is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Set joint positions
        joint_indices = list(range(self.n_dof))
        self.pb.resetJointStates(
            self._pb_robot_id,
            joint_indices,
            joint_positions,
            physicsClientId=self.pb_client
        )
        
        # Get link index
        if link_name is None:
            link_index = self.pb.getNumJoints(self._pb_robot_id) - 1
        else:
            link_index = None
            for i in range(self.pb.getNumJoints(self._pb_robot_id)):
                link_info = self.pb.getLinkState(self._pb_robot_id, i)
                if link_info[12].decode('utf-8') == link_name:
                    link_index = i
                    break
            if link_index is None:
                raise ValueError(f"Link '{link_name}' not found")
        
        # Compute Jacobian
        linear_jacobian, angular_jacobian = self.pb.calculateJacobian(
            self._pb_robot_id,
            link_index,
            [0, 0, 0],  # Local position (center of link)
            joint_positions.tolist(),
            [0] * self.n_dof,  # Joint velocities (not used)
            [0] * self.n_dof,  # Joint accelerations (not used)
            physicsClientId=self.pb_client
        )
        
        # Combine linear and angular Jacobians
        jacobian = np.vstack([
            np.array(linear_jacobian, dtype=np.float32),
            np.array(angular_jacobian, dtype=np.float32)
        ])
        
        return jacobian
    
    def _jacobian_pinocchio(
        self,
        joint_positions: np.ndarray,
        link_name: Optional[str] = None
    ) -> np.ndarray:
        """Compute Jacobian using Pinocchio."""
        if self.model is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Forward kinematics first
        self.pin.forwardKinematics(self.model, self.data, joint_positions)
        self.pin.updateFramePlacements(self.model, self.data)
        
        # Get frame ID
        if link_name is None:
            frame_id = len(self.model.frames) - 1
        else:
            if not self.model.existFrame(link_name):
                raise ValueError(f"Frame '{link_name}' not found")
            frame_id = self.model.getFrameId(link_name)
        
        # Compute Jacobian
        self.pin.computeFrameJacobian(
            self.model,
            self.data,
            frame_id,
            self.pin.LOCAL_WORLD_ALIGNED
        )
        
        jacobian = self.data.J.copy().astype(np.float32)
        
        return jacobian
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute inverse kinematics.
        
        Args:
            target_position: Target 3D position, shape (3,)
            target_orientation: Target orientation (quaternion), shape (4,), optional
            initial_guess: Initial guess for joint positions, shape (n_dof,)
            **kwargs: Additional IK parameters
            
        Returns:
            Joint positions, shape (n_dof,)
        """
        target_position = np.asarray(target_position, dtype=np.float32)
        
        if self.backend == "pybullet":
            return self._inverse_kinematics_pybullet(
                target_position, target_orientation, initial_guess, **kwargs
            )
        elif self.backend == "pinocchio":
            return self._inverse_kinematics_pinocchio(
                target_position, target_orientation, initial_guess, **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _inverse_kinematics_pybullet(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_guess: Optional[np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Inverse kinematics using PyBullet."""
        if self._pb_robot_id is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Get end-effector link index
        link_index = self.pb.getNumJoints(self._pb_robot_id) - 1
        
        # PyBullet IK
        if target_orientation is not None:
            target_orientation = np.asarray(target_orientation, dtype=np.float32)
            joint_positions = self.pb.calculateInverseKinematics(
                self._pb_robot_id,
                link_index,
                target_position.tolist(),
                targetOrientation=target_orientation.tolist(),
                residualThreshold=kwargs.get('residualThreshold', 1e-5),
                maxNumIterations=kwargs.get('maxNumIterations', 100),
                physicsClientId=self.pb_client
            )
        else:
            joint_positions = self.pb.calculateInverseKinematics(
                self._pb_robot_id,
                link_index,
                target_position.tolist(),
                residualThreshold=kwargs.get('residualThreshold', 1e-5),
                maxNumIterations=kwargs.get('maxNumIterations', 100),
                physicsClientId=self.pb_client
            )
        
        return np.array(joint_positions[:self.n_dof], dtype=np.float32)
    
    def _inverse_kinematics_pinocchio(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray],
        initial_guess: Optional[np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Inverse kinematics using Pinocchio (iterative)."""
        if self.model is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Initialize
        if initial_guess is None:
            q = np.zeros(self.n_dof, dtype=np.float32)
        else:
            q = np.asarray(initial_guess, dtype=np.float32)
        
        # Get end-effector frame
        frame_id = len(self.model.frames) - 1
        
        # Iterative IK
        max_iter = kwargs.get('maxNumIterations', 100)
        tolerance = kwargs.get('residualThreshold', 1e-5)
        
        for _ in range(max_iter):
            # Forward kinematics
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            
            # Compute error
            placement = self.data.oMf[frame_id]
            pos_error = target_position - placement.translation
            
            if target_orientation is not None:
                # Orientation error (simplified)
                target_quat = np.asarray(target_orientation, dtype=np.float32)
                current_quat = self._rotation_matrix_to_quaternion(placement.rotation)
                # Quaternion error (simplified)
                ori_error = target_quat - current_quat
                error = np.concatenate([pos_error, ori_error[:3]])  # Use first 3 components
            else:
                error = pos_error
            
            if np.linalg.norm(error) < tolerance:
                break
            
            # Compute Jacobian
            self.pin.computeFrameJacobian(
                self.model, self.data, frame_id, self.pin.LOCAL_WORLD_ALIGNED
            )
            J = self.data.J
            
            # Update (damped least squares)
            damping = kwargs.get('damping', 1e-6)
            dq = np.linalg.solve(J.T @ J + damping * np.eye(self.n_dof), J.T @ error)
            q = q + dq
        
        return q.astype(np.float32)
    
    def forward_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute forward dynamics.
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            joint_velocities: Joint velocities, shape (n_dof,)
            joint_torques: Joint torques, shape (n_dof,)
            **kwargs: Additional parameters
            
        Returns:
            Joint accelerations, shape (n_dof,)
        """
        if self.backend == "pinocchio":
            return self._forward_dynamics_pinocchio(
                joint_positions, joint_velocities, joint_torques, **kwargs
            )
        else:
            # PyBullet doesn't provide direct forward dynamics
            # Would need to use simulation step
            raise NotImplementedError(
                f"Forward dynamics not implemented for backend: {self.backend}. "
                "Use Pinocchio backend for forward dynamics."
            )
    
    def _forward_dynamics_pinocchio(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Forward dynamics using Pinocchio."""
        if self.model is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Compute dynamics
        self.pin.computeAllTerms(self.model, self.data, joint_positions, joint_velocities)
        
        # M * qdd + C * qd + G = tau
        # qdd = M^{-1} * (tau - C * qd - G)
        M = self.data.M
        Cqd = self.pin.nle(self.model, self.data, joint_positions, joint_velocities)
        tau = np.asarray(joint_torques, dtype=np.float32)
        
        qdd = np.linalg.solve(M, tau - Cqd)
        
        return qdd.astype(np.float32)
    
    def inverse_dynamics(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute inverse dynamics.
        
        Args:
            joint_positions: Joint positions, shape (n_dof,)
            joint_velocities: Joint velocities, shape (n_dof,)
            joint_accelerations: Joint accelerations, shape (n_dof,)
            **kwargs: Additional parameters
            
        Returns:
            Joint torques, shape (n_dof,)
        """
        if self.backend == "pinocchio":
            return self._inverse_dynamics_pinocchio(
                joint_positions, joint_velocities, joint_accelerations, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Inverse dynamics not implemented for backend: {self.backend}. "
                "Use Pinocchio backend for inverse dynamics."
            )
    
    def _inverse_dynamics_pinocchio(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Inverse dynamics using Pinocchio."""
        if self.model is None:
            raise ValueError("Robot not loaded. Call load_urdf() first.")
        
        # Compute inverse dynamics
        tau = self.pin.rnea(
            self.model,
            self.data,
            joint_positions,
            joint_velocities,
            joint_accelerations
        )
        
        return tau.astype(np.float32)
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get joint limits from loaded model."""
        if self.backend == "pybullet" and self._pb_robot_id is not None:
            lower = []
            upper = []
            for i in range(self.n_dof):
                joint_info = self.pb.getJointInfo(self._pb_robot_id, i)
                lower.append(joint_info[8])  # lowerLimit
                upper.append(joint_info[9])  # upperLimit
            return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)
        elif self.backend == "pinocchio" and self.model is not None:
            # Pinocchio doesn't store limits in model by default
            # Return unlimited
            return super().get_joint_limits()
        else:
            return super().get_joint_limits()
    
    def close(self) -> None:
        """Close robot model and free resources."""
        if self.backend == "pybullet" and self.pb_client is not None:
            if self._pb_robot_id is not None:
                self.pb.removeBody(self._pb_robot_id, physicsClientId=self.pb_client)
            # Don't disconnect if client was provided externally
            # self.pb.disconnect(physicsClientId=self.pb_client)
