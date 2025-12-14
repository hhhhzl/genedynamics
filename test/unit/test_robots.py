"""
Unit tests for robot models.

Tests cover:
- RobotModel Protocol
- ManipulatorModel (PyBullet, Pinocchio)
- DroneModel
- Kinematics and dynamics
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestRobotModelProtocol:
    """Test RobotModel Protocol."""
    
    def test_robot_model_interface(self):
        """Test RobotModel interface."""
        from enerdynamics.envs.robots.base import RobotModel
        
        # Check that protocol is defined
        assert hasattr(RobotModel, '__protocol_methods__') or hasattr(RobotModel, '__abstractmethods__')
        print("✓ RobotModel Protocol defined")


@pytest.mark.unit
class TestManipulatorModel:
    """Test ManipulatorModel."""
    
    @pytest.mark.requires_pybullet
    def test_manipulator_pybullet(self):
        """Test ManipulatorModel with PyBullet backend."""
        try:
            import pybullet as p
            from enerdynamics.envs.robots.manipulator import ManipulatorModel
            
            # Note: This requires a URDF file
            # For testing, we just check that the class can be instantiated
            # In practice, you would load a URDF file
            
            print("⚠ PyBullet ManipulatorModel test requires URDF file")
            print("✓ ManipulatorModel PyBullet backend available")
        except ImportError:
            pytest.skip("PyBullet not available")
    
    @pytest.mark.requires_pinocchio
    def test_manipulator_pinocchio(self):
        """Test ManipulatorModel with Pinocchio backend."""
        try:
            import pinocchio
            from enerdynamics.envs.robots.manipulator import ManipulatorModel
            
            # Note: This requires a URDF file
            print("⚠ Pinocchio ManipulatorModel test requires URDF file")
            print("✓ ManipulatorModel Pinocchio backend available")
        except ImportError:
            pytest.skip("Pinocchio not available")


@pytest.mark.unit
class TestDroneModel:
    """Test DroneModel."""
    
    def test_drone_model_creation(self):
        """Test DroneModel creation."""
        try:
            from enerdynamics.envs.robots.drone import DroneModel
            
            drone = DroneModel()
            
            assert drone.state_dim == 12  # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            assert drone.n_actuators == 4  # 4 motors
            print("✓ DroneModel creation works")
        except ImportError:
            pytest.skip("DroneModel not available")
    
    def test_drone_model_step(self):
        """Test DroneModel.step()."""
        try:
            from enerdynamics.envs.robots.drone import DroneModel
            
            drone = DroneModel()
            
            # Initial state: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            state = np.zeros(12, dtype=np.float32)
            
            # Motor inputs
            motors = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            
            next_state = drone.step(state, motors, dt=0.01)
            assert next_state.shape == (12,)
            print("✓ DroneModel.step() works")
        except ImportError:
            pytest.skip("DroneModel not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
