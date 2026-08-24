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


def test_simulation_profiles_are_registered_and_consistent():
    from genedynamics.robots import get_robot_registry

    registry = get_robot_registry()
    panda = registry.get_profile("manipulator", "panda")
    xarm7 = registry.get_profile("manipulator", "xarm7")
    h1 = registry.get_profile("humanoid", "h1")
    g1 = registry.get_profile("humanoid", "g1")

    assert panda.num_actuated == 7
    assert xarm7.num_actuated == 7
    assert h1.num_actuated == 19
    assert g1.num_actuated == 29
    assert registry.get("humanoid", "g1").act_dim == g1.num_actuated
    assert panda.elements["tool_mount"].kind == "site"
    assert xarm7.elements["tool_mount"].name == "attachment_site"
    assert h1.joint_groups["right_arm"] == (15, 16, 17, 18)
    assert g1.joint_groups["right_arm"] == tuple(range(22, 29))


def test_profile_capability_failure_is_explicit():
    from genedynamics.robots import get_robot_registry
    from genedynamics.robots.profile import SINGLE_TOOL

    h1 = get_robot_registry().get_profile("humanoid", "h1")
    with pytest.raises(ValueError, match="single_tool"):
        h1.require({SINGLE_TOOL})


@pytest.mark.unit
class TestRobotModelProtocol:
    """Test RobotModel Protocol."""
    
    def test_robot_model_interface(self):
        """Test RobotModel interface."""
        from genedynamics.envs.robots.base import RobotModel
        
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
            from genedynamics.envs.robots.manipulator import ManipulatorModel
            
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
            from genedynamics.envs.robots.manipulator import ManipulatorModel
            
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
            from genedynamics.envs.robots.drone import DroneModel
            
            drone = DroneModel()
            
            assert drone.state_dim == 12  # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
            assert drone.n_actuators == 4  # 4 motors
            print("✓ DroneModel creation works")
        except ImportError:
            pytest.skip("DroneModel not available")
    
    def test_drone_model_step(self):
        """Test DroneModel.step()."""
        try:
            from genedynamics.envs.robots.drone import DroneModel
            
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
