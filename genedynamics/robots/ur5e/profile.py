"""Six-DoF UR5e profile backed by the vendored Menagerie MJCF."""
from pathlib import Path
from genedynamics.robots.profile import RobotProfile,NamedElement,FIXED_BASE,CARTESIAN_JACOBIAN,TORQUE_CONTROL,SINGLE_TOOL


def _model_path():
    path=Path(__file__).resolve().parents[3]/"third_party/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    if not path.exists(): raise FileNotFoundError(path)
    return str(path)


def ur5e_profile():
    joints=("shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint")
    return RobotProfile(model_id="ur5e",robot_type="manipulator",base_type=FIXED_BASE,model_path_resolver=_model_path,
        actuated_joints=joints,actuator_names=("shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"),
        joint_groups={"arm":tuple(range(6))},elements={"tool_mount":NamedElement("site","attachment_site")},
        capabilities=frozenset({FIXED_BASE,CARTESIAN_JACOBIAN,TORQUE_CONTROL,SINGLE_TOOL}),
        controller_defaults={"home_qpos":(-1.5708,-1.5708,1.5708,-1.5708,-1.5708,0.),
                             "torque_limits":(150.,150.,150.,28.,28.,28.)})
