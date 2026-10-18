"""Host-only ObstacleTransfer composition. Meshes become conservative primitive proxies.

Robot profile names are resolved here, never in a compiled collector. This is a
development scene, not a calibrated real robot collision/dynamics model.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from genedynamics.tasks.transfer.runtime import ENVIRONMENT_NAME
from genedynamics.robots.profile import RobotBinding
from genedynamics.robots.panda.profile import panda_profile
from genedynamics.robots.xarm7.profile import xarm7_profile
from genedynamics.robots.ur5e.profile import ur5e_profile


def _vec(x):
    return " ".join(str(float(v)) for v in x)


def rotation(yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])


def _mesh_bounds(model, name):
    bid = model.body(name).id
    points = []
    has_collision=any(model.geom_contype[gid] or model.geom_conaffinity[gid] for gid in np.flatnonzero(model.geom_bodyid==bid))
    for gid in np.flatnonzero(model.geom_bodyid == bid):
        if has_collision and not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        kind=model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = model.mesh_vert[start:start + count]
        else:
            size=model.geom_size[gid]
            if kind==mujoco.mjtGeom.mjGEOM_CAPSULE: half=np.array([size[0],size[0],size[0]+size[1]])
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER: half=np.array([size[0],size[0],size[1]])
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE: half=np.repeat(size[0],3)
            elif kind==mujoco.mjtGeom.mjGEOM_BOX: half=size
            else: raise ValueError(f"unhandled native collision geometry on {name}")
            import itertools
            vertices=np.array(list(itertools.product((-1,1),repeat=3)))*half
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, model.geom_quat[gid])
        points.append(vertices @ mat.reshape(3, 3).T + model.geom_pos[gid])
    if not points:
        raise ValueError(f"no native collision coverage for {name}")
    vertices = np.concatenate(points)
    low, high = vertices.min(0), vertices.max(0)
    return (low + high) / 2, np.maximum((high - low) / 2, .003), vertices


def _enclosing_capsule(vertices):
    center = vertices.mean(0)
    _, _, axes = np.linalg.svd(vertices-center, full_matrices=False)
    axis = axes[0]
    projection = (vertices-center) @ axis
    middle = (projection.max()+projection.min())/2
    center = center + middle * axis
    projection -= middle
    radial_sq = np.sum((vertices-center-projection[:, None]*axis)**2, axis=1)
    half = (projection.max()-projection.min())/2
    candidates = []
    for fraction in np.linspace(.2, 1., 17):
        length = half * fraction
        radius = np.sqrt(np.max(radial_sq + np.maximum(np.abs(projection)-length, 0)**2)) + 1e-6
        volume = 2*np.pi*radius**2*length + 4*np.pi*radius**3/3
        candidates.append((volume, length, radius))
    _, length, radius = min(candidates)
    return center-axis*length, center+axis*length, radius


def solve_ik(model, site_id, q0, position, target_rotation):
    """Construction-time DLS feasibility check; not a learned source query."""
    from scipy.spatial.transform import Rotation
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    rng=np.random.default_rng(1701)
    for iteration in range(350*(12 if model.nq==6 else 1)):
        if iteration and iteration%350==0:
            data.qpos[:]=np.clip(np.asarray(q0)+rng.uniform(-3.,3.,model.nq),
                                 model.jnt_range[:,0]+.002,model.jnt_range[:,1]-.002)
        mujoco.mj_forward(model, data)
        current = data.site_xmat[site_id].reshape(3, 3)
        angular = Rotation.from_matrix(target_rotation @ current.T).as_rotvec()
        error = np.r_[position - data.site_xpos[site_id], angular]
        if np.linalg.norm(error[:3]) < 2e-5 and np.linalg.norm(error[3:]) < 2e-4:
            return data.qpos.copy()
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        jac = np.vstack((jp, jr))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .002**2 * np.eye(6), error)
        data.qpos[:] = np.clip(data.qpos + np.clip(dq, -.12, .12),
                               model.jnt_range[:, 0] + .002, model.jnt_range[:, 1] - .002)
    raise ValueError(f"ObstacleTransfer {model.names[:30]!r} pose unreachable: residual {error}")


@dataclass
class Workcell:
    model: object
    binding: RobotBinding
    start_qpos: np.ndarray
    goal_qpos: np.ndarray
    nominal_masses: np.ndarray
    nominal_ipos: np.ndarray
    xml: str
    manifest: dict


def build_workcell(task, mechanism, *, validate_scene=True):
    profile = {"panda":panda_profile,"xarm7":xarm7_profile,"ur5e":ur5e_profile}[task.robot]()
    root_dir = Path(__file__).resolve().parents[5]
    native_path = (root_dir / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
                   if task.robot == "panda" else Path(profile.model_path()))
    native = mujoco.MjModel.from_xml_path(str(native_path))
    root = ET.parse(profile.model_path()).getroot()
    for name in ("asset", "keyframe", "actuator", "contact", "tendon", "equality"):
        for element in list(root.findall(name)):
            root.remove(element)
    for parent in root.iter():
        for geom in list(parent.findall("geom")):
            parent.remove(geom)
        # Removed assets must not remain referenced by inherited defaults.
        for material in list(parent.findall("material")):
            parent.remove(material)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(timestep=str(task.physics_dt), integrator="Euler", solver="Newton",
                         iterations="8", ls_iterations="8", gravity="0 0 -9.81")
    compiler = root.find("compiler")
    compiler.attrib.pop("meshdir", None)
    compiler.set("inertiafromgeom", "auto")
    world = root.find("worldbody")
    names = (["base","shoulder_link","upper_arm_link","forearm_link","wrist_1_link","wrist_2_link","wrist_3_link"]
             if task.robot=="ur5e" else (["link0"] if task.robot == "panda" else ["link_base"]) + [f"link{i}" for i in range(1, 8)])
    proxy_bounds = {}
    for name in names:
        body = root.find(f".//body[@name='{name}']")
        center, size, vertices = _mesh_bounds(native, name)
        # Convex capsules enclosing every transformed mesh vertex also enclose
        # the native collision convex hull, without box-box compile overhead.
        start_point, end_point, radius = _enclosing_capsule(vertices)
        shape = ({"type": "box", "pos": _vec(center), "size": _vec(size)}
                 if task.robot == "panda" and name in {"link1", "link4"}
                 else {"type": "capsule", "fromto": _vec(np.r_[start_point, end_point]), "size": str(radius)})
        native_primitives=[gid for gid in np.flatnonzero(native.geom_bodyid==native.body(name).id)
                           if (native.geom_contype[gid] or native.geom_conaffinity[gid])
                           and native.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH]
        if task.robot=="ur5e" and native_primitives:
            kinds={mujoco.mjtGeom.mjGEOM_CAPSULE:"capsule",mujoco.mjtGeom.mjGEOM_CYLINDER:"cylinder",
                   mujoco.mjtGeom.mjGEOM_BOX:"box",mujoco.mjtGeom.mjGEOM_SPHERE:"sphere"}
            for index,gid in enumerate(native_primitives):
                kind=kinds[native.geom_type[gid]]; count=3 if kind=="box" else (1 if kind=="sphere" else 2)
                ET.SubElement(body,"geom",name=f"collision_{name}_{index}",type=kind,pos=_vec(native.geom_pos[gid]),
                    quat=_vec(native.geom_quat[gid]),size=_vec(native.geom_size[gid,:count]),mass="0",contype="1",conaffinity="3",
                    margin=str(task.contact_margin),rgba=".3 .45 .65 1")
        else:
            ET.SubElement(body, "geom", name=f"collision_{name}", **shape,
                          mass="0", contype="1", conaffinity="3",
                          margin=str(task.contact_margin), rgba="0.3 0.45 0.65 1")
        proxy_bounds[name] = {"center": center.tolist(), "half_size": size.tolist(),
                              "capsule_start": start_point.tolist(), "capsule_end": end_point.tolist(),
                              "capsule_radius": radius, "proxy_type": shape["type"]}
    # Neighboring conservative boxes can overlap at articulated joints. Exclude
    # graph distance <=2 explicitly; nonlocal self collisions remain enabled.
    contact = ET.SubElement(root, "contact")
    for i, name in enumerate(names):
        for other in names[i+1:i+3]:
            ET.SubElement(contact, "exclude", body1=name, body2=other)
    site = root.find(f".//site[@name='{profile.elements['tool_mount'].name}']")
    parent = next(body for body in root.iter("body") if site in list(body))
    tool = ET.SubElement(parent, "body", name="transfer_tool", pos=site.get("pos", "0 0 0"),
                         quat=site.get("quat", "1 0 0 0"))
    tool_geoms = [("gripper_left", (0., -.025, .025), (.025, .008, .025), .05),
                  ("gripper_right", (0., .025, .025), (.025, .008, .025), .05),
                  ("payload_bar", (0., 0., .07), (.06, .01, .01), mechanism.payload_mass / 2),
                  ("payload_leg", (-.05, .04, .07), (.01, .03, .01), mechanism.payload_mass / 2)]
    for name, pos, size, mass in tool_geoms:
        ET.SubElement(tool, "geom", name=name, type="box", pos=_vec(pos), size=_vec(size),
                      mass=str(mass), contype="1", conaffinity="3", margin=str(task.contact_margin),
                      rgba="0.85 0.55 0.2 1")
    # The attached tool and two preceding links have intended mounting overlap.
    for name in names[-3:]:
        ET.SubElement(contact, "exclude", body1=name, body2="transfer_tool")
    obstacles = [("blocker", (.50, 0., .42), (.035, .055, .09)),
                 ("side_panel", (.50, .40, .52), (.18, .015, .17)),
                 ("crossbeam", (.50, 0., 1.10), (.18, .30, .025))]
    for name, pos, size in obstacles:
        ET.SubElement(world, "geom", name=name, type="box", pos=_vec(pos), size=_vec(size),
                      contype="2", conaffinity="1", margin=str(task.contact_margin), rgba=".6 .6 .6 1")
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 -.15", size="2 2 .05",
                  contype="2", conaffinity="1", margin=str(task.contact_margin))
    ET.SubElement(world, "site", name="goal", pos=_vec(task.goal_position), size=".012", rgba="0 1 0 .5")
    actuators = ET.SubElement(root, "actuator")
    for name, joint, limit in zip(profile.actuator_names, profile.actuated_joints,
                                  profile.controller_defaults["torque_limits"]):
        ET.SubElement(actuators, "motor", name=name, joint=joint, gear="1",
                      ctrlrange=f"{-limit} {limit}", forcerange=f"{-limit} {limit}")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    binding = RobotBinding.from_mujoco_model(profile, model)
    sid = binding.element_ids["tool_mount"]
    start = solve_ik(model, sid, profile.controller_defaults["home_qpos"],
                     np.array(task.start_position), rotation())
    goal = solve_ik(model, sid, start, np.array(task.goal_position), rotation(task.goal_yaw))
    for label, q in (("start", start), ("goal", goal)):
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        penetrations = [(model.geom(c.geom1).name, model.geom(c.geom2).name, c.dist)
                        for c in data.contact if c.dist < -task.collision_tolerance]
        if penetrations and validate_scene:
            raise ValueError(f"{task.robot} ObstacleTransfer {label} collides: {penetrations}")
    # Compute nominal tool COM using the known nominal payload, not actual mass.
    nominal_masses = model.body_mass.copy()
    nominal_ipos = model.body_ipos.copy()
    tid = model.body("transfer_tool").id
    nominal_masses[tid] = .1 + task.nominal_payload_mass
    nominal_ipos[tid] = sum(np.array(pos) * (mass if not name.startswith("payload") else task.nominal_payload_mass / 2)
                            for name, pos, size, mass in tool_geoms) / nominal_masses[tid]
    manifest = {"task": ENVIRONMENT_NAME, "config": asdict(task), "mechanism": asdict(mechanism),
                "mujoco_version": mujoco.__version__,
                "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                "collision_geometry": "native-mesh enclosing capsules; Panda links 1/4 use AABBs (conservative)",
                "self_collision_exclusion": "chain distance <=2; tool mounting links 5,6,7",
                "proxy_bounds": proxy_bounds, "command_dim": 13, "force_channel_enabled": False,
                "task_frame": "world", "angular_velocity": "spatial radians/second",
                "controller_signature": {"version": "cartesian_impedance_pi_null_v1", "joint_damping": 2.,
                                         "null_stiffness": 10., "null_dls_regularization": .001,
                                         "integral_gains": [400.]*3 + [20.]*3,
                                         "integral_limits": [.1]*6},
                "clearance": "contact-derived, censored at contact_margin; not exact distance",
                "start_qpos": start.tolist(), "goal_qpos": goal.tolist()}
    return Workcell(model, binding, start, goal, nominal_masses, nominal_ipos, xml, manifest)
