"""Go2 + Z1 composite with real floating inertia, feet and common keyed fixture."""
from dataclasses import replace,asdict
from pathlib import Path
import copy
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from genedynamics.robots.profile import RobotProfile,RobotBinding,NamedElement,FLOATING_BASE,CARTESIAN_JACOBIAN,TORQUE_CONTROL,SINGLE_TOOL
from .contact_workcell import build_contact_workcell
from .workcell import Workcell,rotation


def build_go2_z1(config,mechanism):
    repo=Path(__file__).resolve().parents[5]
    go=ET.parse(repo/"third_party/mujoco_menagerie/unitree_go2/go2.xml").getroot()
    arm=ET.parse(repo/"third_party/mujoco_menagerie/unitree_z1/z1.xml").getroot()
    for root in (go,arm):
        for name in ("asset","keyframe","sensor"):
            for element in list(root.findall(name)): root.remove(element)
        for parent in root.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("mesh") is not None: parent.remove(geom)
                else:
                    geom.attrib.pop("material",None)
            for material in list(parent.findall("material")): parent.remove(material)
        root.find("compiler").attrib.pop("meshdir",None)
    # Default class names are local to the original files and must be prefixed.
    for element in arm.iter():
        for key in ("class","childclass","joint"):
            if key in element.attrib: element.set(key,"z1_"+element.get(key))
        if "name" in element.attrib: element.set("name","z1_"+element.get("name"))
    defaults=go.find("default")
    for child in list(arm.find("default")): defaults.append(child)
    trunk=go.find("worldbody/body")
    mounted=arm.find("worldbody/body"); mounted.set("pos",".02 0 .11"); trunk.append(mounted)
    endpoint=mounted.find(".//body[@name='z1_link06']")
    ET.SubElement(endpoint,"site",name="mobile_tcp",pos=".055 0 0",quat=".707106781 0 .707106781 0",size=".004")
    fixture=build_contact_workcell(replace(config,robot="panda",mobile=False),mechanism)
    fixture_root=ET.fromstring(fixture.xml)
    tool=copy.deepcopy(fixture_root.find(".//body[@name='transfer_tool']"))
    tool.set("pos",".055 0 0"); tool.set("quat",".707106781 0 .707106781 0"); endpoint.append(tool)
    world=go.find("worldbody")
    for geom in fixture_root.find("worldbody").findall("geom"):
        if geom.get("name")=="floor": geom.set("pos","0 0 0")
        world.append(copy.deepcopy(geom))
    contact=ET.SubElement(go,"contact")
    arm_bodies=["z1_link00","z1_link01","z1_link02","z1_link03","z1_link04","z1_link05","z1_link06","transfer_tool"]
    for i,name in enumerate(arm_bodies):
        for other in arm_bodies[i+1:i+3]: ET.SubElement(contact,"exclude",body1=name,body2=other)
    for name in arm_bodies[-4:-1]: ET.SubElement(contact,"exclude",body1=name,body2="transfer_tool")
    ET.SubElement(contact,"exclude",body1="base",body2="z1_link00")
    actuator=go.find("actuator")
    for old in arm.find("actuator"):
        ET.SubElement(actuator,"motor",name=old.get("name"),joint=old.get("joint"),gear="1",ctrlrange="-30 30",forcerange="-30 30")
    option=go.find("option"); option.attrib.update(timestep=str(config.physics_dt),gravity="0 0 -9.81",integrator="Euler",solver="Newton",iterations="20")
    xml=ET.tostring(go,encoding="unicode"); model=mujoco.MjModel.from_xml_string(xml)
    legs=tuple(model.joint(i).name for i in range(1,13)); arms=tuple(f"z1_joint{i}" for i in range(1,7))
    profile=RobotProfile(model_id="go2_z1",robot_type="mobile_manipulator",base_type=FLOATING_BASE,
        model_path_resolver=lambda:"composed_go2_z1",actuated_joints=legs+arms,
        actuator_names=tuple(model.actuator(i).name for i in range(model.nu)),
        joint_groups={"legs":tuple(range(12)),"arm":tuple(range(12,18))},elements={"tool_mount":NamedElement("site","mobile_tcp")},
        capabilities=frozenset({FLOATING_BASE,CARTESIAN_JACOBIAN,TORQUE_CONTROL,SINGLE_TOOL}),
        controller_defaults={"torque_limits":tuple(np.max(np.abs(model.actuator_ctrlrange),axis=-1))})
    binding=RobotBinding.from_mujoco_model(profile,model)
    q=np.r_[[0.,0.,.30,1.,0.,0.,0.],np.tile([0.,.9,-1.8],4),[0.,.785,-.261,-.523,0.,0.]]
    data=mujoco.MjData(model); data.qpos[:]=q; mujoco.mj_forward(model,data)
    feet=[model.geom(name).id for name in ("FL","FR","RL","RR")]
    q[2]+=.022-min(data.geom_xpos[feet,2])+.0002
    sid=binding.element_ids["tool_mount"]; indices=np.asarray(binding.qpos_indices[12:]); dofs=np.asarray(binding.dof_indices[12:])
    limits=np.array([model.jnt_range[model.joint(name).id] for name in arms])
    def arm_ik(position,yaw,seed):
        rng=np.random.default_rng(77); candidate=seed.copy()
        for attempt in range(12):
            if attempt: candidate[indices]=rng.uniform(limits[:,0]+.01,limits[:,1]-.01)
            for _ in range(300):
                data.qpos[:]=candidate; mujoco.mj_forward(model,data)
                error=np.r_[position-data.site_xpos[sid],Rotation.from_matrix(rotation(yaw)@data.site_xmat[sid].reshape(3,3).T).as_rotvec()]
                if np.linalg.norm(error[:3])<2e-5 and np.linalg.norm(error[3:])<2e-4: return candidate.copy()
                p,r=np.zeros((3,model.nv)),np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,p,r,sid)
                jac=np.r_[p,r][:,dofs]
                dq=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),error)
                candidate[indices]=np.clip(candidate[indices]+np.clip(dq,-.1,.1),limits[:,0]+.002,limits[:,1]-.002)
        raise ValueError(f"Go2+Z1 source pose unreachable: {error}")
    start=arm_ik(np.array([.44,-.2,.605]),0.,q); goal=arm_ik(np.array(config.goal),0.,start)
    manifest={**fixture.manifest,"model_sha256":hashlib.sha256(xml.encode()).hexdigest(),"body":{
        "type":"go2_z1_floating_composite","mount_xyz":[.02,0.,.11],"nq":model.nq,"nv":model.nv,"nu":model.nu,
        "support":"four real foot contacts; unconstrained floating base","wbc":"stance constrained inverse dynamics QP CPU",
        "qualification":"pending physical standing and manipulation"}}
    manifest["task"]=asdict(config)
    manifest["controller_signature"]="stance_inverse_dynamics_qp_v5_arm_torque_impedance"
    manifest["stiffness_semantics"]="six log ratios to Cartesian wrench stiffness [800 N/m]*3+[60 Nm/rad]*3; arm torque objective subject to hard floating-base support constraints"
    manifest["wbc_carry"]={"qp_solution":model.nv+12,"task_integral":6 if config.socket_profile=="matched_keyway" else 0,
        "packing":"QP solution followed by persistent task integral, zero padded to 52 in portable carry"}
    manifest["wbc_dt"]=.004
    manifest["reset_preparation_control_steps"]=50 if config.socket_profile=="matched_keyway" else 0
    manifest["reset_preparation_accounting"]="included within each charged reset/probe interaction; task clock starts after stance preparation"
    manifest["contact_whitelist"]+=[[name,"floor"] for name in ("FL","FR","RL","RR")]
    nominal=model.body_mass.copy(); nominal[model.body("transfer_tool").id]=.51
    scene=Workcell(model,binding,start,goal,nominal,model.body_ipos.copy(),xml,manifest)
    return scene,arm_ik
