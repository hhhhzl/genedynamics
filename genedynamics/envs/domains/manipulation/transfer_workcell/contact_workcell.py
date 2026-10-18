"""Real arm/peg/fixture MJCF composition for CPU contact qualification."""
from dataclasses import asdict
import hashlib
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from genedynamics.tasks.transfer.runtime import ObstacleTransferConfig
from .workcell import build_workcell,solve_ik,rotation,_vec


def build_contact_workcell(config,mechanism):
    task=ObstacleTransferConfig(robot=config.robot,physics_dt=config.physics_dt,control_dt=config.control_dt,
        start_position=(.38,-.20,.605),goal_position=(.44,-.20,.605),goal_yaw=0.,reset_joint_noise=0.)
    scene=build_workcell(task,mechanism,validate_scene=False)
    root=ET.fromstring(scene.xml); world=root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name")!="floor": world.remove(geom)
    tool=root.find(".//body[@name='transfer_tool']")
    for geom in list(tool.findall("geom")): tool.remove(geom)
    if config.insertion:
        ET.SubElement(tool,"geom",name="contact_peg",type="box",pos="0 0 .08",size=".010 .006 .035",
            mass=str(mechanism.payload_mass),contype="1",conaffinity="3",margin="0",friction=".4 .01 .001")
        ET.SubElement(tool,"geom",name="contact_key",type="box",pos=".012 0 .08",size=".003 .003 .035",
            mass=".01",contype="1",conaffinity="3",margin="0",friction=".4 .01 .001")
        fixtures=[("socket_left",(.411,-.20,.458),(.01,.04,.023)),
                  ("socket_right",(.469,-.20,.458),(.01,.04,.023)),
                  ("socket_front",(.44,-.179,.458),(.019,.01,.023)),
                  ("socket_back",(.44,-.221,.458),(.019,.01,.023)),
                  ("socket_floor",(.44,-.20,config.goal[2]-.120),(.04,.04,.005))]
        if config.socket_profile=="matched_keyway":
            # Free cross-section is the union of the rectangular peg envelope
            # and its +x keyway, with c clearance. Two shoulders prevent the
            # key from rotating into a broad generic rectangular opening.
            c=config.mating_clearance; hx,hy,kx,ky=.010+c,.006+c,.015+c,.003+c
            wall=.010; z=.458; hz=.023
            fixtures=[("socket_left",(.44-hx-wall/2,-.20,z),(wall/2,hy+wall,hz)),
                ("socket_right",(.44+kx+wall/2,-.20,z),(wall/2,hy+wall,hz)),
                ("socket_front",(.44+(kx-hx)/2,-.20+hy+wall/2,z),((kx+hx)/2,wall/2,hz)),
                ("socket_back",(.44+(kx-hx)/2,-.20-hy-wall/2,z),((kx+hx)/2,wall/2,hz)),
                ("keyway_shoulder_front",(.44+(kx+hx)/2,-.20+(hy+ky)/2,z),((kx-hx)/2,(hy-ky)/2,hz)),
                ("keyway_shoulder_back",(.44+(kx+hx)/2,-.20-(hy+ky)/2,z),((kx-hx)/2,(hy-ky)/2,hz)),
                ("socket_floor",(.44,-.20,config.goal[2]-.120),(.04,.04,.005))]
    else:
        ET.SubElement(tool,"geom",name="contact_peg",type="sphere",pos="0 0 .10",size=".006",
            mass=str(mechanism.payload_mass),contype="1",conaffinity="3",margin="0",friction=".4 .01 .001")
        fixtures=[("guide_surface",(.43,-.20,.47),(.10,.07,.01))]
        if config.guide_profile=="spatial_dual_channel":
            # A rising surface and a central ridge leave two 16 mm channels
            # for a 12 mm tool tip. Both route orders are physically distinct.
            fixtures=[("guide_surface",(.42,-.19,.480),(.070,.040,.005)),
                ("guide_ridge",(.42,-.19,.502),(.015,.002,.022)),
                ("guide_left",(.364,-.19,.502),(.008,.030,.022)),
                ("guide_right",(.478,-.19,.502),(.008,.030,.022)),
                ("guide_back",(.42,-.216,.502),(.060,.008,.022)),
                ("guide_front",(.42,-.162,.502),(.060,.008,.022))]
    for name,pos,size in fixtures:
        geom=ET.SubElement(world,"geom",name=name,type="box",pos=_vec(pos),size=_vec(size),contype="2",conaffinity="1",
            margin="0",friction=".4 .01 .001",solref=".02 1",solimp=".95 .99 .001")
        if name=="guide_surface" and config.guide_profile=="spatial_dual_channel":
            from scipy.spatial.transform import Rotation
            quat=Rotation.from_euler("xy",[np.arctan(.1),-np.arctan(.1)]).as_quat()
            geom.set("quat",_vec(quat[[3,0,1,2]]))
    for geom in list(tool.findall("geom"))+[g for g in world.findall("geom") if g.get("name") in {f[0] for f in fixtures}]:
        geom.set("friction",f"{getattr(mechanism,'contact_friction',.4)} .01 .001")
        geom.set("solref",f"{getattr(mechanism,'contact_time_constant',.02)} 1")
    # Zero margins on arm proxies: only actual penetration is forbidden.
    for geom in root.iter("geom"): geom.set("margin","0")
    if config.mobile:
        # A ground-supported omnidirectional planar base. Roll/pitch/height are
        # constrained here; this is not a floating-base stability certificate.
        robot=world.find("body")
        world.remove(robot)
        chassis=ET.SubElement(world,"body",name="mobile_chassis")
        ET.SubElement(chassis,"joint",name="base_x",type="slide",axis="1 0 0",range="-.2 .2",damping="20")
        ET.SubElement(chassis,"joint",name="base_y",type="slide",axis="0 1 0",range="-.2 .2",damping="20")
        ET.SubElement(chassis,"joint",name="base_yaw",type="hinge",axis="0 0 1",range="-.5 .5",damping="10")
        ET.SubElement(chassis,"geom",name="chassis_collision",type="box",pos="0 0 -.05",size=".16 .12 .04",
            mass="30",contype="1",conaffinity="3",rgba=".2 .2 .2 1")
        for i,(x,y) in enumerate(((-.13,-.1),(-.13,.1),(.13,-.1),(.13,.1))):
            ET.SubElement(chassis,"geom",name=f"base_caster_{i}",type="sphere",pos=_vec((x,y,-.11)),size=".04",
                mass=".5",friction=".001 .001 .001",priority="1",contype="1",conaffinity="2")
        chassis.append(robot)
        contact=root.find("contact")
        for body in list(robot.iter("body"))[:3]:
            ET.SubElement(contact,"exclude",body1="mobile_chassis",body2=body.get("name"))
        actuators=root.find("actuator")
        for name,limit in (("base_x",300),("base_y",300),("base_yaw",100)):
            ET.SubElement(actuators,"motor",name=name+"_motor",joint=name,gear="1",ctrlrange=f"{-limit} {limit}")
    xml=ET.tostring(root,encoding="unicode"); model=mujoco.MjModel.from_xml_string(xml)
    sid=model.site(scene.binding.profile.elements["tool_mount"].name).id
    start=np.array([.44,-.20,.605]) if config.insertion else np.array([.38,-.20,.605])
    # Arm IK is solved with the base at its neutral pose, rather than letting
    # construction-time IK consume body mobility to hide arm infeasibility.
    if config.mobile:
        fixed=build_contact_workcell(__import__("dataclasses").replace(config,mobile=False),mechanism)
        scene.start_qpos=np.r_[np.zeros(3),fixed.start_qpos]
        scene.goal_qpos=np.r_[np.zeros(3),fixed.goal_qpos]
    else:
        scene.start_qpos=solve_ik(model,sid,scene.start_qpos,start,rotation())
        scene.goal_qpos=solve_ik(model,sid,scene.start_qpos,np.array(config.goal),rotation())
    scene.model,scene.xml=model,xml
    scene.nominal_masses=model.body_mass.copy(); scene.nominal_ipos=model.body_ipos.copy()
    scene.nominal_masses[model.body("transfer_tool").id]=.51 if config.insertion else .5
    # The contact tool replaces the asymmetric obstacle payload. Its nominal
    # COM must describe this tool, independently of the unknown actual mass.
    scene.nominal_ipos[model.body("transfer_tool").id]=([.012*.01/.51,0.,.08] if config.insertion else [0.,0.,.10])
    scene.manifest={"task":config.environment,"config":asdict(config),"model_sha256":hashlib.sha256(xml.encode()).hexdigest(),
        "contact_whitelist":[[name,fixture[0]] for name in ("contact_peg","contact_key") for fixture in fixtures],
        "force_channel":"positive compression along world -z, N; measured constraint wrench separate",
        "controller_signature":"hybrid_normal_force_tangential_impedance_v5_intent_release_strict_metrics","gpu_qualification":"not_run"}
    scene.manifest["nominal_compensation"]="known gravity and native joint viscous damping; actual payload excluded"
    scene.manifest["socket_geometry"]={"profile":config.socket_profile,"clearance_m":config.mating_clearance,
        "peg_cross_section_m":[.020,.012],"key_x_extent_m":.015,"dedicated_keyway":config.socket_profile=="matched_keyway"}
    scene.manifest["wrench_semantics"]=config.wrench_semantics
    scene.manifest["guide_geometry"]={"profile":config.guide_profile,
        "tip_diameter_m":.012,"channel_width_m":[.016,.018],"surface_slopes_xy":[.1,.1],
        "central_ridge":config.guide_profile=="spatial_dual_channel"} if not config.insertion else None
    scene.manifest["wrench_limits"]={"tangent_N":12.,"orientation_Nm":2.}
    scene.manifest["controller_clocks"]={"skill_dt":config.control_dt,"impedance_and_force_feedback_dt":config.physics_dt}
    if config.mobile:
        scene.manifest["contact_whitelist"] += [[f"base_caster_{i}","floor"] for i in range(4)]
        scene.manifest["body"]={"type":"ground_supported_planar_omnidirectional","dofs":["x","y","yaw"],
            "constraint":"roll/pitch/height mechanically constrained; not quadruped/humanoid support qualification",
            "controller":"task-space priority with nullspace base posture intent; world TCP retained",
            "support_contacts":[f"base_caster_{i}" for i in range(4)]}
    return scene
