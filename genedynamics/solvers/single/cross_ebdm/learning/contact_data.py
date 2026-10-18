"""Fixed-width full-body contact carry and episode-grouped host data boundary."""
from dataclasses import asdict
import numpy as np
from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
from genedynamics.tasks.transfer.contact import ContactCommands, ContactConfig
from genedynamics.tasks.transfer.contact_reference import contact_teacher, classify_contact_modes
from genedynamics.tasks.transfer.runtime import MechanismSpec, RandomRequest
from .artifact_bundle import save_bundle,load_bundle

JOINT_SLOTS=40
SEMANTIC_SLICE=(80,98)
BASE_CARRY_DIM=197
WBC_WARM_SLOTS=52


def contact_carry(state,runner,xp=np):
    if hasattr(state.physics,"qpos"):
        qpos,qvel=state.physics.qpos,state.physics.qvel
        pos=state.physics.site_xpos[:,runner._impl.sid]; rot=state.physics.site_xmat[:,runner._impl.sid]
        from mujoco.mjx._src import support
        import jax
        def velocity(data):
            p,r=support.jac(runner._impl.model,data,data.site_xpos[runner._impl.sid],runner._impl.body)
            return xp.concatenate((p.T@data.qvel,r.T@data.qvel))
        twist=jax.vmap(velocity)(state.physics)
    else:
        import mujoco
        qpos=xp.stack([d.qpos for d in state.physics]); qvel=xp.stack([d.qvel for d in state.physics])
        pos=xp.stack([d.site_xpos[runner._impl.sid] for d in state.physics]); rot=xp.stack([d.site_xmat[runner._impl.sid].reshape(3,3) for d in state.physics])
        velocities=[]
        for d in state.physics:
            p,r=np.zeros((3,runner._impl.model.nv)),np.zeros((3,runner._impl.model.nv))
            mujoco.mj_jacSite(runner._impl.model,d,p,r,runner._impl.sid)
            velocities.append(np.r_[p@d.qvel,r@d.qvel])
        twist=xp.asarray(velocities)
    if qpos.shape[-1]>JOINT_SLOTS or qvel.shape[-1]>JOINT_SLOTS: raise ValueError("body exceeds declared joint slots")
    pad=lambda x:xp.pad(x,((0,0),(0,JOINT_SLOTS-x.shape[-1])))
    b=qpos.shape[0]
    fields=(pad(qpos),pad(qvel),pos,rot,twist,state.reference_position,state.reference_rotation,state.last_command,
            state.command_buffer,state.force_integral,state.approach_integral,state.body_reference,state.measured_force,
            state.contact_seen,state.loss_count,state.hold_count)
    value=xp.concatenate([xp.asarray(x).reshape(b,-1) for x in fields],axis=-1)
    if hasattr(state,"wbc_warm"):
        if state.wbc_warm.shape[-1]>WBC_WARM_SLOTS: raise ValueError("WBC carry exceeds declared slots")
        extras=(state.foot_anchors,state.body_height,xp.pad(state.wbc_warm,((0,0),(0,WBC_WARM_SLOTS-state.wbc_warm.shape[-1]))),state.wbc_residual)
        value=xp.concatenate((value,*[xp.asarray(x).reshape(b,-1) for x in extras]),axis=-1)
    return value


def observe_contact(state, runner, xp=np):
    """Project method carry into the workcell's public contact semantics."""
    from genedynamics.envs.domains.manipulation.transfer_workcell.queries.contact import ContactSemantics

    carry = contact_carry(state, runner, xp)
    start, _ = SEMANTIC_SLICE
    return ContactSemantics(
        carry[:, start:start + 3],
        carry[:, start + 3:start + 12].reshape(-1, 3, 3),
        carry[:, start + 12:start + 18],
        state.measured_force,
        (state.measured_force > runner.config.contact_threshold).astype(xp.int32),
    )


def pad_contact_carry(value,dimension,xp=np):
    """Declared fixed/planar zero WBC state when sharing a floating surrogate."""
    if value.shape[-1]>dimension: raise ValueError("response omits required controller/WBC carry")
    pads=[(0,0)]*value.ndim; pads[-1]=(0,dimension-value.shape[-1])
    return xp.pad(value,pads)


def body_descriptor(config):
    return np.array([{"panda":0.,"xarm7":1.,"ur5e":2.,"go2_z1":3.}[config.robot],float(config.mobile),
                     (6 if config.robot in {"ur5e","go2_z1"} else 7)/7.],np.float32)


def collect_contact_dataset(path,*,environment,role,robots=("panda",),mechanisms=(MechanismSpec(),),groups=2,excite=False):
    from .data import code_fingerprint
    producer_code=code_fingerprint()
    if role not in {"source_demonstrations","response_development","response_validation"}: raise ValueError("unknown contact dataset role")
    if groups<2: raise ValueError("train/validation groups required")
    if role=="source_demonstrations" and (robots!=("panda",) or mechanisms!=(MechanismSpec(),)):
        raise ValueError("source prior data must be nominal Panda")
    records=[]; scenes=[]; identity=10000
    for robot in robots:
        config=ContactConfig(environment=environment,robot=robot,mobile=environment.startswith("mobile"))
        for mi,mechanism in enumerate(mechanisms):
            runner=ContactRunner(config=config,mechanism=mechanism); scenes.append(runner.describe())
            for group in range(groups):
                for mode in (0,1):
                    identity+=1
                    recovery=config.insertion
                    state=runner.reset_batch(RandomRequest(701+group,(identity,)),initial="jammed" if recovery else "clear")
                    rows=[]; rng=np.random.default_rng(100+group*2+mode)
                    for k in range(config.timeout_steps//10):
                        before=contact_carry(state,runner)
                        command=contact_teacher(config,state,k*10*config.control_dt,mode,recovery=recovery)
                        if excite:
                            command[...,:12]+=rng.uniform(-1,1,(1,1,12))*np.array([.002]*3+[.02]*3+[.2]*6)
                            command[...,12]*=rng.uniform(.8,1.2)
                        body=np.zeros((1,10,3),np.float32)
                        if config.mobile and k*10*config.control_dt<4.: body[...,0]=.005
                        result=runner.collect_chunk(state,ContactCommands(command,body))
                        after=contact_carry(result.state,runner)
                        trace=result.trace
                        support_failure=np.any(np.isin(np.asarray(result.state.status),[10,11,12]))
                        risks=np.array([[trace.force_peak.max(),trace.contact_lost.max(),max(trace.forbidden_collision.max(),support_failure)]],np.float32)
                        rows.append((before[0],command[0],after[0]-before[0],risks[0],trace))
                        state=result.state
                        # Keep fixed length including masked terminal padding.
                    traces={name:np.stack([getattr(row[4],name)[0] for row in rows]) for name in rows[0][4]._fields}
                    pos=traces["tcp_position"][None]; force=traces["normal_force"][None]; valid=traces["valid"][None]
                    observed=int(classify_contact_modes(pos,force,valid,insertion=config.insertion)[0])
                    records.append({"carry":np.stack([r[0] for r in rows]),"command":np.stack([r[1] for r in rows]),
                        "applied":np.stack([r[4].command[0] for r in rows]),
                        "delta":np.stack([r[2] for r in rows]),"risk":np.stack([r[3] for r in rows]),
                        "valid":valid[0],"position":pos[0],"normal_force":force[0],"status":int(state.status[0]),
                        "terminal":traces["terminal"],
                        "group":group,"mode":mode,"observed_mode":observed,"sample_id":identity,
                        "body":body_descriptor(config),"mechanism":np.array([mechanism.servo_gain,mechanism.delay_steps,mechanism.payload_mass],np.float32)})
    names=("carry","command","applied","delta","risk","valid","terminal","position","normal_force","status","group","mode","observed_mode","sample_id","body","mechanism")
    tensors={name:np.stack([r[name] for r in records]) for name in names}
    metadata={"artifact_kind":"contact_dataset","contact_schema":1,"environment":environment,"role":role,
        "code_fingerprint":producer_code,
        "joint_slots":JOINT_SLOTS,"semantic_slice":list(SEMANTIC_SLICE),"control_dt":config.control_dt,"horizon":10,
        "carry_dim":tensors["carry"].shape[-1],"floating_extension":"foot_anchors[12], body_height, QP_warm[52], QP_residual" if config.robot=="go2_z1" else None,
        "scenes":scenes,"group_split":"group 0 validation; all variants stay together",
        "episodes_attempted":len(records),"successes":int((tensors["status"]==1).sum()),"observed_modes":tensors["observed_mode"].tolist(),
        "valid_steps":int(tensors["valid"].sum()),"producer":"public contact teachers with optional excitation"}
    save_bundle(path,tensors,metadata)
    return tensors,metadata


def load_contact_dataset(path):
    data,meta=load_bundle(path)
    if meta.get("artifact_kind")!="contact_dataset" or meta.get("contact_schema")!=1: raise ValueError("expected contact dataset")
    b,k,h=data["valid"].shape
    if data["command"].shape!=(b,k,h,13) or data["carry"].shape!=(b,k,meta["carry_dim"]): raise ValueError("contact dataset layout mismatch")
    if meta.get("role") not in {"source_demonstrations","response_development","response_validation"}: raise ValueError("unknown contact data role")
    shapes={"delta":(b,k,meta["carry_dim"]),"risk":(b,k,3),"position":(b,k,h,3),"normal_force":(b,k,h),
            "body":(b,3),"mechanism":(b,3)}
    shapes.update({name:(b,) for name in ("status","group","mode","observed_mode","sample_id")})
    if any(data[name].shape!=shape for name,shape in shapes.items()): raise ValueError("contact tensor shape mismatch")
    if data["valid"].dtype.kind!="b" or np.any(np.diff(data["valid"].reshape(b,-1).astype(int),axis=1)>0):
        raise ValueError("contact valid mask must be sticky")
    if np.any((data["status"]<0)|(data["status"]>12)) or not np.isin(data["mode"],[0,1]).all(): raise ValueError("invalid outcome/mode")
    if meta["episodes_attempted"]!=b or meta["successes"]!=int((data["status"]==1).sum()) or meta["valid_steps"]!=int(data["valid"].sum()):
        raise ValueError("contact outcome denominator mismatch")
    observed=classify_contact_modes(data["position"],data["normal_force"],data["valid"],insertion=meta["environment"]!="guided_contact_alignment")
    if not np.array_equal(observed,data["observed_mode"]): raise ValueError("contact observed modes require physical trace evidence")
    if "terminal" in data:
        if data["terminal"].shape!=(b,k,h) or np.any(data["terminal"]&~data["valid"]): raise ValueError("invalid contact terminal mask")
        if not np.array_equal(data["terminal"].sum(axis=(1,2)),(data["status"]!=0).astype(int)):
            raise ValueError("contact terminal must occur once")
    if len(np.unique(data["sample_id"]))!=b: raise ValueError("duplicate contact episode ID")
    if meta["role"]=="source_demonstrations" and (np.any(data["body"][:,0]!=0) or not np.all(data["mechanism"]==[1.,0.,.5])):
        raise ValueError("privileged target data cannot train source prior")
    return data,meta
