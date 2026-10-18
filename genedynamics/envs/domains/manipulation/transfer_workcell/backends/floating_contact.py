"""CPU floating-base contact adapter with stance inverse-dynamics QP.

QP warm start, foot anchors and body reference live in explicit per-world carry.
The nominal dynamics model is independent of the actual payload mechanism.
This host OSQP provider makes no device-residency/GPU qualification claim.
"""
from collections import namedtuple
import copy
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sparse
import osqp
import mujoco
from genedynamics.tasks.transfer.contact import ContactState,ContactTrace,ContactChunk,ContactCommands,bounded_contact_command,contact_termination,hybrid_wrench
from genedynamics.core.control.transfer import integrate_rotation,orientation_error
from ..floating_workcell import build_go2_z1

FloatingState=namedtuple("FloatingState",ContactState._fields+("foot_anchors","body_height","wbc_warm","wbc_residual"))
FloatingTrace=namedtuple("FloatingTrace",ContactTrace._fields+("base_position","base_tilt","support_count","wbc_residual"))


class FloatingContact:
    def __init__(self,config,device,mechanism):
        if device!="cpu": raise NotImplementedError("floating OSQP adapter is a CPU reference; no GPU fallback")
        self.qp_stride=round(.004/config.physics_dt)
        if self.qp_stride<1 or not np.isclose(self.qp_stride*config.physics_dt,.004,rtol=0.,atol=1e-12):
            raise ValueError("floating physics_dt must divide the fixed 4 ms WBC period")
        if not np.isclose(round(config.control_dt/.004)*.004,config.control_dt,rtol=0.,atol=1e-12):
            raise ValueError("floating control_dt must be a multiple of the fixed 4 ms WBC period")
        self.config,self.mechanism=config,mechanism
        if mechanism.delay_steps>4: raise ValueError("delay exceeds buffer")
        self.scene,self.arm_ik=build_go2_z1(config,mechanism); self.model=self.scene.model
        self.sid=self.model.site("mobile_tcp").id
        self.qindices=np.asarray(self.scene.binding.qpos_indices); self.dofs=np.asarray(self.scene.binding.dof_indices)
        self.limits=np.max(np.abs(self.model.actuator_ctrlrange),axis=1)
        self.feet=np.array([self.model.geom(n).id for n in ("FL","FR","RL","RR")]); self.floor=self.model.geom("floor").id
        names=[self.model.geom(i).name for i in range(self.model.ngeom)]
        self.allowed=frozenset(tuple(sorted((names.index(a),names.index(b)))) for a,b in self.scene.manifest["contact_whitelist"] if a in names and b in names)
        nominal=ET.fromstring(self.scene.xml); nominal.find(".//geom[@name='contact_peg']").set("mass",".5")
        self.nominal=mujoco.MjModel.from_xml_string(ET.tostring(nominal,encoding="unicode"))
        self.nominal_data=mujoco.MjData(self.nominal)

    def describe(self): return {**self.scene.manifest,"backend":"numpy","compiled_chunk":False,"wbc_backend":"host_osqp",
        "nominal_dynamics":"known 0.5 kg payload model; no actual gain/delay/mass labels in QP"}
    def synchronize(self,value): pass
    @staticmethod
    def _slice(state,i): return type(state)(*(x[i] for x in state))
    @staticmethod
    def _stack(states): return type(states[0])(tuple(s.physics for s in states),*(np.stack([s[i] for s in states]) for i in range(1,len(states[0]))))

    def reset_batch(self,request,*,previous=None,mask=None,initial="clear"):
        if (previous is None)!=(mask is None): raise ValueError("partial reset needs state and mask")
        if previous is not None and (len(previous.sample_id)!=len(request.sample_ids) or np.shape(mask)!=(len(request.sample_ids),)):
            raise ValueError("partial reset batch mismatch")
        q=self.scene.start_qpos
        if initial=="jammed":
            if self.config.socket_profile!="matched_keyway":q=self.arm_ik(np.array([.45,-.2,.597]),.12,q)
        elif initial!="clear": raise ValueError("unknown reset")
        states=[]
        for i,identity in enumerate(request.sample_ids):
            if previous is not None and not mask[i]: states.append(self._slice(previous,i)); continue
            data=mujoco.MjData(self.model); data.qpos[:]=q
            arm=self.qindices[12:]
            noise=np.random.default_rng(np.random.SeedSequence([request.seed,identity])).normal(size=len(arm))
            data.qpos[arm]+=self.config.reset_joint_noise*noise
            mujoco.mj_forward(self.model,data)
            one=FloatingState(data,data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),
                np.zeros(13),np.zeros((5,13)),0.,0.,np.array([q[0],q[1],0.]),0.,identity,0,False,0,0,0,
                data.geom_xpos[self.feet].copy(),q[2],np.zeros(self.model.nv+12+(6 if self.config.socket_profile=="matched_keyway" else 0)),0.)
            if self.config.socket_profile=="matched_keyway":
                # Reset preparation is part of the charged interaction. Settle
                # the real compliant feet before placing the arm in the jam;
                # otherwise the standing transient itself strikes the lip.
                for _ in range(50):
                    one,_=self._step(one,np.zeros(13),np.zeros(3))
                    if one.status!=0:raise RuntimeError(f"floating reset stance preparation failed: {one.status}, {self.last_qp_diagnostic}")
                data=one.physics
                pose=np.array([.45,-.2,.597]) if initial=="jammed" else np.array([.44,-.2,.605])
                data.qpos[:]=self.arm_ik(pose,.12 if initial=="jammed" else 0.,data.qpos)
                data.qvel[:]=0.;data.qacc_warmstart[:]=0.;mujoco.mj_forward(self.model,data)
                base=data.xmat[self.model.body("base").id].reshape(3,3)
                one=FloatingState(data,data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),
                    np.zeros(13),np.zeros((5,13)),0.,0.,np.array([data.qpos[0],data.qpos[1],np.arctan2(base[1,0],base[0,0])]),
                    0.,identity,0,False,0,0,0,data.geom_xpos[self.feet].copy(),data.qpos[2],np.zeros(self.model.nv+18),0.)
            states.append(one)
        return self._stack(states)

    def _measure(self,data):
        wrench=np.zeros(6); peak=0.;contact=False; forbidden=False; feet=set()
        for i,c in enumerate(data.contact):
            pair=tuple(sorted((c.geom1,c.geom2)))
            if pair not in self.allowed: forbidden|=c.dist < -1e-4; continue
            foot=c.geom1 if c.geom1 in self.feet else c.geom2 if c.geom2 in self.feet else None
            force=np.zeros(6); mujoco.mj_contactForce(self.model,data,i,force)
            if foot is not None:
                if force[0]>1.: feet.add(foot)
                continue
            if c.dist>0: continue
            peak=max(peak,float(np.linalg.norm(force[:3])))
            world=(force.reshape(2,3)@c.frame.reshape(3,3)).ravel()
            if self.config.wrench_semantics=="tool_at_tcp_world":
                tool2=self.model.geom(c.geom2).name in {"contact_peg","contact_key"}
                world*=1. if tool2 else -1.
                world[3:]+=np.cross(c.pos-data.site_xpos[self.sid],world[:3])
            else: world[2]=abs(world[2])
            wrench+=world; contact=True
        return wrench,max(peak,float(np.linalg.norm(wrench[:3]))),contact,forbidden,len(feet)

    def _axial_support(self,data):
        return any(c.dist<=0 and tuple(sorted((c.geom1,c.geom2))) in self.allowed and
            any(self.model.geom(g).name in {"contact_peg","contact_key"} for g in (c.geom1,c.geom2)) and
            abs(c.frame.reshape(3,3)[0,2])>.7 for c in data.contact)

    def _qp(self,data,state,ref,refrot,command,body_reference,warm):
        m,d=self.nominal,self.nominal_data; n=m.nv; size=n+12
        d.qpos[:]=data.qpos; d.qvel[:]=data.qvel; mujoco.mj_forward(m,d)
        mass=np.zeros((n,n)); mujoco.mj_fullM(m,mass,d.qM)
        p,r=np.zeros((3,n)),np.zeros((3,n)); mujoco.mj_jacSite(m,d,p,r,self.sid); tool=np.r_[p,r]
        measured,_,touch,_,_=self._measure(data)
        external=tool.T@measured
        foot_j=[]; foot_acc=[]
        for i,gid in enumerate(self.feet):
            point=d.geom_xpos[gid]; bid=int(m.geom_bodyid[gid]); mujoco.mj_jac(m,d,p,r,point,bid)
            jac=p.copy(); dotp,dotr=np.zeros_like(p),np.zeros_like(r)
            mujoco.mj_jacDot(m,d,dotp,dotr,point,bid)
            foot_j.append(jac)
            foot_acc.append(100.*(state.foot_anchors[i]-point)-20.*(jac@d.qvel)-dotp@d.qvel)
        feet=np.concatenate(foot_j); foot_acc=np.concatenate(foot_acc)
        rows=[]; targets=[]
        def objective(jac,target,weight):
            rows.append(np.c_[jac,np.zeros((len(jac),12))]*weight**.5); targets.append(target*weight**.5)
        mujoco.mj_jacBody(m,d,p,r,m.body("base").id); base=np.r_[p,r]
        bodygoal=np.array([body_reference[0],body_reference[1],state.body_height])
        bodyrot=np.array([[np.cos(body_reference[2]),-np.sin(body_reference[2]),0.],[np.sin(body_reference[2]),np.cos(body_reference[2]),0.],[0.,0.,1.]])
        error=np.r_[bodygoal-d.xpos[m.body("base").id],orientation_error(np,d.xmat[m.body("base").id].reshape(3,3),bodyrot)]
        objective(base,60.*error-12.*(base@d.qvel),20.)
        error=np.r_[ref-d.site_xpos[self.sid],orientation_error(np,d.site_xmat[self.sid].reshape(3,3),refrot)]
        # Preserve all six shared impedance channels in the WBC adapter.
        # A floating adapter must not silently discard the skill stiffness.
        stiffness=np.exp(command[6:12])
        acceleration=80.*stiffness*error+15.*np.sqrt(stiffness)*(command[:6]-tool@d.qvel)
        if self.config.socket_profile=="matched_keyway":acceleration+=400.*state.wbc_warm[-6:]
        if command[12]>.01:
            acceleration[2]=(-.5*(command[12]-measured[2])-10.*(tool@d.qvel)[2]
                             if state.contact_seen or (self._axial_support(data) if self.config.socket_profile=="matched_keyway" else touch) else -.4+800.*state.approach_integral-80.*(tool@d.qvel)[2])
            if self.config.socket_profile=="matched_keyway" and (state.contact_seen or state.loss_count>0):
                acceleration[2]-=.5*state.force_integral
        # Foot acceleration/friction/torque constraints remain hard. Give the
        # 6D manipulation task priority over the soft base-posture objective;
        # otherwise a millimetre-scale tool bias survives at a tight keyway.
        if self.config.socket_profile!="matched_keyway":objective(tool,acceleration,1000.)
        selector=np.eye(n)[self.dofs]
        posture=40.*(self.scene.start_qpos[self.qindices]-d.qpos[self.qindices])-8.*d.qvel[self.dofs]
        objective(selector,posture,.02)
        rows.append(np.c_[np.zeros((12,n)),np.eye(12)]*.0001); targets.append(np.zeros(12))
        a=np.concatenate(rows); target=np.concatenate(targets); h=a.T@a+1e-5*np.eye(size); g=-a.T@target
        objective_scale=max(1.,float(np.diag(h).max())) if self.config.socket_profile=="matched_keyway" else 1.
        h/=objective_scale;g/=objective_scale
        equality=np.r_[np.c_[mass[:6],-feet[:,:6].T],np.c_[feet,np.zeros((12,12))]]
        rhs=np.r_[-d.qfrc_bias[:6]+external[:6],foot_acc]
        torque=np.c_[mass[self.dofs],-feet[:,self.dofs].T]
        bias=d.qfrc_bias[self.dofs]-external[self.dofs]+m.dof_damping[self.dofs]*d.qvel[self.dofs]
        if self.config.socket_profile=="matched_keyway":
            # Track physical arm torque impedance inside the stance QP. An
            # acceleration PID can request an infeasible low-inertia wrist
            # response and amplify native dry friction; the shared Cartesian
            # wrench has the same units as the fixed-arm decoder instead.
            wrench=hybrid_wrench(np,d.site_xpos[self.sid],d.site_xmat[self.sid].reshape(3,3),tool@d.qvel,
                ref,refrot,command,state.force_integral,measured[2],self.config,
                state.contact_seen or state.loss_count>0 or self._axial_support(data),state.approach_integral)
            task_i=np.array([400.]*3+[20.]*3)*state.wbc_warm[-6:]
            if command[12]>.01:task_i[2]=0.
            wrench+=task_i
            wrench[:2]=np.clip(wrench[:2],-12.,12.)
            wrench[3:]=np.clip(wrench[3:],-self.config.orientation_wrench_limit,self.config.orientation_wrench_limit)
            arm_target=(tool.T@wrench+d.qfrc_bias+m.dof_damping*d.qvel-2.*d.qvel)[self.dofs[12:]]
            rows.append(torque[12:]*1000.**.5);targets.append((arm_target-bias[12:])*1000.**.5)
            a=np.concatenate(rows);target=np.concatenate(targets)
            h=a.T@a+1e-5*np.eye(size);g=-a.T@target
            objective_scale=max(1.,float(np.diag(h).max()));h/=objective_scale;g/=objective_scale
        inequalities=[torque]; low=[-self.limits-bias]; high=[self.limits-bias]
        force_rows=[]; force_low=[]; force_high=[]
        for i in range(4):
            row=np.zeros(size); row[n+3*i+2]=1; force_rows.append(row); force_low.append(0.); force_high.append(250.)
            for axis in (0,1):
                for sign in (-1,1):
                    row=np.zeros(size); row[n+3*i+axis]=sign; row[n+3*i+2]=-.6
                    force_rows.append(row); force_low.append(-np.inf); force_high.append(0.)
        inequalities.append(np.array(force_rows)); low.append(np.array(force_low)); high.append(np.array(force_high))
        inequalities.append(np.c_[np.eye(n),np.zeros((n,12))]); low.append(np.full(n,-80.)); high.append(np.full(n,80.))
        matrix=np.concatenate([equality,*inequalities]); lower=np.concatenate([rhs,*low]); upper=np.concatenate([rhs,*high])
        solver=osqp.OSQP(); solver.setup(P=sparse.csc_matrix(np.triu(h)),q=g,A=sparse.csc_matrix(matrix),l=lower,u=upper,
            verbose=False,eps_abs=1e-5 if self.config.socket_profile=="matched_keyway" else 1e-4,
            eps_rel=1e-5 if self.config.socket_profile=="matched_keyway" else 1e-4,
            max_iter=20000 if self.config.socket_profile=="matched_keyway" else 2000,polishing=self.config.socket_profile=="matched_keyway")
        solver.warm_start(x=warm); result=solver.solve()
        self.last_qp_diagnostic=dict(status=result.info.status,iterations=int(result.info.iter),
            primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),objective_scale=objective_scale)
        if result.x is None or result.info.status_val not in (1,2): return np.zeros(m.nu),warm,1e3
        solution=result.x
        residual=max(float(np.max(np.abs(equality@solution-rhs))),float(np.max(np.maximum(lower-matrix@solution,0.))),float(np.max(np.maximum(matrix@solution-upper,0.))))
        tau=torque@solution+bias
        return tau,solution,residual

    def _step(self,state,raw,body):
        config=self.config; active=state.status==0; data=copy.copy(state.physics)
        bounded=bounded_contact_command(np,np.nan_to_num(raw),state.last_command,config)
        buffer=np.r_[bounded[None],state.command_buffer[:-1]]; command=buffer[self.mechanism.delay_steps]
        ref=state.reference_position+command[:3]*config.control_dt
        if command[12]>.01: ref[2]=data.site_xpos[self.sid,2]
        refrot=integrate_rotation(np,state.reference_rotation,command[3:6],config.control_dt)
        bodyref=np.clip(state.body_reference+np.clip(np.nan_to_num(body),-.03,.03)*config.control_dt,[-.03,-.03,-.15],[.03,.03,.15])
        # Explicit admittance memory overcomes posture bias before contact.
        p,r=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,data,p,r,self.sid)
        approach=np.clip(state.approach_integral+(-config.approach_speed-(p@data.qvel)[2])*config.control_dt,-config.approach_integral_limit,config.approach_integral_limit) if command[12]>.01 and not state.contact_seen else 0.
        controlstate=state._replace(approach_integral=approach)
        if config.socket_profile=="matched_keyway":
            integral=np.clip(state.force_integral+(command[12]-state.measured_force)*config.control_dt,-config.force_integral_limit,config.force_integral_limit)
            integral=integral if command[12]>.01 and state.contact_seen else 0.
            controlstate=controlstate._replace(force_integral=integral)
        warm=state.wbc_warm[:self.model.nv+12]
        task_integral=None
        if config.socket_profile=="matched_keyway":
            task_error=np.r_[ref-data.site_xpos[self.sid],orientation_error(np,data.site_xmat[self.sid].reshape(3,3),refrot)]
            if command[12]>.01:task_error[2]=0.
            task_integral=np.clip(state.wbc_warm[-6:]+task_error*config.control_dt,[-.02]*3+[-.1]*3,[.02]*3+[.1]*3)
            controlstate=controlstate._replace(wbc_warm=np.r_[warm,task_integral])
        peak=0.; lateral_peak=0.; saturation=0.; forbidden=False; residual=0.; tau=np.zeros(self.model.nu)
        wrench,_,touch,_,support=self._measure(data)
        if active:
            for substep in range(config.substeps):
                if substep%self.qp_stride==0:
                    tau,warm,error=self._qp(data,controlstate,ref,refrot,command,bodyref,warm); residual=max(residual,error)
                actual=self.mechanism.servo_gain*tau
                if hasattr(self.mechanism,"body_servo_gain"):
                    actual[:12]=self.mechanism.body_servo_gain*tau[:12]
                saturation=max(saturation,float(np.max(np.abs(actual)/self.limits)))
                data.ctrl[:]=np.clip(actual,-self.limits,self.limits)
                mujoco.mj_step(self.model,data); mujoco.mj_forward(self.model,data)
                wrench,force,is_contact,bad,support=self._measure(data); peak=max(peak,force); lateral_peak=max(lateral_peak,float(np.linalg.norm(wrench[:2]))); touch|=is_contact; forbidden|=bad
        status,seen,lost_count,hold,lost=contact_termination(np,state._replace(last_command=np.where(raw[12]>.1,command,np.zeros_like(command))),data.site_xpos[self.sid],wrench[2],peak,
            forbidden,np.isfinite(data.qpos).all()&np.isfinite(data.qvel).all()&np.isfinite(raw).all()&np.isfinite(body).all(),saturation>1,config,rotation=data.site_xmat[self.sid].reshape(3,3),lateral_peak=lateral_peak,
            axial_contact=self._axial_support(data) if config.socket_profile=="matched_keyway" else None,
            seated_history=(state.hold_count>0 or state.loss_count>0) if config.socket_profile=="matched_keyway" else None)
        tilt=np.linalg.norm(orientation_error(np,data.xmat[self.model.body("base").id].reshape(3,3),np.eye(3))[:2])
        if data.qpos[2]<.15 or tilt>.5: status=10
        if residual>.01: status=11
        if state.steps>50 and support<2: status=12
        joint_ids=self.model.dof_jntid[self.dofs]
        if np.any(data.qpos[self.qindices]<self.model.jnt_range[joint_ids,0]-1e-4) or np.any(data.qpos[self.qindices]>self.model.jnt_range[joint_ids,1]+1e-4): status=3
        if np.max(np.abs(data.qvel[self.dofs[12:]]))>3.: status=4
        candidate=state._replace(physics=data,reference_position=ref,reference_rotation=refrot,last_command=bounded,command_buffer=buffer,
            body_reference=bodyref,measured_force=wrench[2],steps=state.steps+1,contact_seen=seen,loss_count=lost_count,hold_count=hold,status=status,
            wbc_warm=warm if task_integral is None else np.r_[warm,task_integral if saturation<=1 else state.wbc_warm[-6:]],wbc_residual=residual,approach_integral=approach,
            force_integral=controlstate.force_integral if saturation<=1 else state.force_integral)
        new=candidate if active else state
        stage=3 if data.site_xpos[self.sid,2]<config.goal[2]+.01 else (1 if wrench[2]>.3 else (2 if state.contact_seen else 0))
        trace=FloatingTrace(data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),command,wrench,wrench[2],peak,
            touch,forbidden,lost,saturation,new.force_integral,stage,new.status,active,active and new.status!=0,lateral_peak,
            data.qpos[:3].copy(),tilt,support,residual)
        return new,trace

    def collect_chunk(self,state,commands):
        body=None
        if isinstance(commands,ContactCommands): commands,body=commands
        commands=np.asarray(commands,np.float32)
        if commands.ndim!=3 or commands.shape[0]!=len(state.sample_id) or commands.shape[-1]!=13 or commands.shape[1]<1: raise ValueError("commands must be [B,L,13]")
        if body is None: body=np.zeros((*commands.shape[:2],3))
        body=np.asarray(body)
        if body.shape!=(*commands.shape[:2],3): raise ValueError("body commands must be [B,L,3]")
        states=[]; histories=[]
        for i in range(len(state.sample_id)):
            current=self._slice(state,i); trace=[]
            for j,command in enumerate(commands[i]): current,t=self._step(current,command,body[i,j]); trace.append(t)
            states.append(current); histories.append(FloatingTrace(*(np.stack([t[j] for t in trace]) for j in range(len(FloatingTrace._fields)))))
        return ContactChunk(self._stack(states),FloatingTrace(*(np.stack([t[j] for t in histories]) for j in range(len(FloatingTrace._fields)))))
