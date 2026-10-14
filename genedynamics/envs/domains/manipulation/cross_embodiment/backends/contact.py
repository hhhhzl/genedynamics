"""Serial MuJoCo reference for the shared contact runner protocol."""
import copy
import numpy as np
import mujoco
from genedynamics.learning.cross_embodiment.contact import (
    ContactState,ContactTrace,ContactChunk,ContactCommands,bounded_contact_command,hybrid_wrench,contact_termination)
from genedynamics.learning.cross_embodiment.core.control_math import integrate_rotation
from ..contact_workcell import build_contact_workcell
from ..workcell import solve_ik,rotation


class MujocoContact:
    def __init__(self,config,device,mechanism):
        if device!="cpu": raise ValueError("MuJoCo contact reference supports CPU only")
        self.config,self.mechanism=config,mechanism
        if mechanism.delay_steps>4: raise ValueError("contact delay buffer capacity is four control steps")
        self.scene=build_contact_workcell(config,mechanism); self.model=self.scene.model
        self.sid=self.model.site(self.scene.binding.profile.elements["tool_mount"].name).id
        self.actuator_dofs=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        self.limits=np.zeros(self.model.nv)
        self.limits[self.actuator_dofs]=np.max(np.abs(self.model.actuator_ctrlrange),axis=-1)
        names=[self.model.geom(i).name for i in range(self.model.ngeom)]
        self.allowed=frozenset(tuple(sorted((names.index(a),names.index(b)))) for a,b in self.scene.manifest["contact_whitelist"] if a in names and b in names)

    def describe(self): return {**self.scene.manifest,"backend":"numpy","compiled_chunk":False}
    def synchronize(self,value): pass

    @staticmethod
    def _stack(states):
        return ContactState(tuple(s.physics for s in states),*(np.stack([s[i] for s in states]) for i in range(1,len(ContactState._fields))))
    @staticmethod
    def _slice(state,i): return ContactState(*(x[i] for x in state))

    def reset_batch(self,request,*,previous=None,mask=None,initial="clear"):
        if initial not in {"clear","jammed","raised"}: raise ValueError("unknown reset family")
        if (previous is None)!=(mask is None): raise ValueError("partial reset needs state and mask")
        if previous is not None and (len(previous.sample_id)!=len(request.sample_ids) or np.shape(mask)!=(len(request.sample_ids),)):
            raise ValueError("partial reset shape mismatch")
        qpos=self.scene.start_qpos
        if initial=="jammed":
            if not self.config.insertion: raise ValueError("jam reset requires socket task")
            qpos=solve_ik(self.model,self.sid,qpos,np.array([.45,-.20,.596]),rotation(.12))
        elif initial=="raised":
            if self.config.guide_profile!="spatial_dual_channel":raise ValueError("raised calibration reset requires spatial guide")
            qpos=solve_ik(self.model,self.sid,qpos,np.array([.38,-.20,.645]),rotation())
        states=[]
        for i,identity in enumerate(request.sample_ids):
            if previous is not None and not mask[i]: states.append(self._slice(previous,i)); continue
            data=mujoco.MjData(self.model)
            rng=np.random.default_rng(np.random.SeedSequence([request.seed,identity]))
            noise=rng.normal(size=self.model.nq)
            data.qpos[:]=qpos+self.config.reset_joint_noise*noise; mujoco.mj_forward(self.model,data)
            if self.config.reset_task_noise:
                # Perturb the public task pose into the free side of the fixture;
                # unconstrained joint noise can start inside a rigid socket.
                task_rng=np.random.default_rng(np.random.SeedSequence([request.seed,identity,1129])) if self.config.socket_profile=="matched_keyway" else rng
                offset=task_rng.uniform(-1.,1.,3)*self.config.reset_task_noise; offset[2]=abs(offset[2])+.001
                yaw=(.12 if initial=="jammed" else 0.)+task_rng.uniform(-.01,.01)
                data.qpos[:]=solve_ik(self.model,self.sid,data.qpos,data.site_xpos[self.sid]+offset,rotation(yaw))
                mujoco.mj_forward(self.model,data)
            states.append(ContactState(data,data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),
                np.zeros(13),np.zeros((5,13)),0.,0.,data.qpos[:3].copy() if self.config.mobile else np.zeros(3),0.,identity,0,False,0,0,0))
        return self._stack(states)

    def _measure(self,data):
        wrench=np.zeros(6); peak=0.; contact=False; forbidden=False
        for i,c in enumerate(data.contact):
            if c.dist>0: continue
            allowed=tuple(sorted((c.geom1,c.geom2))) in self.allowed
            if not allowed: forbidden|=c.dist < -1e-4; continue
            if self.model.geom(c.geom1).name.startswith("base_caster") or self.model.geom(c.geom2).name.startswith("base_caster"):
                continue  # support wrench is not tool normal force
            force=np.zeros(6); mujoco.mj_contactForce(self.model,data,i,force)
            world=(force.reshape(2,3)@c.frame.reshape(3,3)).ravel()
            # Compression is positive regardless of contact geom ordering.
            if self.config.wrench_semantics=="tool_at_tcp_world":
                # mj_contactForce is the wrench on geom2 in the contact
                # frame. Select the force ON the tool and shift its moment
                # from the contact point to the TCP, all in world axes.
                tool2=self.model.geom(c.geom2).name in {"contact_peg","contact_key"}
                signed=world*(1. if tool2 else -1.)
                moment=signed[3:]+np.cross(c.pos-data.site_xpos[self.sid],signed[:3])
                wrench+=np.r_[signed[:3],moment]
            else:
                normal=abs(world[2]); wrench+=np.r_[world[:2],normal,world[3:]]
            peak=max(peak,float(np.linalg.norm(force[:3]))); contact=True
        return wrench,max(peak,float(np.linalg.norm(wrench[:3]))),contact,forbidden

    def _axial_support(self,data):
        return any(c.dist<=0 and tuple(sorted((c.geom1,c.geom2))) in self.allowed and
                   abs(c.frame.reshape(3,3)[0,2])>.7 for c in data.contact)

    def _step(self,state,raw,body_twist=None):
        config=self.config; active=state.status==0
        bounded=bounded_contact_command(np,np.nan_to_num(raw),state.last_command,config)
        buffer=np.concatenate((bounded[None],state.command_buffer[:-1]))
        command=buffer[self.mechanism.delay_steps]
        established=state.contact_seen or (config.guide_profile=="spatial_dual_channel" and state.loss_count>0)
        data=copy.copy(state.physics)
        ref=state.reference_position+command[:3]*config.control_dt
        if command[12]>.01: ref[2]=data.site_xpos[self.sid,2]
        refrot=integrate_rotation(np,state.reference_rotation,command[3:6],config.control_dt)
        error=command[12]-state.measured_force
        integral=np.clip(state.force_integral+error*config.control_dt,-config.force_integral_limit,config.force_integral_limit)
        integral=np.where((command[12]>.01)&established,integral,0.)
        jp0=np.zeros((3,self.model.nv)); jr0=np.zeros_like(jp0)
        mujoco.mj_jacSite(self.model,data,jp0,jr0,self.sid)
        approach=np.where((command[12]>.01)&~np.asarray(established),np.clip(state.approach_integral+
            (-config.approach_speed-(jp0@data.qvel)[2])*config.control_dt,-config.approach_integral_limit,config.approach_integral_limit),0.)
        body_reference=state.body_reference
        if config.mobile:
            body_reference=np.clip(body_reference+np.clip(np.nan_to_num(body_twist),-.05,.05)*config.control_dt,
                [-.15,-.15,-.4],[.15,.15,.4])
        wrench,_,allowed,forbidden=self._measure(data)
        peak=0.; lateral_peak=0.; saturation=0.
        if active:
            for _ in range(config.substeps):
                jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                mujoco.mj_jacSite(self.model,data,jp,jr,self.sid); jac=np.concatenate((jp,jr))
                local_wrench,_,local_touch,_=self._measure(data)
                normal_touch=self._axial_support(data) if config.socket_profile=="matched_keyway" or config.guide_profile=="spatial_dual_channel" else local_touch
                desired=hybrid_wrench(np,data.site_xpos[self.sid],data.site_xmat[self.sid].reshape(3,3),jac@data.qvel,
                    ref,refrot,command,integral,local_wrench[2],config,established or normal_touch,approach)
                posture=10.*(self.scene.start_qpos-data.qpos)
                if config.mobile: posture[:3]=100.*(body_reference-data.qpos[:3])-20.*data.qvel[:3]
                null=posture-jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),jac@posture)
                # Known nominal mass model, independent of unknown payload.
                gravity=np.zeros(self.model.nv)
                for body in range(1,self.model.nbody):
                    point=data.xpos[body]+data.xmat[body].reshape(3,3)@self.scene.nominal_ipos[body]
                    mujoco.mj_jac(self.model,data,jp,jr,point,body)
                    gravity+=jp.T@(-self.scene.nominal_masses[body]*self.model.opt.gravity)
                tau=self.mechanism.servo_gain*(jac.T@desired+null)+gravity+self.model.dof_damping*data.qvel-2.*data.qvel
                ratio=np.max(np.abs(tau)/self.limits); saturation=max(saturation,ratio)
                data.ctrl[:]=np.clip(tau,-self.limits,self.limits)[self.actuator_dofs]
                mujoco.mj_step(self.model,data); mujoco.mj_forward(self.model,data)
                wrench,force,is_contact,bad=self._measure(data)
                peak=max(peak,force); lateral_peak=max(lateral_peak,float(np.linalg.norm(wrench[:2]))); allowed|=is_contact; forbidden|=bad
        normal=wrench[2]
        candidate=state._replace(last_command=np.where(raw[12]>.1,command,np.zeros_like(command)))
        finite=np.all(np.isfinite(data.qpos))&np.all(np.isfinite(data.qvel))&np.all(np.isfinite(raw))
        if config.mobile: finite&=np.all(np.isfinite(body_twist))
        status,seen,loss,hold,lost=contact_termination(np,candidate,data.site_xpos[self.sid],normal,peak,forbidden,finite,saturation>1,config,rotation=data.site_xmat[self.sid].reshape(3,3),lateral_peak=lateral_peak,
            axial_contact=self._axial_support(data) if config.socket_profile=="matched_keyway" or config.guide_profile=="spatial_dual_channel" else None)
        # Freeze the integrator on torque saturation (conditional anti-windup).
        integral=np.where(saturation>1,state.force_integral,integral)
        if np.any(data.qpos<self.model.jnt_range[:,0]-1e-4) or np.any(data.qpos>self.model.jnt_range[:,1]+1e-4): status=3
        if np.max(np.abs(data.qvel))>3.: status=4
        approach=np.where(saturation>1,state.approach_integral,approach)
        new=ContactState(data,ref,refrot,bounded,buffer,integral,approach,body_reference,normal,state.sample_id,state.steps+1,seen,loss,hold,status) if active else state
        stage=3 if config.insertion and data.site_xpos[self.sid,2]<config.goal[2]+.01 else (1 if normal>config.contact_threshold else (2 if state.contact_seen else 0))
        trace=ContactTrace(data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),command,wrench,normal,peak,
            allowed,forbidden,lost,saturation,new.force_integral,stage,new.status,active,active and new.status!=0,lateral_peak)
        if getattr(self,"record_events",False):
            # Diagnostic side channel only: it neither changes ContactTrace's
            # ABI nor supplies mechanism truth to a policy/model.
            event_jp=np.zeros((3,self.model.nv));event_jr=np.zeros_like(event_jp)
            mujoco.mj_jacSite(self.model,data,event_jp,event_jr,self.sid)
            self.last_step_events=dict(axial_support=self._axial_support(data),
                contact_seen=bool(new.contact_seen),loss_count=int(new.loss_count),
                hold_count=int(new.hold_count),approach_integral=float(new.approach_integral),
                established_at_start=bool(established),normal_velocity=float((event_jp@data.qvel)[2]))
        return new,trace

    def collect_chunk(self,state,commands):
        body=None
        if isinstance(commands,ContactCommands): commands,body=commands
        commands=np.asarray(commands,np.float32)
        if commands.ndim!=3 or commands.shape[0]!=len(state.sample_id) or commands.shape[-1]!=13 or commands.shape[1]<1:
            raise ValueError("commands must be [B,L,13]")
        if body is None: body=np.zeros((*commands.shape[:2],3),np.float32)
        if np.shape(body)!=(*commands.shape[:2],3): raise ValueError("body commands must be [B,L,3]")
        states=[]; traces=[]; event_batches=[]
        for i in range(len(state.sample_id)):
            current=self._slice(state,i); history=[]; event_history=[]
            for j,raw in enumerate(commands[i]):
                current,trace=self._step(current,raw,body[i,j]); history.append(trace)
                if getattr(self,"record_events",False):event_history.append(self.last_step_events)
            states.append(current); traces.append(ContactTrace(*(np.stack([t[j] for t in history]) for j in range(len(ContactTrace._fields)))))
            if event_history:event_batches.append(event_history)
        if event_batches:
            self.last_event_trace={name:np.asarray([[row[name] for row in batch] for batch in event_batches])
                                   for name in event_batches[0][0]}
        return ContactChunk(self._stack(states),ContactTrace(*(np.stack([t[j] for t in traces]) for j in range(len(ContactTrace._fields)))))
