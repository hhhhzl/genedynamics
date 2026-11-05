"""Serial MuJoCo reference for the shared contact runner protocol."""
import copy
import numpy as np
import mujoco
from genedynamics.tasks.transfer.contact import (
    ContactState,ContactTrace,ContactChunk,ContactCommands,bounded_contact_command,hybrid_wrench,
    govern_substep_reference,govern_substep_wrench,contact_termination)
from genedynamics.core.control.transfer import integrate_rotation
from ..contact_workcell import build_contact_workcell
from ..workcell import solve_ik,rotation


class MujocoContact:
    _build_scene = staticmethod(build_contact_workcell)

    def __init__(self,config,device,mechanism):
        if device!="cpu": raise ValueError("MuJoCo contact reference supports CPU only")
        self.config,self.mechanism=config,mechanism
        if mechanism.delay_steps>4: raise ValueError("contact delay buffer capacity is four control steps")
        self.scene=self._build_scene(config,mechanism); self.model=self.scene.model
        self.sid=self.model.site(self.scene.binding.profile.elements["tool_mount"].name).id
        self.actuator_dofs=self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        self.limits=np.zeros(self.model.nv)
        self.limits[self.actuator_dofs]=np.max(np.abs(self.model.actuator_ctrlrange),axis=-1)
        names=[self.model.geom(i).name for i in range(self.model.ngeom)]
        self.allowed=frozenset(tuple(sorted((names.index(a),names.index(b)))) for a,b in self.scene.manifest["contact_whitelist"] if a in names and b in names)
        # Opt-in evaluator diagnostic.  It never enters ContactTrace or the
        # public observation and therefore cannot become policy feedback.
        self.record_substeps=False

    def describe(self): return {**self.scene.manifest,"backend":"numpy","compiled_chunk":False}
    def synchronize(self,value): pass

    def _task_position(self, data):
        return data.site_xpos[self.sid]

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
        ik_joints=getattr(self,"robot_joint_ids",None)
        if initial=="jammed":
            if not self.config.insertion: raise ValueError("jam reset requires socket task")
            qpos=solve_ik(self.model,self.sid,qpos,np.array([.45,-.20,.596]),rotation(.12),joint_ids=ik_joints)
        elif initial=="raised":
            if self.config.guide_profile!="spatial_dual_channel":raise ValueError("raised calibration reset requires spatial guide")
            qpos=solve_ik(self.model,self.sid,qpos,np.array([.38,-.20,.645]),rotation(),joint_ids=ik_joints)
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
                data.qpos[:]=solve_ik(self.model,self.sid,data.qpos,data.site_xpos[self.sid]+offset,rotation(yaw),joint_ids=ik_joints)
                mujoco.mj_forward(self.model,data)
            states.append(ContactState(data,data.site_xpos[self.sid].copy(),data.site_xmat[self.sid].reshape(3,3).copy(),
                np.zeros(13),np.zeros((5,13)),0.,0.,data.qpos[:3].copy() if self.config.mobile else np.zeros(3),0.,
                0,0,np.zeros(6),0,0,identity,0,False,0,0,0))
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

    def _contact_region_mask(self,data):
        """Evaluator-only contact identities encoded as a stable bit mask."""
        mask=0
        for c in data.contact:
            if c.dist>0 or tuple(sorted((c.geom1,c.geom2))) not in self.allowed:
                continue
            names={self.model.geom(c.geom1).name,self.model.geom(c.geom2).name}
            if "guide_surface" in names: mask|=1
            if any(name.startswith("corner_wall_") for name in names): mask|=2
            if "corner_end_stop" in names: mask|=4
            if any(name.startswith("socket_") or name.startswith("keyway_") for name in names): mask|=8
        return mask

    def _step(self,state,raw,body_twist=None):
        config=self.config; active=state.status==0
        bounded=bounded_contact_command(np,np.nan_to_num(raw),state.last_command,config)
        buffer=np.concatenate((bounded[None],state.command_buffer[:-1]))
        command=buffer[self.mechanism.delay_steps]
        established=state.contact_seen or (config.guide_profile in {"spatial_dual_channel","planar_corner_v1"} and state.loss_count>0)
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
        governor_phase=int(state.governor_phase)
        governor_phase_age=int(state.governor_phase_age)
        previous_substep_wrench=np.asarray(state.governor_previous_wrench).copy()
        governor_support_dwell=int(state.governor_support_dwell)
        governor_release_progress=int(state.governor_release_progress)
        governor_active=False
        peak=0.; lateral_peak=0.; saturation=0.; substep_events=[]
        # Evaluator-only physical-rate evidence.  Endpoint contact is an
        # aliased observation of a hybrid contact process, especially when
        # the physics timestep changes while the public 50 Hz contract does
        # not.  Keep the complete support predicate and the quantities needed
        # to audit it without exposing them to the policy.
        axial_support_substeps=np.zeros(config.substeps,dtype=bool)
        normal_force_substeps=np.zeros(config.substeps,dtype=float)
        tcp_position_substeps=np.zeros((config.substeps,3),dtype=float)
        joint_speed_substeps=np.zeros(config.substeps,dtype=float)
        force_peak_substeps=np.zeros(config.substeps,dtype=float)
        lateral_force_substeps=np.zeros(config.substeps,dtype=float)
        contact_region_substeps=np.zeros(config.substeps,dtype=np.uint8)
        fixture_position_substeps=np.full((config.substeps,3),np.nan,dtype=float)
        fixture_twist_substeps=np.full((config.substeps,6),np.nan,dtype=float)
        fixture_stop_substeps=np.zeros(config.substeps,dtype=bool)
        if active:
            for substep in range(config.substeps):
                jp,jr=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
                mujoco.mj_jacSite(self.model,data,jp,jr,self.sid); jac=np.concatenate((jp,jr))
                local_wrench,_,local_touch,_=self._measure(data)
                normal_touch=self._axial_support(data) if config.socket_profile=="matched_keyway" or config.guide_profile in {"spatial_dual_channel","planar_corner_v1"} else local_touch
                guide_prefix=(data.site_xpos[self.sid,0]<=.455
                              and data.site_xpos[self.sid,1]<=-.181)
                measured_support=np.linalg.norm(local_wrench[:3])>config.contact_threshold
                desired=hybrid_wrench(np,data.site_xpos[self.sid],data.site_xmat[self.sid].reshape(3,3),jac@data.qvel,
                    ref,refrot,command,integral,local_wrench[2],config,established or normal_touch,approach)
                nominal_desired=desired.copy()
                governor_previous=previous_substep_wrench.copy()
                desired,governor_phase,governor_phase_age,previous_substep_wrench,governor_support_dwell,governor_release_progress,governor_active=govern_substep_wrench(
                    np,desired,command[:2],local_wrench,
                    data.site_xpos[self.sid],jac@data.qvel,
                    measured_support,guide_prefix,governor_phase,
                    governor_phase_age,previous_substep_wrench,
                    governor_support_dwell,governor_release_progress,config)
                ref,reference_energy,reference_projected,effort_allowance,admitted_effort,admitted_spring,admitted_tangent=govern_substep_reference(
                    np,ref,data.site_xpos[self.sid],jac@data.qvel,command,
                    desired,governor_phase,config,
                    support_dwell=governor_support_dwell)
                desired=np.concatenate((admitted_tangent,desired[2:]))
                if self.record_substeps:
                    event=dict(substep=substep,measured_before=local_wrench.copy(),
                               previous_measured=governor_previous,
                               nominal_desired=nominal_desired.copy(),desired_before=desired.copy(),
                               governor_active=bool(governor_active),governor_phase=int(governor_phase),
                               governor_phase_age=int(governor_phase_age),
                               governor_support_dwell=int(governor_support_dwell),
                               governor_release_progress=int(governor_release_progress),
                               admitted_reference=ref.copy(),
                               tangent_reference_energy=float(reference_energy),
                               reference_projected=bool(reference_projected),
                               tangent_effort_allowance=float(effort_allowance),
                               admitted_tangent_effort=float(admitted_effort),
                               admitted_tangent_spring=float(admitted_spring),
                               admitted_tangent_wrench=admitted_tangent.copy(),
                               tcp_position_before=data.site_xpos[self.sid].copy(),tcp_twist_before=(jac@data.qvel).copy())
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
                ratio=np.max(np.abs(tau[self.actuator_dofs])/self.limits[self.actuator_dofs]); saturation=max(saturation,ratio)
                data.ctrl[:]=np.clip(tau,-self.limits,self.limits)[self.actuator_dofs]
                mujoco.mj_step(self.model,data); mujoco.mj_forward(self.model,data)
                wrench,force,is_contact,bad=self._measure(data)
                axial_support_substeps[substep]=self._axial_support(data)
                normal_force_substeps[substep]=wrench[2]
                tcp_position_substeps[substep]=data.site_xpos[self.sid]
                joint_speed_substeps[substep]=np.max(np.abs(data.qvel[getattr(self,"robot_dof_indices",slice(None))]))
                force_peak_substeps[substep]=force
                lateral_force_substeps[substep]=np.linalg.norm(wrench[:2])
                contact_region_substeps[substep]=self._contact_region_mask(data)
                if hasattr(self,"fixture_observation"):
                    fixture=self.fixture_observation(data)
                    fixture_position_substeps[substep]=fixture["fixture_position"]
                    fixture_twist_substeps[substep]=fixture["fixture_twist"]
                    coordinate=abs(float(data.qpos[self.fixture_qpos_index]))
                    fixture_stop_substeps[substep]=coordinate>=.98*config.fixture_travel_m
                if self.record_substeps:
                    event.update(measured_after=wrench.copy(),tcp_position_after=data.site_xpos[self.sid].copy(),
                                 force_peak_after=float(force),lateral_force_after=float(np.linalg.norm(wrench[:2])))
                    substep_events.append(event)
                peak=max(peak,force); lateral_peak=max(lateral_peak,float(np.linalg.norm(wrench[:2]))); allowed|=is_contact; forbidden|=bad
        normal=wrench[2]
        candidate=state._replace(last_command=np.where(raw[12]>.1,command,np.zeros_like(command)))
        finite=np.all(np.isfinite(data.qpos))&np.all(np.isfinite(data.qvel))&np.all(np.isfinite(raw))
        if config.mobile: finite&=np.all(np.isfinite(body_twist))
        status,seen,loss,hold,lost=contact_termination(np,candidate,self._task_position(data),normal,peak,forbidden,finite,saturation>1,config,rotation=data.site_xmat[self.sid].reshape(3,3),lateral_peak=lateral_peak,
            axial_contact=self._axial_support(data) if config.socket_profile=="matched_keyway" or config.guide_profile in {"spatial_dual_channel","planar_corner_v1"} else None)
        # Freeze the integrator on torque saturation (conditional anti-windup).
        integral=np.where(saturation>1,state.force_integral,integral)
        robot_joints=getattr(self,"robot_joint_ids",slice(None)); robot_q=getattr(self,"robot_qpos_indices",slice(None))
        if np.any(data.qpos[robot_q]<self.model.jnt_range[robot_joints,0]-1e-4) or np.any(data.qpos[robot_q]>self.model.jnt_range[robot_joints,1]+1e-4): status=3
        if np.max(np.abs(data.qvel[getattr(self,"robot_dof_indices",slice(None))]))>3.: status=4
        approach=np.where(saturation>1,state.approach_integral,approach)
        new=ContactState(data,ref,refrot,bounded,buffer,integral,approach,body_reference,normal,
            governor_phase,governor_phase_age,previous_substep_wrench,
            governor_support_dwell,governor_release_progress,
            state.sample_id,state.steps+1,seen,loss,hold,status) if active else state
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
                established_at_start=bool(established),normal_velocity=float((event_jp@data.qvel)[2]),
                axial_support_substeps=axial_support_substeps,
                normal_force_substeps=normal_force_substeps,
                tcp_position_substeps=tcp_position_substeps,
                joint_speed_substeps=joint_speed_substeps,
                force_peak_substeps=force_peak_substeps,
                lateral_force_substeps=lateral_force_substeps,
                contact_region_substeps=contact_region_substeps)
            if hasattr(self,"fixture_observation"):
                self.last_step_events.update(
                    fixture_position_substeps=fixture_position_substeps,
                    fixture_twist_substeps=fixture_twist_substeps,
                    fixture_stop_substeps=fixture_stop_substeps,
                )
        if self.record_substeps:
            names=substep_events[0].keys() if substep_events else ()
            self.last_substep_trace={name:np.asarray([row[name] for row in substep_events]) for name in names}
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


class MujocoCompliantSocket(MujocoContact):
    """CPU passive-socket provider with robot-only control and measured fixture state."""
    from ..contact_workcell import build_compliant_socket_workcell as _builder
    _build_scene = staticmethod(_builder)

    def __init__(self, config, device, mechanism):
        super().__init__(config, device, mechanism)
        self.fixture_joint_id = self.model.joint("socket_slide_x").id
        self.fixture_body_id = self.model.body("compliant_socket").id
        self.fixture_qpos_index = int(self.model.jnt_qposadr[self.fixture_joint_id])
        self.fixture_dof_index = int(self.model.jnt_dofadr[self.fixture_joint_id])
        self.robot_joint_ids = np.asarray([i for i in range(self.model.njnt) if i != self.fixture_joint_id])
        self.robot_qpos_indices = self.model.jnt_qposadr[self.robot_joint_ids].copy()
        self.robot_dof_indices = self.model.jnt_dofadr[self.robot_joint_ids].copy()
        if np.any(self.actuator_dofs == self.fixture_dof_index):
            raise ValueError("fixture slide must remain passive")

    def _task_position(self, data):
        position = data.site_xpos[self.sid].copy()
        position[0] -= data.qpos[self.fixture_qpos_index]
        return position

    def fixture_observation(self, data):
        jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacBody(self.model, data, jp, jr, self.fixture_body_id)
        return dict(
            fixture_position=data.xpos[self.fixture_body_id].copy(),
            fixture_rotation=data.xmat[self.fixture_body_id].reshape(3, 3).copy(),
            fixture_twist=np.r_[jp @ data.qvel, jr @ data.qvel],
        )
