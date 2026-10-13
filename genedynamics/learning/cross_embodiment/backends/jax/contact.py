"""Whole-chunk JIT/vmap/scan contact collector, with measured constraint forces."""
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support
from .execution import select_device,_fp32_scope
from ...contact import ContactState,ContactTrace,ContactChunk,ContactCommands,bounded_contact_command,hybrid_wrench,contact_termination
from ...core.control_math import integrate_rotation


class JaxContact:
    @_fp32_scope
    def __init__(self,config,device,mechanism,microbatch_size=None):
        from genedynamics.envs.domains.manipulation.cross_embodiment.backends.contact import MujocoContact
        from ...contracts import mechanism_specs,MechanismSpec
        self.mechanisms=mechanism_specs(mechanism)
        self.heterogeneous=not isinstance(mechanism,MechanismSpec)
        self.microbatch_size=microbatch_size
        self.references=tuple(MujocoContact(config,"cpu",m) for m in self.mechanisms)
        self.reference=self.references[0]
        self.config,self.mechanism=config,self.mechanisms[0]
        self.scene=self.reference.scene; host=self.scene.model
        self.sid=self.reference.sid; self.body=int(host.site_bodyid[self.sid]); self.device=select_device(device)
        cast=lambda x:x.astype(jp.float32) if hasattr(x,"dtype") and x.dtype.kind=="f" else x
        self.model=jax.tree.map(cast,mjx.put_model(host,device=self.device,impl="jax"))
        self.initial=jax.tree.map(cast,mjx.make_data(host,device=self.device,impl="jax"))
        with jax.default_device(self.device):
            self.start=jp.asarray(self.scene.start_qpos,dtype=jp.float32)
            self.limits=jp.asarray(self.reference.limits,dtype=jp.float32)
            self.actuator_dofs=jp.asarray(self.reference.actuator_dofs)
            self.support_geom=jp.asarray([host.geom(i).name.startswith("base_caster") for i in range(host.ngeom)])
            self.ranges=jp.asarray(host.jnt_range,dtype=jp.float32)
            self.masses=jp.asarray(self.scene.nominal_masses,dtype=jp.float32)
            self.ipos=jp.asarray(self.scene.nominal_ipos,dtype=jp.float32)
            self.root_ids=jp.asarray(host.body_rootid)
            allowed=np.zeros((host.ngeom,host.ngeom),bool)
            for a,b in self.reference.allowed: allowed[a,b]=allowed[b,a]=True
            self.allowed=jp.asarray(allowed)
            mask=np.zeros((host.nbody,host.nv),np.float32)
            for bid in range(1,host.nbody):
                ancestor=bid
                while ancestor:
                    mask[bid,host.dof_bodyid==ancestor]=1
                    ancestor=host.body_parentid[ancestor]
            self.body_dof_mask=jp.asarray(mask)
        self.models=None
        if self.heterogeneous:
            models=[self.model]+[jax.tree.map(cast,mjx.put_model(r.model,device=self.device,impl="jax")) for r in self.references[1:]]
            self.models=jax.tree.map(lambda *xs:jp.stack(xs),*models)
        self.gains=jax.device_put(jp.asarray([m.servo_gain for m in self.mechanisms],jp.float32),self.device)
        self.delays=jax.device_put(jp.asarray([m.delay_steps for m in self.mechanisms],jp.int32),self.device)
        self._reset_batched=jax.jit(jax.vmap(lambda model,q,i,n:self._reset_one(q,i,n,model)),device=self.device)
        self._collect_batched=jax.jit(jax.vmap(lambda model,g,d,s,c,b:self._chunk_one(s,c,b,model,g,d)),device=self.device)
        self._reset=jax.jit(jax.vmap(self._reset_one,in_axes=(None,0,0)),device=self.device)
        self._collect=jax.jit(jax.vmap(self._chunk_one),device=self.device)
        self._zero_body=jax.jit(lambda c:jp.zeros((*c.shape[:2],3),jp.float32),device=self.device)

    def describe(self):
        return {**self.scene.manifest,"backend":"jax","compiled_chunk":True,"device":str(self.device),
                "force_refresh":"post-integration mjx.forward; force/contact indexing at identical instant",
                "heterogeneous":self.heterogeneous,"microbatch_size":self.microbatch_size,
                "world_models":[r.scene.manifest for r in self.references]}

    @staticmethod
    def synchronize(value): jax.block_until_ready(value)

    def _reset_one(self,qpos,identity,noise,model=None):
        model=self.model if model is None else model
        data=mjx.forward(model,self.initial.replace(qpos=qpos+self.config.reset_joint_noise*noise))
        return ContactState(data,data.site_xpos[self.sid],data.site_xmat[self.sid],jp.zeros(13,jp.float32),
            jp.zeros((5,13),jp.float32),jp.float32(0),jp.float32(0),qpos[:3] if self.config.mobile else jp.zeros(3,jp.float32),
            jp.float32(0),identity,jp.int32(0),jp.bool_(False),jp.int32(0),jp.int32(0),jp.int32(0))

    @_fp32_scope
    def reset_batch(self,request,*,previous=None,mask=None,initial="clear"):
        if self.heterogeneous and len(request.sample_ids)!=len(self.mechanisms): raise ValueError("contact mechanism/world batch mismatch")
        if (previous is None)!=(mask is None): raise ValueError("partial reset needs state and mask")
        if previous is not None and (previous.sample_id.shape!=(len(request.sample_ids),) or np.shape(mask)!=(len(request.sample_ids),)):
            raise ValueError("partial reset shape mismatch")
        if initial=="clear": qpos=self.start
        elif initial=="jammed":
            host=self.reference.reset_batch(request,initial=initial)
            qpos=jax.device_put(jp.asarray(host.physics[0].qpos,dtype=jp.float32),self.device)
        else: raise ValueError("unknown contact reset")
        ids=jax.device_put(np.asarray(request.sample_ids,np.uint32),self.device)
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(request.seed),i))(ids)
        noise=jax.vmap(lambda k:jax.random.normal(k,(self.scene.model.nq,),dtype=jp.float32))(keys)
        state=self._reset_batched(self.models,jp.broadcast_to(qpos,(len(request.sample_ids),qpos.size)),ids,noise) if self.heterogeneous else self._reset(qpos,ids,noise)
        if previous is not None:
            chosen=jax.device_put(np.asarray(mask,bool),self.device)
            state=jax.tree.map(lambda n,o:jp.where(chosen.reshape((-1,)+(1,)*(n.ndim-1)),n,o),state,previous)
        return state

    def _gravity(self,data,model=None):
        model=self.model if model is None else model
        point=data.xpos+jp.einsum("bij,bj->bi",data.xmat,self.ipos)
        offset=point-data.subtree_com[self.root_ids]
        jac=(data.cdof[None,:,3:]+jp.cross(data.cdof[None,:,:3],offset[:,None,:]))*self.body_dof_mask[:,:,None]
        return jp.einsum("bnk,b,k->n",jac,-self.masses,model.opt.gravity)

    def _measure(self,data,model=None):
        model=self.model if model is None else model
        forces,indices=support.contact_force_dim(model,data,3)
        world=jp.einsum("nij,njk->nik",forces.reshape(-1,2,3),data.contact.frame[indices]).reshape(-1,6)
        pair=data.contact.geom[indices]
        valid=(pair[:,0]>=0)&(data.contact.dist[indices]<=0.)
        allowed=self.allowed[jp.maximum(pair[:,0],0),jp.maximum(pair[:,1],0)]&valid
        allowed&=~self.support_geom[jp.maximum(pair[:,0],0)]&~self.support_geom[jp.maximum(pair[:,1],0)]
        world=world.at[:,2].set(jp.abs(world[:,2]))
        wrench=jp.sum(jp.where(allowed[:,None],world,0.),axis=0)
        peak=jp.maximum(jp.max(jp.where(allowed,jp.linalg.norm(forces[:,:3],axis=-1),0.),initial=0.),jp.linalg.norm(wrench[:3]))
        # Includes every non-whitelisted penetrating contact, not just condim3.
        allpair=data.contact.geom; allvalid=(allpair[:,0]>=0)&(data.contact.dist < -1e-4)
        forbidden=jp.any(allvalid&~self.allowed[jp.maximum(allpair[:,0],0),jp.maximum(allpair[:,1],0)])
        return wrench,peak,jp.any(allowed),forbidden

    def _step(self,state,values,model=None,gain=None,delay=None):
        model=self.model if model is None else model
        gain=self.mechanism.servo_gain if gain is None else gain
        delay=self.mechanism.delay_steps if delay is None else delay
        raw,body_twist=values
        config=self.config; active=state.status==0
        bounded=bounded_contact_command(jp,jp.nan_to_num(raw),state.last_command,config)
        buffer=jp.concatenate((bounded[None],state.command_buffer[:-1]))
        command=buffer[delay]
        ref=state.reference_position+command[:3]*config.control_dt
        ref=ref.at[2].set(jp.where(command[12]>.01,state.physics.site_xpos[self.sid,2],ref[2]))
        refrot=integrate_rotation(jp,state.reference_rotation,command[3:6],config.control_dt)
        integral=jp.where((command[12]>.01)&state.contact_seen,jp.clip(state.force_integral+(command[12]-state.measured_force)*config.control_dt,
            -config.force_integral_limit,config.force_integral_limit),0.)
        jp0,_=support.jac(model,state.physics,state.physics.site_xpos[self.sid],self.body)
        approach=jp.where((command[12]>.01)&~state.contact_seen,jp.clip(state.approach_integral+
            (-config.approach_speed-(jp0.T@state.physics.qvel)[2])*config.control_dt,-config.approach_integral_limit,config.approach_integral_limit),0.)
        body_reference=state.body_reference
        if config.mobile:
            body_reference=jp.clip(body_reference+jp.clip(jp.nan_to_num(body_twist),-.05,.05)*config.control_dt,
                                   jp.array([-.15,-.15,-.4]),jp.array([.15,.15,.4]))
        def physics(carry,_):
            data,peak,lateral_peak,saturation,allowed,forbidden=carry
            jacp,jacr=support.jac(model,data,data.site_xpos[self.sid],self.body)
            jac=jp.concatenate((jacp.T,jacr.T))
            local_wrench,_,local_touch,_=self._measure(data,model)
            wrench=hybrid_wrench(jp,data.site_xpos[self.sid],data.site_xmat[self.sid],jac@data.qvel,
                ref,refrot,command,integral,local_wrench[2],config,state.contact_seen|local_touch,approach)
            posture=10.*(self.start-data.qpos)
            if config.mobile: posture=posture.at[:3].set(100.*(body_reference-data.qpos[:3])-20.*data.qvel[:3])
            null=posture-jac.T@jp.linalg.solve(jac@jac.T+1e-6*jp.eye(6),jac@posture)
            tau=gain*(jac.T@wrench+null)+self._gravity(data,model)+model.dof_damping*data.qvel-2.*data.qvel
            saturation=jp.maximum(saturation,jp.max(jp.abs(tau)/self.limits))
            data=mjx.forward(model,mjx.step(model,data.replace(ctrl=jp.clip(tau,-self.limits,self.limits)[self.actuator_dofs])))
            measured,force,touch,bad=self._measure(data,model)
            return (data,jp.maximum(peak,force),jp.maximum(lateral_peak,jp.linalg.norm(measured[:2])),saturation,allowed|touch,forbidden|bad),measured
        result,wrenches=jax.lax.scan(physics,(state.physics,jp.float32(0),jp.float32(0),jp.float32(0),jp.bool_(False),jp.bool_(False)),None,length=config.substeps)
        data,peak,lateral_peak,saturation,allowed,forbidden=result; wrench=wrenches[-1]; normal=wrench[2]
        finite=jp.all(jp.isfinite(data.qpos))&jp.all(jp.isfinite(data.qvel))&jp.all(jp.isfinite(raw))&jp.all(jp.isfinite(body_twist))
        status,seen,loss,hold,lost=contact_termination(jp,state._replace(last_command=jp.where(raw[12]>.1,command,jp.zeros_like(command))),data.site_xpos[self.sid],normal,peak,forbidden,finite,saturation>1,config,rotation=data.site_xmat[self.sid],lateral_peak=lateral_peak)
        status=jp.where(jp.any(data.qpos<self.ranges[:,0]-1e-4)|jp.any(data.qpos>self.ranges[:,1]+1e-4),3,status)
        status=jp.where(jp.max(jp.abs(data.qvel))>3.,4,status)
        integral=jp.where(saturation>1,state.force_integral,integral)
        approach=jp.where(saturation>1,state.approach_integral,approach)
        candidate=ContactState(data,ref,refrot,bounded,buffer,integral,approach,body_reference,normal,state.sample_id,state.steps+1,seen,loss,hold,status)
        new=jax.tree.map(lambda n,o:jp.where(active,n,o),candidate,state)
        stage=jp.where(self.config.insertion&(new.physics.site_xpos[self.sid,2]<config.goal[2]+.01),3,
            jp.where(normal>config.contact_threshold,1,jp.where(state.contact_seen,2,0)))
        trace=ContactTrace(new.physics.site_xpos[self.sid],new.physics.site_xmat[self.sid],command,wrench,
            jp.where(active,normal,state.measured_force),jp.where(active,peak,0.),allowed,forbidden,lost,saturation,
            new.force_integral,stage,new.status,active,active&(new.status!=0),jp.where(active,lateral_peak,0.))
        return new,trace

    def _chunk_one(self,state,commands,body,model=None,gain=None,delay=None):
        state,trace=jax.lax.scan(lambda s,v:self._step(s,v,model,gain,delay),state,(commands,body))
        return ContactChunk(state,trace)

    @_fp32_scope
    def collect_chunk(self,state,commands):
        body=None
        if isinstance(commands,ContactCommands): commands,body=commands
        if len(np.shape(commands))!=3 or np.shape(commands)[0]!=state.sample_id.shape[0] or np.shape(commands)[-1]!=13 or np.shape(commands)[1]<1:
            raise ValueError("commands must be [B,L,13]")
        commands=jax.device_put(jp.asarray(commands,dtype=jp.float32),self.device)
        if body is None: body=self._zero_body(commands)
        if np.shape(body)!=(*np.shape(commands)[:2],3): raise ValueError("body commands must be [B,L,3]")
        commands=jax.device_put(jp.asarray(commands,dtype=jp.float32),self.device)
        body=jax.device_put(jp.asarray(body,dtype=jp.float32),self.device)
        batch=state.sample_id.shape[0]
        if self.heterogeneous and batch!=len(self.mechanisms): raise ValueError("contact mechanism/world batch mismatch")
        if self.microbatch_size is None or self.microbatch_size>=batch:
            return self._collect_batched(self.models,self.gains,self.delays,state,commands,body) if self.heterogeneous else self._collect(state,commands,body)
        from .execution import JaxExecutionBackend
        cut=JaxExecutionBackend._slice_tree; results=[]
        for start in range(0,batch,self.microbatch_size):
            stop=min(start+self.microbatch_size,batch)
            s,c,b=(cut(v,start,stop) for v in (state,commands,body))
            results.append(self._collect_batched(cut(self.models,start,stop),cut(self.gains,start,stop),cut(self.delays,start,stop),s,c,b)
                if self.heterogeneous else self._collect(s,c,b))
        return JaxExecutionBackend._join(results)
