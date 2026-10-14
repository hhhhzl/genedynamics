"""Event-conditioned source references and backend-neutral pose feedback.

No mechanism truth enters these features. The reference table is a transparent
source teacher; PhaseGaussianPrior distils the table on nominal-source data.
Phase memory is explicit and belongs to the caller, never to the provider.
"""
from typing import NamedTuple
import numpy as np


class BehaviorState(NamedTuple):
    phase: object
    contact_count: object
    mode: object
    tracking_integral: object


def behavior_reset(batch,mode,xp=np):
    return BehaviorState(xp.zeros(batch,dtype="int32"),xp.zeros(batch,dtype="int32"),xp.asarray(mode,dtype="int32"),xp.zeros((batch,6),dtype="float32"))


def keyway_alignment(position,rotation,clearance,*,xp=np,margin=.0001):
    """Conservative projected solid-envelope check before entering the hole.

    Uses known task geometry, not mechanism truth. Both peg end faces matter:
    roll/pitch can displace the tip even when the TCP itself is centered.
    """
    vertices=xp.asarray([[x,y,z] for x in (-.010,.010) for y in (-.006,.006) for z in (.045,.115)] +
                        [[x,y,z] for x in (.009,.015) for y in (-.003,.003) for z in (.045,.115)],dtype=position.dtype)
    points=xp.einsum("bij,kj->bki",rotation,vertices)+position[:,None,:]
    relative=points[...,:2]-xp.asarray([.44,-.20],dtype=position.dtype)
    c=clearance-margin
    base=xp.all((xp.abs(relative[:,:8,0])<.010+c)&(xp.abs(relative[:,:8,1])<.006+c),axis=-1)
    key=xp.all((relative[:,8:,0]>-.010-c)&(relative[:,8:,0]<.015+c)&(xp.abs(relative[:,8:,1])<.003+c),axis=-1)
    return base&key


def advance_behavior(memory,position,rotation,force,*,insertion=True,xp=np,dt=.2,goal_z=.550,alignment_tolerance=.0025,alignment_mask=None):
    """Jam/load → withdraw → align → insert. Transitions use actual feedback."""
    count=xp.where(force>.5,memory.contact_count+1,0)
    phase=memory.phase
    desired=xp.asarray([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],dtype=position.dtype)
    angle=xp.sqrt(xp.sum((rotation-desired)**2,axis=(-2,-1)))
    if insertion:
        phase=xp.where((phase==0)&(count>=2),1,phase)
        phase=xp.where((phase==1)&(position[:,2]>.609)&(angle<.045),2,phase)
        xy=xp.sqrt(xp.sum((position[:,:2]-xp.asarray([.44,-.20],dtype=position.dtype))**2,axis=-1))
        aligned=xy<alignment_tolerance if alignment_mask is None else alignment_mask
        phase=xp.where((phase==2)&aligned&(angle<.035),3,phase)
    else:
        phase=xp.where((phase==0)&(count>=2),1,phase)
        first=xp.where(memory.mode==0,position[:,0]>.455,position[:,1]>-.183) if alignment_mask is None else alignment_mask
        phase=xp.where((phase==1)&first,3,phase)
    updated=BehaviorState(phase,count,memory.mode,memory.tracking_integral)
    reference=teacher_reference(updated,insertion=insertion,xp=xp,goal_z=goal_z)
    angular=.5*xp.sum(xp.cross(xp.swapaxes(rotation,-1,-2),xp.swapaxes(xp.broadcast_to(desired,rotation.shape),-1,-2)),axis=-2)
    error=xp.concatenate((reference[:,:3]-position,angular),axis=-1)
    previous=xp.where((phase!=memory.phase)[:,None],0.,memory.tracking_integral)
    integral=xp.clip(previous+.8*dt*error,-.08,.08)
    integral=xp.concatenate((integral[:,:2],xp.where(reference[:,4:5]>.01,0.,integral[:,2:3]),integral[:,3:]),axis=-1)
    return updated._replace(tracking_integral=integral)


def teacher_reference(memory,*,insertion=True,xp=np,goal_z=.550):
    """Source-only nominal waypoints; never adjusted using target truth."""
    phase,mode=memory.phase,memory.mode
    if insertion:
        x=xp.where(phase==0,.45,.44)
        y=xp.where((phase==1)&(mode==1),-.212,-.20)
        x=xp.where(phase==1,.45,x)
        z=xp.where((phase==1)|(phase==2),.614,goal_z)
    else:
        x=xp.where((phase==0)|((phase==1)&(mode==1)),.38,.46)
        y=xp.where((phase==0)|((phase==1)&(mode==0)),-.20,-.18)
        z=xp.ones_like(x)*(.586 if goal_z==.550 else goal_z)
    force=xp.where((phase==1)|(phase==2),0.,5.) if insertion else xp.ones_like(x)*5.
    # Output xyz, yaw offset from downward tool orientation, force.
    return xp.stack((x,y,z,xp.zeros_like(x),force),axis=-1)


def reference_commands(reference,reference_position,reference_rotation,position,rotation,*,xp=np,horizon=10,dt=.02,integral_bias=None):
    """Bounded commands: integrated reference + actual SE(3) tracking error.

    Angular correction remains possible when source feed-forward is zero.
    The loop is over static skill time; batch operations are vectorized.
    """
    from .core.control_math import integrate_rotation,orientation_error
    b=position.shape[0]; p=reference_position; r=reference_rotation
    bias=xp.zeros((b,6),dtype=position.dtype) if integral_bias is None else integral_bias
    yaw=reference[:,3]; c,s=xp.cos(yaw),xp.sin(yaw); zero=xp.zeros_like(c)
    desired=xp.stack((xp.stack((c,s,zero),-1),xp.stack((s,-c,zero),-1),xp.stack((zero,zero,-xp.ones_like(c)),-1)),-2)
    commands=[]
    for _ in range(horizon):
        velocity=xp.clip(2.*(reference[:,:3]-p)+2.*(reference[:,:3]-position)+bias[:,:3],-.02,.02)
        # The loaded normal channel is force controlled. Do not integrate a
        # fictitious insertion-position reference while force overrides it.
        velocity=xp.concatenate((velocity[:,:2],xp.where(reference[:,4:5]>.01,0.,velocity[:,2:3])),axis=-1)
        # orientation_error's single-world helper is expressed in batch form.
        err=.5*xp.sum(xp.cross(xp.swapaxes(r,-1,-2),xp.swapaxes(desired,-1,-2)),axis=-2)
        actual=.5*xp.sum(xp.cross(xp.swapaxes(rotation,-1,-2),xp.swapaxes(desired,-1,-2)),axis=-2)
        angular=xp.clip(2.*err+2.*actual+bias[:,3:],-.12,.12)
        command=xp.concatenate((velocity,angular,xp.zeros((b,6),dtype=position.dtype),reference[:,4:5]),axis=-1)
        commands.append(command); p=p+velocity*dt
        if xp.__name__=="jax.numpy":
            import jax
            r=jax.vmap(lambda rr,ww:integrate_rotation(xp,rr,ww,dt))(r,angular)
        else:
            r=xp.stack([integrate_rotation(xp,rr,ww,dt) for rr,ww in zip(r,angular)])
    return xp.stack(commands,axis=1)


class PhaseGaussianPrior:
    """Small learned source law over event-conditioned reference poses/force.

    FrozenRegressor retains the existing JAX/Torch/native model interface.
    Noise is explicit. Geometry feedback is identical for every baseline.
    """
    def __init__(self,regressor):
        if regressor.metadata.get("learned_role")!="phase_source_prior": raise ValueError("expected phase source prior")
        self.regressor=regressor

    def reference(self,memory,noise):
        backend=self.regressor.model.backend
        if backend=="torch":
            import torch
            inputs=torch.nn.functional.one_hot(memory.phase.to(torch.int64)*2+memory.mode.to(torch.int64),8).to(noise.dtype)
            value=self.regressor.predict(inputs)[0]
            sampled=value+noise*value.new_tensor(self.regressor.metadata["reference_std"])
            force=torch.where(value[:,4:5]>1.,sampled[:,4:5].clamp(0.,10.),torch.zeros_like(sampled[:,4:5]))
            return torch.cat((sampled[:,:4],force),dim=-1)
        if backend=="jax":
            import jax.numpy as xp
        else: xp=np
        # Frozen source regressors use FP32 on every provider, even when the
        # contact physics/controller carry explicitly uses FP64.
        inputs=xp.eye(8,dtype=xp.float32)[memory.phase*2+memory.mode]
        value=self.regressor.predict(inputs)[0]
        sampled=value+noise*xp.asarray(self.regressor.metadata["reference_std"])
        force=xp.where(value[:,4:5]>1.,xp.clip(sampled[:,4:5],0.,10.),0.)
        return xp.concatenate((sampled[:,:4],force),axis=-1)
