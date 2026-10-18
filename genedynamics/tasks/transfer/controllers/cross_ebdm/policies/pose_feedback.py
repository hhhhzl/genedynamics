"""A separately versioned public pose decoder, preserving the frozen v1 actor.

The pose servo uses observed TCP pose/twist and the actor's reference. All
methods using this decoder receive the same six-dimensional control support.
Force is still a setpoint; an insertion z residual now acts through admittance.
"""
from dataclasses import dataclass, replace

import numpy as np

from genedynamics.core.control.transfer import orientation_error
from genedynamics.solvers.single.cross_ebdm.learning.learned import FrozenRegressor
from genedynamics.solvers.single.cross_ebdm.learning.learning_data import Normalization
from genedynamics.solvers.single.cross_ebdm.learning.models import LearnedModel, ModelSpec
from genedynamics.solvers.single.cross_ebdm.learning.reference_transport import ReferenceTransport
from .feedback import PublicFeedbackPolicy, PublicFeedbackSpec


@dataclass(frozen=True)
class PublicPoseFeedbackSpec(PublicFeedbackSpec):
    normal_control: str = "position_admittance"
    position_gain: float = 4.
    rotation_gain: float = 4.
    issued_reference_gain: float = 0.
    integral_scale: float = 0.
    twist_damping: float = .5

    def __post_init__(self):
        super().__post_init__()
        if self.execution_horizon != 1:
            raise ValueError("pose feedback requires each observed control tick")
        values=(self.position_gain,self.rotation_gain,self.issued_reference_gain,self.integral_scale,self.twist_damping)
        if not np.isfinite(values).all() or not (0 < self.position_gain <= 12 and 0 < self.rotation_gain <= 12
                and 0 <= self.issued_reference_gain <= 2 and 0 <= self.integral_scale <= 1 and 0 <= self.twist_damping <= 2):
            raise ValueError("pose gains outside declared public support")


class PublicPoseFeedbackPolicy(PublicFeedbackPolicy):
    """Public v2 servo in phases2/3; the original recovery/source draw is kept."""
    def __init__(self,spec,*,feedback=None,prior=None,transport=None,modes=(0,),seed=0):
        self.pose_feedback=feedback or PublicPoseFeedbackSpec(normal_control="position_admittance")
        if not isinstance(self.pose_feedback,PublicPoseFeedbackSpec):
            raise TypeError("explicit pose feedback specification required")
        self.pose_transport=transport
        # V1 produces the shared recovery and spatial stiffness, with force
        # still present. Apply the new residual before the admittance below.
        super().__init__(spec,feedback=replace(self.pose_feedback,normal_control="hybrid"),
            prior=prior,transport=None,modes=modes,seed=seed)

    def act(self,public):
        command=super().act(public).copy()
        phase=self.observer.memory.phase;selected=phase>=2;fb=self.pose_feedback
        reference=self.reference;desired=np.zeros((len(phase),3,3))
        c,s=np.cos(reference[:,3]),np.sin(reference[:,3])
        desired[:,0,0]=c;desired[:,0,1]=s;desired[:,1,0]=s;desired[:,1,1]=-c;desired[:,2,2]=-1.
        angular=np.stack([orientation_error(np,r,d) for r,d in zip(public["tcp_rotation"],desired)])
        issued_angular=np.stack([orientation_error(np,r,d) for r,d in zip(public["reference_rotation"],desired)])
        error=np.concatenate((reference[:,:3]-public["tcp_position"],angular),axis=-1)
        issued=np.concatenate((reference[:,:3]-public["reference_position"],issued_angular),axis=-1)
        gains=np.asarray([fb.position_gain]*3+[fb.rotation_gain]*3)
        servo=(gains*error+fb.issued_reference_gain*issued+fb.integral_scale*self.observer.memory.tracking_integral
            -fb.twist_damping*np.asarray(public["tcp_twist"]))
        command[:,:6]=np.where(selected[:,None],servo,command[:,:6])
        # Normal residual is evaluated separately from the force setpoint.
        # Raw pose z is not added during insertion: it targets the seat pose,
        # whereas the admittance itself regulates measured contact force.
        command[:,2]=np.where(phase==3,0.,command[:,2])
        if self.pose_transport is not None:
            command=self.pose_transport.apply(command[:,None],phase=phase)[:,0]
        if fb.normal_control=="position_admittance":
            normal=fb.normal_admittance_gain*(np.asarray(public["measured_wrench"])[:,2]-command[:,12])+command[:,2]
            command[:,2]=np.where(phase==3,np.clip(normal,-fb.normal_velocity_limit,fb.normal_velocity_limit),command[:,2])
            command[:,12]=np.where(phase==3,0.,command[:,12])
        command[:,:6]=np.clip(command[:,:6],[-.02]*3+[-.12]*3,[.02]*3+[.12]*3)
        return command.astype(np.float32)


def pose_transport(*,force_scale=1.,alignment_bias=(0.,)*6,insertion_bias=(0.,)*6,stiffness_offset=0.):
    """Native affine16→13 model; phase-gated translation/rotation residuals."""
    a,b=np.asarray(alignment_bias),np.asarray(insertion_bias)
    if a.shape!=(6,) or b.shape!=(6,) or not np.isfinite(np.r_[a,b,force_scale,stiffness_offset]).all():
        raise ValueError("finite six-dimensional residuals required")
    if not .5<=force_scale<=2.5 or abs(stiffness_offset)>1 or np.any(np.abs(a)>[.02]*3+[.12]*3) or np.any(np.abs(b)>[.02]*3+[.12]*3):
        raise ValueError("pose residual outside shared command support")
    model=LearnedModel(ModelSpec(16,13,hidden=()),backend="native",device="cpu")
    weights=np.zeros((16,13),np.float32);weights[:13]=np.eye(13);weights[13,12]=force_scale-1
    weights[14,:6]=a;weights[15,:6]=b
    weights[14,6:12]=weights[15,6:12]=stiffness_offset*np.array([1,1,0,1,1,1])
    parameters=model.import_parameters({"layer.0.weight":weights[None],"layer.0.bias":np.zeros((1,13),np.float32)})
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    return ReferenceTransport(FrozenRegressor(model,parameters,norm(16),norm(13),dict(
        learned_role="reference_transport",phase_conditioned=True,alignment_stage=True,force_scale=force_scale,
        alignment_bias6=a.tolist(),insertion_bias6=b.tolist(),stiffness_offset=stiffness_offset,
        support="phase2/3 six-dimensional residuals; phase3 z enters force admittance; unchanged physical bounds")))
