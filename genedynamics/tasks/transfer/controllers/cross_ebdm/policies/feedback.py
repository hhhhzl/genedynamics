"""Public geometric feedback shared by source-prior actors and a control reference.

No executor, support witness or actuator/controller carry enters this actor.
Hybrid mode keeps load enabled throughout insertion. Position-admittance mode
uses measured force to regulate the position channel throughout insertion;
it never toggles normal mode in response to the evaluator's loss clock.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from genedynamics.tasks.transfer.observation import PublicControllerMemory
from genedynamics.solvers.single.cross_ebdm.learning.learned import FrozenRegressor
from genedynamics.tasks.transfer.behavior import (
    PhaseGaussianPrior,
    reference_commands,
    teacher_reference,
)


@dataclass(frozen=True)
class PublicFeedbackSpec:
    execution_horizon: int = 1
    insertion_force_scale: float = 1.5
    recovery_spatial_stiffness: float = .25
    alignment_spatial_stiffness: float = 1.
    insertion_spatial_stiffness: float = 1.
    normal_control: str = "hybrid"
    normal_admittance_gain: float = .004
    normal_velocity_limit: float = .01
    lift_before_lateral: bool = False

    def __post_init__(self):
        if self.execution_horizon not in (1,10): raise ValueError("declared feedback horizon is1 or10")
        if not .5 <= self.insertion_force_scale <= 2.: raise ValueError("force scale outside feedback support")
        if self.normal_control not in {"hybrid","position_admittance"}:raise ValueError("unknown normal controller")
        if self.normal_control=="position_admittance" and self.execution_horizon!=1:raise ValueError("admittance needs every control tick")
        numbers=[v for v in asdict(self).values() if isinstance(v,(float,int))]
        if not np.isfinite(numbers).all() or not 0<self.normal_admittance_gain<=.01 or not 0<self.normal_velocity_limit<=.02 or any(not -1 <= x <= 1 for x in
                (self.recovery_spatial_stiffness,self.alignment_spatial_stiffness,self.insertion_spatial_stiffness)):
            raise ValueError("feedback stiffness outside shared action support")


class PublicFeedbackPolicy:
    """One feedback decoder; reference is either frozen source prior or geometry.

    Prior draws retain the ten-tick source clock. execution_horizon=1 refreshes
    actual sensor feedback each20ms without increasing source sampling rate.
    Geometry mode is a named ordinary control reference, not a learned baseline.
    """
    def __init__(self,spec,*,feedback=None,prior=None,transport=None,modes=(0,),seed=0):
        if spec.alias != "T3": raise ValueError("public contact feedback qualifies T3 only")
        if prior is not None and not isinstance(prior,PhaseGaussianPrior): raise TypeError("frozen phase prior required")
        self.spec,self.feedback,self.prior,self.transport=spec,feedback or PublicFeedbackSpec(),prior,transport
        self.reset(modes=modes,seed=seed)

    @classmethod
    def load(cls,spec,prior_path=None,**kwargs):
        prior=None if prior_path is None else PhaseGaussianPrior(FrozenRegressor.load(prior_path,backend="native",device="cpu"))
        return cls(spec,prior=prior,**kwargs)

    def reset(self,*,modes=None,seed=None):
        modes=getattr(self,"_reset_modes",(0,)) if modes is None else modes
        seed=getattr(self,"_reset_seed",0) if seed is None else seed
        self._reset_modes,self._reset_seed=tuple(modes),int(seed)
        self.observer=PublicControllerMemory(self.spec,modes)
        self.rng=np.random.default_rng(seed)
        self.reference=None; self.reference_tick=None; self.commands=[]
        self.initial_position=None;self.lateral_complete=np.zeros(len(modes),dtype=bool)

    def get_state(self):
        return deepcopy(dict(observer=self.observer.get_state(),rng=self.rng.bit_generator.state,
            reference=self.reference,reference_tick=self.reference_tick,commands=self.commands,
            initial_position=self.initial_position,lateral_complete=self.lateral_complete))

    def restore(self,state):
        self.observer.restore(state["observer"]); self.rng.bit_generator.state=deepcopy(state["rng"])
        self.reference=deepcopy(state["reference"]); self.reference_tick=deepcopy(state["reference_tick"])
        self.commands=deepcopy(state["commands"])
        self.initial_position=deepcopy(state["initial_position"]);self.lateral_complete=deepcopy(state["lateral_complete"])

    def reorder(self,indices):
        indices=np.asarray(indices,dtype=int); self.observer.reorder(indices)
        if self.reference is not None:self.reference=self.reference[indices].copy()
        if self.reference_tick is not None:self.reference_tick=self.reference_tick[indices].copy()
        self.commands=[row[indices].copy() for row in self.commands]
        if self.initial_position is not None:self.initial_position=self.initial_position[indices].copy()
        self.lateral_complete=self.lateral_complete[indices].copy()

    def act(self,public):
        self.observer.update(public); memory=self.observer.memory
        tick=self.observer.last_tick
        if self.initial_position is None:self.initial_position=np.array(public["tcp_position"],copy=True)
        if self.reference is None or np.any(tick//10 != self.reference_tick//10):
            if self.prior is None:
                self.reference=teacher_reference(memory,goal_z=self.spec.config.goal[2])
            else:
                noise=self.rng.standard_normal((len(memory.phase),5)).astype(np.float32)
                self.reference=self.prior.reference(memory,noise)
            self.reference_tick=tick.copy()
        if not self.commands:
            reference=self.reference.copy()
            if self.feedback.lift_before_lateral:
                p=np.asarray(public["tcp_position"])
                self.lateral_complete|=p[:,1]<self.initial_position[:,1]-.006
                recovering=memory.phase==1
                lifting=recovering&(p[:,2]<.602)
                reference[:,:2]=np.where(lifting[:,None],self.initial_position[:,:2],reference[:,:2])
                # Pause at a known clear height until the lateral route is
                # physically distinct, before completing the12mm withdrawal.
                lateral=recovering&(memory.mode==1)&~self.lateral_complete&~lifting
                reference[:,2]=np.where(lateral,.603,reference[:,2])
            commands=reference_commands(reference,public["reference_position"],public["reference_rotation"],
                public["tcp_position"],public["tcp_rotation"],integral_bias=memory.tracking_integral,
                horizon=self.feedback.execution_horizon,dt=self.spec.config.control_dt)
            phase=memory.phase
            spatial=np.where(phase<=1,self.feedback.recovery_spatial_stiffness,
                np.where(phase==2,self.feedback.alignment_spatial_stiffness,self.feedback.insertion_spatial_stiffness))
            commands[...,6:12]+=spatial[:,None,None]*np.asarray([1.,1.,0.,1.,1.,1.])
            commands[...,12]*=np.where(phase==3,self.feedback.insertion_force_scale,1.)[:,None]
            if self.transport is not None: commands=self.transport.apply(commands,phase=phase)
            if self.feedback.normal_control=="position_admittance":
                inserted=phase==3
                # The prior/transport force is a setpoint, not discarded.
                # Positive compression below setpoint commands downward motion.
                velocity=np.clip(self.feedback.normal_admittance_gain*(np.asarray(public["measured_wrench"])[:,2,None]-commands[...,12]),
                    -self.feedback.normal_velocity_limit,self.feedback.normal_velocity_limit)
                commands[...,2]=np.where(inserted[:,None],velocity,commands[...,2])
                commands[...,12]=np.where(inserted[:,None],0.,commands[...,12])
            self.commands=[row.copy().astype(np.float32) for row in commands.swapaxes(0,1)]
        return self.commands.pop(0)
