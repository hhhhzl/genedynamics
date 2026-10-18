"""DIAL-MPC sampling/annealing over the frozen public response model.

This is a task adapter around the repository's source-ported DIAL primitives:
quadratic node splines, incumbent sampling, self-normalized MPPI weighting,
geometric noise annealing, and shifted receding-horizon warm starts.  The
public response ensemble supplies the model rollout objective; physical task
state, evaluator truth, and target labels never enter candidate rollouts.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.solvers.single.cross_ebdm.learning.task_response import task_amplitude
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import (
    BOUNDS,decode_pose_transport,task_proxy,
)
from genedynamics.tasks.transfer.controllers.cross_ebdm.arm_dense import PublicArmTaskDenseActor
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import PublicDenseTransportActor
from genedynamics.tasks.transfer.controllers.cross_ebdm.mobile_dense import PublicLivenessMobileDenseTransportActor


class TorchDIALPlanner:
    """Exact DIAL node update in Torch for a Torch response-model rollout."""
    VERSION="public-response-DIAL-node-planner-v1"

    def __init__(self,*,device="cpu",seed=0,n_samples=2048,h_node=4,h_sample=16,
                 n_diffuse=2,n_diffuse_init=10,temp_sample=.06,
                 horizon_diffuse_factor=.9,traj_diffuse_factor=.5,
                 sigma_scale=1.,action_limit=1.):
        if (h_node,h_sample)!=(4,16) or n_samples!=2048 or n_diffuse!=2 or n_diffuse_init!=10:
            raise ValueError("frozen source-ported DIAL opportunity required")
        self.device=torch.device(device);self.n_samples=n_samples
        self.n_diffuse=n_diffuse;self.n_diffuse_init=n_diffuse_init
        self.temp_sample=float(temp_sample);self.traj_diffuse_factor=float(traj_diffuse_factor)
        self.action_limit=float(action_limit)
        cache=Path(__file__).resolve().parents[4]/"solvers"/"single"/"dial"/"spline_cache.npz"
        with np.load(cache) as archive:
            self.n2u=torch.as_tensor(archive["n2u_4_16"],device=self.device,dtype=torch.float32)
            self.u2n=torch.as_tensor(archive["u2n_4_16"],device=self.device,dtype=torch.float32)
        base=(float(horizon_diffuse_factor)**torch.arange(
            h_node,-1,-1,device=self.device,dtype=torch.float32))*float(sigma_scale)
        self.sigma_control=base
        self.generator=torch.Generator(device=self.device);self.generator.manual_seed(int(seed))
        self.reset()

    def reset(self):
        self.nodes=torch.zeros((5,13),device=self.device)
        self.cold=True;self.replans=0;self.candidate_rollouts=0

    def state(self):
        return {"version":self.VERSION,"nodes":self.nodes.detach().cpu(),
                "cold":self.cold,"replans":self.replans,
                "candidate_rollouts":self.candidate_rollouts,
                "rng":self.generator.get_state().cpu()}

    def restore(self,state):
        if state["version"] != self.VERSION: raise ValueError("DIAL state version changed")
        self.nodes=state["nodes"].to(self.device).clone();self.cold=bool(state["cold"])
        self.replans=int(state["replans"]);self.candidate_rollouts=int(state["candidate_rollouts"])
        self.generator.set_state(state["rng"].cpu())

    def dense(self,nodes):
        return torch.einsum("hn,bnu->bhu",self.n2u,nodes) if nodes.ndim==3 else self.n2u@nodes

    def replan(self,cost):
        steps=self.n_diffuse_init if self.cold else self.n_diffuse
        for step in range(steps):
            scale=self.sigma_control*(self.traj_diffuse_factor**step)
            noise=torch.randn((self.n_samples,5,13),generator=self.generator,
                              device=self.device,dtype=torch.float32)
            candidates=(self.nodes[None]+noise*scale[None,:,None]).clamp(
                -self.action_limit,self.action_limit)
            candidates[:,0]=self.nodes[0]
            candidates=torch.cat((candidates,self.nodes[None]),dim=0)
            values=cost(self.dense(candidates))
            if values.shape != (self.n_samples+1,) or not bool(torch.isfinite(values).all()):
                raise FloatingPointError("one finite DIAL cost per candidate required")
            rewards=-values;incumbent=rewards[-1];standard=rewards.std(unbiased=False)
            standard=torch.where(standard<1e-6,standard.new_ones(()),standard)
            weights=torch.softmax((rewards-incumbent)/(standard*self.temp_sample),dim=0)
            self.nodes=torch.einsum("b,bnu->nu",weights,candidates)
            self.candidate_rollouts+=self.n_samples+1
        action=self.dense(self.nodes)[0].clone()
        dense=self.dense(self.nodes);shifted=torch.roll(dense,-1,dims=0)
        shifted[-1]=0.;self.nodes=self.u2n@shifted
        self.cold=False;self.replans+=1
        return action


class _DIALActorMixin:
    def __init__(self,*args,response,seed=0,n_samples=2048,**kwargs):
        spec=args[0] if args else kwargs.get("spec")
        if spec is None:
            raise TypeError("DIAL actor requires an explicit task spec")
        base_kwargs=dict(kwargs)
        if spec.alias not in {"T1","T2"}:
            base_kwargs["seed"]=seed
        super().__init__(*args,**base_kwargs)
        self.response=response.to(self.device).eval()
        for parameter in self.response.parameters(): parameter.requires_grad_(False)
        self.dial=TorchDIALPlanner(device=self.device,seed=seed,n_samples=n_samples)
        self.irreversible_model_calls=0;self.irreversible_candidate_rollouts=0
        self.inference_seconds=[]

    def reset(self):
        super().reset()
        if hasattr(self,"dial"): self.dial.reset()
        self.irreversible_model_calls=0;self.irreversible_candidate_rollouts=0
        self.inference_seconds=[]

    def get_state(self):
        return {"actor":super().get_state(),"dial":self.dial.state(),
                "model_calls":self.irreversible_model_calls,
                "candidate_rollouts":self.irreversible_candidate_rollouts,
                "inference_seconds":list(self.inference_seconds)}

    def restore(self,state):
        super().restore(state["actor"]);self.dial.restore(state["dial"])
        self.irreversible_model_calls=int(state["model_calls"])
        self.irreversible_candidate_rollouts=int(state["candidate_rollouts"])
        self.inference_seconds=list(state["inference_seconds"])

    def _run_dial(self,cost):
        before=self.dial.candidate_rollouts
        value=self.dial.replan(cost)
        self.irreversible_model_calls+=self.dial.n_diffuse_init if self.dial.replans==1 else self.dial.n_diffuse
        self.irreversible_candidate_rollouts+=self.dial.candidate_rollouts-before
        return value


class PublicArmTaskDIALMPCActor(_DIALActorMixin,PublicArmTaskDenseActor):
    VERSION="public-T1-T2-response-DIAL-MPC-v1"

    def __init__(self,*args,response,mode,seed=0,**kwargs):
        super().__init__(*args,response=response,mode=mode,seed=seed,**kwargs)

    def act(self,public):
        begun=time.perf_counter()
        tick=int(np.rint(np.asarray(public["elapsed_seconds"]).item()/self.spec.config.control_dt))
        if tick==self.last_action_tick:return self.cached_action.copy()
        batch,base_action=self.prepare(public)
        if batch["source_query_distance"][0]>self.maximum_query_distance:
            self.fallback_controls+=1;self.cached_action=base_action.copy();self.last_action_tick=tick
            self.inference_seconds.append(time.perf_counter()-begun);return self.cached_action.copy()
        feature=torch.as_tensor(batch["target_features"],device=self.device,dtype=torch.float32)
        base=torch.as_tensor(batch["base_proto"],device=self.device,dtype=torch.float32)
        amplitude=task_amplitude(feature)[:,None]
        current={key:torch.as_tensor(value,device=self.device,dtype=torch.float32)
                 for key,value in batch["current"].items()}
        reference=torch.as_tensor(batch["task_reference"],device=self.device,dtype=torch.float32)
        with torch.inference_mode():
            def cost(residual):
                count=len(residual);commands=base+.025*amplitude*residual[:,:self.response.spec.horizon]
                commands=torch.minimum(torch.maximum(commands,-amplitude),amplitude)
                force=torch.where(base[...,12:13]*25>.1,commands[...,12:13].clamp(min=0),
                                  torch.zeros_like(commands[...,12:13]))
                commands=torch.cat((commands[...,:12],force),dim=-1)
                predicted=self.response(feature.expand(count,-1),commands)
                task=task_proxy(predicted,{key:value.expand(count,*value.shape[1:])
                                           for key,value in current.items()},
                                reference.expand(count,-1)).mean(0)
                peaks=torch.expm1(predicted[...,18:20]).clamp(min=0)
                risk=((peaks[...,0]/25.-1).clamp(min=0).square()
                      +(peaks[...,1]/10.-1).clamp(min=0).square()).amax(0)
                change=(commands-base).square().mean((1,2))
                return task+risk+.01*change
            residual=self._run_dial(cost)
            value=base[:,0]+.025*amplitude[:,0]*residual[None]
            value=torch.minimum(torch.maximum(value,-amplitude[:,0]),amplitude[:,0])
            value=torch.cat((value[...,:12],torch.where(base[:,0,12:13]*25>.1,
                value[...,12:13].clamp(min=0),torch.zeros_like(value[...,12:13]))),dim=-1)
            self.cached_action=(value*value.new_tensor(BOUNDS)).cpu().numpy().astype(np.float32)
        self.last_action_tick=tick;self.inference_seconds.append(time.perf_counter()-begun)
        return self.cached_action.copy()


class _ContactDIALMPCActor(_DIALActorMixin):
    VERSION="public-T3-T4-response-DIAL-MPC-v1"

    def _dial_contact_act(self,public):
        begun=time.perf_counter();batch,base_action=self.prepare(public)
        if self.last_query_distance>self.maximum_query_distance:
            self.fallback_controls+=1;self.inference_seconds.append(time.perf_counter()-begun)
            return base_action
        feature=torch.as_tensor(batch["target_features"],device=self.device,dtype=torch.float32)
        base=torch.as_tensor(batch["base_proto"],device=self.device,dtype=torch.float32)
        state={key:torch.as_tensor(value,device=self.device) for key,value in batch["decoder"].items()}
        state={key:value.float() if value.is_floating_point() else value for key,value in state.items()}
        current={key:torch.as_tensor(value,device=self.device,dtype=torch.float32)
                 for key,value in batch["current"].items()}
        waypoint=teacher_reference(self.base.observer.memory,goal_z=self.spec.config.goal[2])
        cosine,sine=np.cos(waypoint[:,3]),np.sin(waypoint[:,3]);rotation=np.zeros((1,3,3))
        rotation[:,0,0]=cosine;rotation[:,0,1]=sine;rotation[:,1,0]=sine
        rotation[:,1,1]=-cosine;rotation[:,2,2]=-1.
        reference=torch.as_tensor(np.concatenate((waypoint[:,:3],rotation.reshape(1,9),
                                                   waypoint[:,4:5]),axis=1),
                                  device=self.device,dtype=torch.float32)
        with torch.inference_mode():
            def cost(residual):
                count=len(residual);normalized=base+.025*residual[:,:self.response.spec.horizon]
                normalized=torch.cat((normalized[...,:12].clamp(-1,1),
                                      normalized[...,12:13].clamp(0,1)),dim=-1)
                expanded={key:value.expand(count,*value.shape[1:]) for key,value in state.items()}
                actual=decode_pose_transport(normalized*normalized.new_tensor(BOUNDS),
                                             expanded,self.feedback)/normalized.new_tensor(BOUNDS)
                predicted=self.response(feature.expand(count,-1),actual)
                task=task_proxy(predicted,{key:value.expand(count,*value.shape[1:])
                                           for key,value in current.items()},
                                reference.expand(count,-1)).mean(0)
                peaks=torch.expm1(predicted[...,18:20]).clamp(min=0)
                risk=((peaks[...,0]/25.-1).clamp(min=0).square()
                      +(peaks[...,1]/10.-1).clamp(min=0).square()).amax(0)
                return task+risk+.01*(normalized-base).square().mean((1,2))
            residual=self._run_dial(cost)
            normalized=base[:,0]+.025*residual[None]
            normalized=torch.cat((normalized[...,:12].clamp(-1,1),
                                  normalized[...,12:13].clamp(0,1)),dim=-1)
            self.interceptor.override=(normalized[:,None]*normalized.new_tensor(BOUNDS)).cpu().numpy()
        value=self.base.act(public);self.inference_seconds.append(time.perf_counter()-begun)
        return value

    def act(self,public): return self._dial_contact_act(public)


class PublicContactDIALMPCActor(_ContactDIALMPCActor,PublicDenseTransportActor):
    """T3 DIAL task adapter."""


class PublicMobileDIALMPCActor(_ContactDIALMPCActor,PublicLivenessMobileDenseTransportActor):
    """T4 DIAL task adapter under the frozen public liveness governor."""
