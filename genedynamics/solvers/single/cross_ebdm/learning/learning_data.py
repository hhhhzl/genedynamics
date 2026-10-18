"""Host-only supervised tables and normalization, separated from device kernels."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Normalization:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, value, mask):
        selected=np.asarray(value)[np.asarray(mask,dtype=bool)]
        if len(selected)<2 or not np.isfinite(selected).all():
            raise ValueError("normalization needs at least two finite TRAIN examples")
        return cls(selected.mean(axis=0).astype(np.float32),
                   np.maximum(selected.std(axis=0),1e-3).astype(np.float32))

    def encode(self,value):
        return (value-self.center)/self.scale

    def decode(self,value):
        return value*self.scale+self.center


def prior_condition(tensors, control_dt):
    count,chunks=tensors["chunk.start_steps"].shape
    flatten=lambda key: tensors[key].reshape(count,chunks,-1)
    goal=lambda key: np.broadcast_to(tensors[key].reshape(count,1,-1),(count,chunks,tensors[key][0].size))
    # Physical semantic pose/twist + task goal + elapsed time; neither body
    # joint coordinates nor mechanism truth nor target controller carry enters.
    return np.concatenate((flatten("chunk.semantic.tcp_position"),flatten("chunk.semantic.tcp_rotation"),
        flatten("chunk.semantic.tcp_twist"),goal("task.goal_position"),goal("task.goal_rotation"),
        tensors["chunk.start_steps"][...,None]*control_dt),axis=-1).astype(np.float32)


def online_prior_condition(semantics, goal, goal_rotation, elapsed, xp):
    b=semantics.tcp_position.shape[0]
    return xp.concatenate((semantics.tcp_position,semantics.tcp_rotation.reshape(b,9),semantics.tcp_twist,
        xp.broadcast_to(goal,(b,3)),xp.broadcast_to(goal_rotation,(b,3,3)).reshape(b,9),
        elapsed.reshape(b,1)),axis=-1)


CARRY_FIELDS=("chunk.body.qpos","chunk.body.qvel","chunk.semantic.tcp_position",
    "chunk.semantic.tcp_rotation","chunk.semantic.tcp_twist","chunk.controller.reference_position",
    "chunk.controller.reference_rotation","chunk.controller.command_buffer",
    "chunk.controller.last_command","chunk.controller.integral_error")


def response_carry(tensors):
    b,k=tensors["chunk.start_steps"].shape
    return np.concatenate([tensors[name].reshape(b,k,-1) for name in CARRY_FIELDS],axis=-1).astype(np.float32)


def online_response_carry(state,semantics,xp):
    b=state.sample_id.shape[0]
    if hasattr(state.physics,"qpos"):
        qpos,qvel=state.physics.qpos,state.physics.qvel
    else:  # Serial MuJoCo diagnostic provider; not a device learner path.
        qpos=xp.stack([x.qpos for x in state.physics]); qvel=xp.stack([x.qvel for x in state.physics])
    fields=(qpos,qvel,semantics.tcp_position,semantics.tcp_rotation,semantics.tcp_twist,
            *state.controller)
    return xp.concatenate([xp.asarray(x).reshape(b,-1) for x in fields],axis=-1)


def response_tables(tensors, metadata):
    """Next *complete* carry plus per-chunk risks; no fixed initial risk reuse.

    Terminal and truncated chunks have no recorded next carry and are excluded
    from this delta table. Their outcomes remain in the original dataset. A
    separate risk-only head is needed before claiming calibrated failure risk.
    """
    if metadata["role"] not in {"response_development","response_validation"}:
        raise ValueError("response pretraining requires public development data")
    carry=response_carry(tensors)
    b,k,d=carry.shape
    command=tensors["command.raw"].reshape(b,k,-1)
    robot=np.eye(2,dtype=np.float32)[tensors["episode.robot"]]
    # Pretraining labels only. Hidden target context never uses these labels.
    truth=tensors["diagnostic.mechanism_truth"]
    context=(truth-np.array([1.,0.,.5],np.float32))/np.array([.25,4.,.5],np.float32)
    inputs=np.concatenate((carry[:,:-1],command[:,:-1],
        np.broadcast_to(robot[:,None],(b,k-1,2)),np.broadcast_to(context[:,None],(b,k-1,3))),axis=-1)
    risks=np.stack((tensors["trace.clearance_lower_bound"][:,:-1].min(axis=-1),
                    tensors["trace.max_joint_speed"][:,:-1].max(axis=-1),
                    tensors["trace.saturation"][:,:-1].max(axis=-1)),axis=-1)
    targets=np.concatenate((carry[:,1:]-carry[:,:-1],risks),axis=-1)
    valid=tensors["trace.valid"][:,:-1].all(axis=-1)&tensors["trace.valid"][:,1:].any(axis=-1)
    valid&=tensors["chunk.feature_finite"][:,:-1]&tensors["chunk.feature_finite"][:,1:]
    split=np.broadcast_to(tensors["episode.split"][:,None],valid.shape)
    if metadata["role"]=="response_validation": split=np.ones_like(split)
    return (inputs.reshape(-1,inputs.shape[-1]).astype(np.float32),
            targets.reshape(-1,targets.shape[-1]).astype(np.float32),
            (valid&(split==0)).ravel(),(valid&(split==1)).ravel(),
            {"carry_dim":d,"command_dim":command.shape[-1],"context_dim":3,
             "semantic_slice":[tensors["chunk.body.qpos"].shape[-1]+tensors["chunk.body.qvel"].shape[-1],
                               tensors["chunk.body.qpos"].shape[-1]+tensors["chunk.body.qvel"].shape[-1]+18],
             "body_feature_dim":2,
             "carry_fields":list(CARRY_FIELDS),"risks":["clearance_censored","max_joint_speed","saturation"],
             "excluded_no_next_carry":int((~valid).sum())})


def make_batches(inputs, targets, mask, *, updates, batch_size, seed):
    from .models import RegressionBatch
    selected=np.flatnonzero(mask)
    if not len(selected):
        raise ValueError("no eligible training examples")
    indices=np.random.default_rng(seed).choice(selected,size=(updates,batch_size),replace=True)
    return RegressionBatch(inputs[indices],targets[indices],np.ones(indices.shape,np.float32))
