"""Frozen normalized model and mode-conditioned Gaussian skill prior."""
from dataclasses import asdict
import numpy as np
from .core.artifacts import save_bundle,load_bundle
from .models import LearnedModel,ModelSpec
from .learning_data import Normalization


class FrozenRegressor:
    def __init__(self,model,parameters,inputs,outputs,metadata=None):
        self.model,self.parameters=model,parameters
        self.inputs,self.outputs=inputs,outputs
        self.metadata=metadata or {}
        if model.backend=="jax":
            import jax
            self.ic,self.iscale,self.oc,self.oscale=jax.device_put((inputs.center,inputs.scale,outputs.center,outputs.scale),model._impl.device)
        elif model.backend=="torch":
            self.ic,self.iscale,self.oc,self.oscale=(model._impl._tensor(x) for x in (inputs.center,inputs.scale,outputs.center,outputs.scale))
        else:
            self.ic,self.iscale,self.oc,self.oscale=inputs.center,inputs.scale,outputs.center,outputs.scale

    def predict(self,inputs):
        return self.model.predict(self.parameters,(inputs-self.ic)/self.iscale)*self.oscale+self.oc

    def save(self,path,**extra):
        tensors={f"layer.{i}.{name}":x for i,layer in enumerate(self.parameters) for name,x in zip(("weight","bias"),layer)}
        tensors=self.model._impl.export_tensors(tensors)
        tensors.update({"normalization.input_center":self.inputs.center,"normalization.input_scale":self.inputs.scale,
                        "normalization.output_center":self.outputs.center,"normalization.output_scale":self.outputs.scale})
        save_bundle(path,tensors,{**self.metadata,**extra,"artifact_kind":"frozen_regressor","activation":"tanh",
                                  "model_spec":asdict(self.model.spec)})

    @classmethod
    def load(cls,path,*,backend="jax",device="cpu"):
        tensors,metadata=load_bundle(path)
        if metadata.get("artifact_kind")!="frozen_regressor" or metadata.get("activation")!="tanh":
            raise ValueError("unsupported regressor artifact")
        values=dict(metadata["model_spec"]); values["hidden"]=tuple(values["hidden"])
        model=LearnedModel(ModelSpec(**values),backend=backend,device=device)
        parameters=model.import_parameters({k:v for k,v in tensors.items() if k.startswith("layer.")})
        norm=lambda name: Normalization(tensors[f"normalization.{name}_center"],tensors[f"normalization.{name}_scale"])
        for name,dimension in (("input",model.spec.input_dim),("output",model.spec.output_dim)):
            if norm(name).center.shape!=(dimension,) or norm(name).scale.shape!=(dimension,):
                raise ValueError("normalization/model dimension mismatch")
        if any(np.any(norm(n).scale<=0) for n in ("input","output")):
            raise ValueError("normalization scales must be positive")
        return cls(model,parameters,norm("input"),norm("output"),metadata)


class GaussianSkillPrior:
    """Persist one sampled mode across an episode; noise varies by skill chunk.

    Mode frequencies are learned from eligible source episodes, while the MLP
    conditions the trajectory on physical source-query semantics and mode.
    This small baseline does not claim a conditional diffusion architecture.
    """
    def __init__(self,regressor):
        if regressor.metadata.get("learned_role")!="source_prior":
            raise ValueError("expected source prior")
        self.regressor=regressor
        meta=regressor.metadata
        self.horizon=meta["horizon"]
        self.command_dim=meta.get("command_dim",12)
        width=self.horizon*self.command_dim
        expected=width+(self.horizon*3 if "reference_feedback_gain" in meta else 0)
        if self.horizon<1 or self.command_dim not in (12,13) or regressor.model.spec.output_dim!=expected:
            raise ValueError("prior output/skill layout mismatch")
        probabilities=np.asarray(meta["mode_probabilities"])
        if probabilities.shape!=(2,) or np.any(probabilities<0) or not np.isclose(probabilities.sum(),1.):
            raise ValueError("prior mode probabilities must define two normalized modes")
        if len(meta["gaussian_std"])!=width or np.any(np.asarray(meta["gaussian_std"])<0): raise ValueError("invalid prior noise scales")
        if len(meta["command_bounds"])<self.command_dim or np.any(np.asarray(meta["command_bounds"][:self.command_dim])<=0):
            raise ValueError("invalid active prior bounds")
        if regressor.model.backend=="jax":
            import jax.numpy as xp
        elif regressor.model.backend=="torch":
            import torch as xp
        else:
            xp=np
        self.xp=xp
        convert=(lambda v: xp.asarray(v,dtype=xp.float32)) if regressor.model.backend!="torch" else regressor.model._impl._tensor
        self.std=convert(meta["gaussian_std"])
        self.bounds=convert(meta["command_bounds"])
        self.probabilities=convert(meta["mode_probabilities"])

    def sample(self,condition,mode,noise):
        xp=self.xp
        if self.regressor.model.backend in {"jax","native"}:
            onehot=xp.eye(2,dtype=xp.float32)[mode]
            masked=condition
            if "reference_feedback_gain" in self.regressor.metadata:
                if self.regressor.model.backend=="jax": masked=condition.at[:,:18].set(0.)
                else: masked=condition.copy(); masked[:,:18]=0.
            prediction=self.regressor.predict(xp.concatenate((masked,onehot),axis=-1))[0]
            width=self.horizon*self.command_dim
            value=prediction[:,:width]+self.std*noise.reshape(-1,width)
            value=xp.clip(value,-1.,1.).reshape(-1,self.horizon,self.command_dim)*self.bounds[:self.command_dim]
            if "reference_feedback_gain" in self.regressor.metadata:
                reference=prediction[:,width:].reshape(-1,self.horizon,3)
                gain=self.regressor.metadata["reference_feedback_gain"]; dt=self.regressor.metadata["control_dt"]
                def step(position,values):
                    command,desired=values
                    velocity=xp.clip(command[:,:3]+gain*(desired-position),-self.bounds[:3],self.bounds[:3])
                    return position+velocity*dt,command.at[:,:3].set(velocity)
                if self.regressor.model.backend=="jax":
                    import jax
                    _,value=jax.lax.scan(step,condition[:,:3],(xp.swapaxes(value,0,1),xp.swapaxes(reference,0,1)))
                    value=xp.swapaxes(value,0,1)
                else:
                    position=condition[:,:3].copy()
                    for k in range(self.horizon):
                        velocity=xp.clip(value[:,k,:3]+gain*(reference[:,k]-position),-self.bounds[:3],self.bounds[:3])
                        value[:,k,:3]=velocity; position=position+velocity*dt
            if self.command_dim==13:
                enable=prediction[:,:width].reshape(-1,self.horizon,13)[...,12]*self.bounds[12]>1.
                force=xp.where(enable,xp.maximum(value[...,12],0.),0.)
                return xp.concatenate((value[...,:12],force[...,None]),axis=-1)
            return xp.concatenate((value,xp.zeros((*value.shape[:-1],1),dtype=value.dtype)),axis=-1)
        onehot=xp.nn.functional.one_hot(mode.to(xp.int64),2).to(condition.dtype)
        masked=condition.clone()
        if "reference_feedback_gain" in self.regressor.metadata: masked[:,:18]=0.
        prediction=self.regressor.predict(xp.cat((masked,onehot),dim=-1))[0]
        width=self.horizon*self.command_dim
        value=(prediction[:,:width]+self.std*noise.reshape(-1,width)).clamp(-1.,1.).reshape(-1,self.horizon,self.command_dim)*self.bounds[:self.command_dim]
        if "reference_feedback_gain" in self.regressor.metadata:
            reference=prediction[:,width:].reshape(-1,self.horizon,3)
            position=condition[:,:3]; commands=[]
            for k in range(self.horizon):
                velocity=(value[:,k,:3]+self.regressor.metadata["reference_feedback_gain"]*(reference[:,k]-position)).clamp(-self.bounds[:3],self.bounds[:3])
                commands.append(xp.cat((velocity,value[:,k,3:]),dim=-1))
                position=position+velocity*self.regressor.metadata["control_dt"]
            value=xp.stack(commands,dim=1)
        if self.command_dim==13:
            enable=prediction[:,:width].reshape(-1,self.horizon,13)[...,12]*self.bounds[12]>1.
            return xp.cat((value[...,:12],xp.where(enable,value[...,12].clamp(min=0.),0.)[...,None]),dim=-1)
        return xp.cat((value,xp.zeros((*value.shape[:-1],1),device=value.device,dtype=value.dtype)),dim=-1)


def fit_regressor(inputs,targets,train_mask,validation_mask,*,spec,seed=0,updates=1000,batch_size=64,
                  update_chunk=100,metadata=None,checkpoint_path=None,resume=False,backend="jax",device="cpu"):
    from .learning_data import make_batches
    if backend!="jax":raise NotImplementedError("fit_regressor currently requires the JAX training provider")
    model=LearnedModel(spec,backend=backend,device=device)
    ni,no=Normalization.fit(inputs,train_mask),Normalization.fit(targets,train_mask)
    x,y=ni.encode(inputs).astype(np.float32),no.encode(targets).astype(np.float32)
    state=model.initialize(seed)
    import jax
    from pathlib import Path
    import hashlib
    from .checkpoint import save_training_checkpoint,load_training_checkpoint
    from .core.budget import ProbeBudget
    from dataclasses import asdict
    fingerprint=hashlib.sha256()
    for array in (inputs,targets,train_mask,validation_mask):
        value=np.ascontiguousarray(array)
        fingerprint.update(str((value.shape,str(value.dtype))).encode()); fingerprint.update(value.tobytes())
    contract={"data_sha256":fingerprint.hexdigest(),"model_spec":asdict(spec),
              "batch_size":batch_size,"update_chunk":update_chunk,"seed":seed}
    # JSON round trip gives tuples the same canonical representation as saved metadata.
    import json
    contract=json.loads(json.dumps(contract))
    history=[]; start=0
    if resume:
        if checkpoint_path is None: raise ValueError("resume requires checkpoint_path")
        state,_,_,saved=load_training_checkpoint(checkpoint_path,model)
        if saved.get("fit_contract")!=contract: raise ValueError("resume data/model/random contract mismatch")
        start=int(np.asarray(state.step)); history=saved.get("loss_history",[])
        if start>updates or (start%update_chunk and start!=updates):
            raise ValueError("resume total updates must preserve update_chunk boundaries")
    elif checkpoint_path is not None and (Path(checkpoint_path)/"metadata.json").exists():
        raise FileExistsError("checkpoint already exists; pass resume=True or use a fresh output")
    for offset in range(start,updates,update_chunk):
        batches=make_batches(x,y,train_mask,updates=min(update_chunk,updates-offset),batch_size=batch_size,seed=seed+offset)
        state,loss=model.train_chunk(state,jax.device_put(batches,model._impl.device))
        model.synchronize(loss)
        history.append(float(np.asarray(loss)[-1]))
        if checkpoint_path is not None:
            count=offset+min(update_chunk,updates-offset)
            save_training_checkpoint(checkpoint_path,state,context=np.zeros(3,np.float32),multipliers=np.zeros(0,np.float32),
                rng_seed=seed,rng_counter=count,budget=ProbeBudget(0),
                metadata={"fit_contract":contract,"loss_history":history,"learned_metadata":metadata or {}})
    regressor=FrozenRegressor(model,state.parameters,ni,no,metadata)
    validation=np.flatnonzero(validation_mask)
    if not len(validation):
        raise ValueError("held-out validation examples required")
    prediction=np.asarray(regressor.predict(jax.device_put(inputs[validation],model._impl.device)))
    report={"updates":updates,"train_examples":int(np.sum(train_mask)),"validation_examples":len(validation),
            "normalized_train_loss":history,"validation_mse":float(np.mean((prediction-targets[validation])**2)),
            "validation_normalized_mse":float(np.mean(((prediction-targets[validation])/no.scale)**2)),
            "validation_ensemble_disagreement":float(np.mean(np.std(prediction,axis=0))),
            "backend":backend,"device":str(model._impl.device),
            "scope":"functional fit; no convergence or calibrated uncertainty claim"}
    return regressor,state,report
