"""Training resume artifacts are separate from frozen inference bundles."""
from .artifact_bundle import save_bundle,load_bundle
from .models import TrainState


def save_training_checkpoint(path,state,*,context,multipliers,rng_seed,rng_counter,budget,metadata=None):
    tensors={}
    for group in ("parameters","first","second"):
        for i,layer in enumerate(getattr(state,group)):
            for name,value in zip(("weight","bias"),layer):
                tensors[f"{group}.{i}.{name}"]=value
    tensors.update({"step":state.step,"context":context,"multipliers":multipliers})
    # Persistence is an explicit host boundary, including autograd tensors.
    tensors={k:(v.detach().cpu().numpy() if hasattr(v,"detach") else v) for k,v in tensors.items()}
    save_bundle(path,tensors,{**(metadata or {}),"artifact_kind":"training_resume","resume_schema":1,
        "rng":{"seed":rng_seed,"counter":rng_counter},"probe_budget":budget.snapshot()})


def load_training_checkpoint(path,model):
    tensors,metadata=load_bundle(path)
    if metadata.get("artifact_kind")!="training_resume" or metadata.get("resume_schema")!=1:
        raise ValueError("expected training resume checkpoint")
    groups=[]
    for group in ("parameters","first","second"):
        layers={"layer."+k[len(group)+1:]:v for k,v in tensors.items() if k.startswith(group+".")}
        groups.append(model.import_parameters(layers))
    if model.backend=="jax":
        import jax.numpy as xp
        step=xp.asarray(tensors["step"],dtype=xp.int32)
    else:
        import torch
        step=torch.as_tensor(tensors["step"],device=model._impl.device,dtype=torch.int64)
    return TrainState(*groups,step),tensors["context"],tensors["multipliers"],metadata
