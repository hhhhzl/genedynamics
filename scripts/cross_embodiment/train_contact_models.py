"""Shared CPU learner on contact carry; source prior remains nominal Panda only."""
import argparse
from pathlib import Path
import numpy as np
import jax
from genedynamics.learning.cross_embodiment.contact_data import load_contact_dataset,SEMANTIC_SLICE
from genedynamics.learning.cross_embodiment.contact import ContactConfig
from genedynamics.learning.cross_embodiment.learned import fit_regressor
from genedynamics.learning.cross_embodiment.models import ModelSpec
from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--response",type=Path,required=True)
    parser.add_argument("--mobile-response",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--updates",type=int,default=5000)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    data,meta=load_contact_dataset(args.source)
    if meta["role"]!="source_demonstrations": raise ValueError("source role required")
    config=ContactConfig(environment=meta["environment"])
    b,k,h=data["command"].shape[:3]
    start,end=SEMANTIC_SLICE
    condition=np.concatenate((np.zeros((b,k,18)),np.broadcast_to(np.array(config.goal),(b,k,3)),
        np.broadcast_to(rotation().reshape(9),(b,k,9)),np.broadcast_to(np.arange(k)[None,:,None]*h*config.control_dt,(b,k,1)),
        np.broadcast_to(np.eye(2)[data["mode"]][:,None],(b,k,2))),axis=-1).reshape(b*k,-1).astype(np.float32)
    bounds=np.array([.08]*3+[.4]*3+[1.]*6+[25.],np.float32)
    applied=data.get("applied",data["command"])
    reference=data["carry"][:,:,98:101][:,:,None,:]+np.cumsum(applied[...,:3]*config.control_dt,axis=2)
    targets=np.concatenate(((data["command"]/bounds).reshape(b*k,-1),reference.reshape(b*k,-1)),axis=-1)
    eligible=(data["status"]==1)&(data["observed_mode"]==data["mode"])
    mask=data["valid"].all(axis=-1)&eligible[:,None]
    train=(mask&(data["group"][:,None]!=0)).ravel(); validation=(mask&(data["group"][:,None]==0)).ravel()
    if set(data["mode"][eligible])!={0,1}: raise ValueError("two actual source modes required")
    prior,state,report=fit_regressor(condition,targets,train,validation,
        spec=ModelSpec(condition.shape[-1],targets.shape[-1],hidden=(96,96),learning_rate=.001),updates=args.updates,
        checkpoint_path=args.output/"source_resume",resume=args.resume,
        metadata={"learned_role":"source_prior","environment":meta["environment"],"horizon":h,"command_dim":13,
        "command_bounds":bounds.tolist(),"mode_probabilities":[.5,.5],"reference_feedback_gain":1.,"control_dt":config.control_dt,
        "source_dataset_sha256":meta["tensor_sha256"],"eligible_episodes":int(eligible.sum()),"attempted_episodes":b})
    residual=targets[train,:h*13]-np.asarray(prior.predict(jax.numpy.asarray(condition[train])))[0,:,:h*13]
    std=np.clip(residual.std(axis=0),.0001,.01).reshape(h,13); std[:,12]=np.minimum(std[:,12],.0002)
    prior.metadata["gaussian_std"]=std.ravel().tolist(); prior.save(args.output/"source_prior",fit_report=report)
    datasets=[load_contact_dataset(args.response)]
    if args.mobile_response: datasets.append(load_contact_dataset(args.mobile_response))
    tables=[]
    for data,meta in datasets:
        if meta["role"]!="response_development": raise ValueError("public development response role required")
        b,k,h=data["command"].shape[:3]
        body=np.broadcast_to(data["body"][:,None],(b,k,3))
        context=(data["mechanism"]-np.array([1.,0.,.5]))/np.array([.25,4.,.5])
        x=np.concatenate((data["carry"],data["command"].reshape(b,k,-1),body,np.broadcast_to(context[:,None],(b,k,3))),axis=-1)
        y=np.concatenate((data["delta"],data["risk"]),axis=-1)
        valid=data["valid"].any(axis=-1)
        tables.append((x.reshape(b*k,-1),y.reshape(b*k,-1),(valid&(data["group"][:,None]!=0)).ravel(),
                       (valid&(data["group"][:,None]==0)).ravel()))
    x,y,train,validation=(np.concatenate([table[i] for table in tables],axis=0).astype(np.float32 if i<2 else bool) for i in range(4))
    response,state,report=fit_regressor(x,y,train,validation,
        spec=ModelSpec(x.shape[-1],y.shape[-1],hidden=(96,96),ensemble=3,learning_rate=.001),updates=args.updates,
        checkpoint_path=args.output/"response_resume",resume=args.resume,
        metadata={"learned_role":"response_ensemble","environment":config.environment,"risk_kind":"contact",
            "context_bounds":np.stack((x[train,-3:].min(axis=0),x[train,-3:].max(axis=0)),axis=-1).tolist(),
            "active_command_dim":13,"command_bounds":bounds.tolist(),
            "layout":{"carry_dim":data["carry"].shape[-1],"context_dim":3,"command_dim":h*13,"body_feature_dim":3,
                      "semantic_slice":list(SEMANTIC_SLICE),"force_index":data["carry"].shape[-1]-4},
            "response_datasets":[m["tensor_sha256"] for _,m in datasets],"uncertainty":"uncalibrated ensemble disagreement"})
    response.save(args.output/"response",fit_report=report)
    print({"source":prior.metadata,"response_fit":report})


if __name__=="__main__": main()
