"""Train actual source-only stochastic prior and public response ensemble on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.data import load_dataset
from genedynamics.learning.cross_embodiment.learning_data import prior_condition,response_tables
from genedynamics.learning.cross_embodiment.learned import fit_regressor
from genedynamics.learning.cross_embodiment.models import ModelSpec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("tmp/cross_embodiment/obstacle_transfer_source_numpy"))
    parser.add_argument("--response",type=Path,default=Path("tmp/cross_embodiment/obstacle_transfer_response_full"))
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/learned_cpu"))
    parser.add_argument("--updates",type=int,default=2000)
    parser.add_argument("--excitations",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    data,meta=load_dataset(args.source)
    if meta["role"]!="source_demonstrations":
        raise ValueError("source prior must train only from nominal source demonstrations")
    condition=prior_condition(data,meta["clocks"]["control_dt"])
    # Learn the source temporal reference law; actual source-query pose enters
    # a feedback decoder. This prevents fitting spurious pose deviations from
    # just four teacher episodes and then extrapolating through them online.
    condition[...,:18]=0.
    b,k,d=condition.shape
    mode=data["episode.observed_mode"]
    mode_code=np.eye(2,dtype=np.float32)[np.clip(mode,0,1)]
    inputs=np.concatenate((condition,np.broadcast_to(mode_code[:,None],(b,k,2))),axis=-1).reshape(b*k,-1)
    bounds=np.array([.12]*3+[.6]*3+[1.]*6+[0.],np.float32)
    commands=(data["command.raw"][...,:12]/bounds[:12]).reshape(b*k,-1)
    reference=data["chunk.controller.reference_position"][:,:,None,:]+np.cumsum(
        data["command.applied"][...,:3]*meta["clocks"]["control_dt"],axis=2)
    targets=np.concatenate((commands,reference.reshape(b*k,-1)),axis=-1)
    train=data["prior.training_mask"].all(axis=-1).ravel()
    validation=data["prior.validation_mask"].all(axis=-1).ravel()
    counts=np.bincount(mode[np.any(data["prior.training_mask"],axis=(1,2))],minlength=2)
    if np.any(counts==0):
        raise ValueError("both independently observed source route modes required")
    regressor,state,report=fit_regressor(inputs,targets,train,validation,
        spec=ModelSpec(inputs.shape[-1],targets.shape[-1],hidden=(96,96),learning_rate=.002),
        updates=args.updates,checkpoint_path=args.output/"source_resume",resume=args.resume,device=args.device,
        metadata={"learned_role":"source_prior","horizon":meta["clocks"]["L"],
        "command_bounds":bounds.tolist(),"mode_probabilities":(counts/counts.sum()).tolist(),
        "source_dataset_sha256":meta["tensor_sha256"],"reference_feedback_gain":3.,
        "control_dt":meta["clocks"]["control_dt"],"condition":"goal, elapsed, source mode; source-query pose feedback"})
    residual=targets[train,:commands.shape[-1]]-np.asarray(regressor.predict(inputs[train]))[0,:,:commands.shape[-1]]
    regressor.metadata["gaussian_std"]=np.clip(residual.std(axis=0),.001,.03).tolist()
    regressor.save(args.output/"source_prior",fit_report=report)
    reports={"source_prior":report}
    data,meta=load_dataset(args.response)
    x,y,train,validation,layout=response_tables(data,meta)
    if args.excitations:
        extra,extra_meta=load_dataset(args.excitations)
        a,b,c,d,extra_layout=response_tables(extra,extra_meta)
        if extra_layout["carry_dim"]!=layout["carry_dim"]: raise ValueError("response carry layouts differ")
        x,y,train,validation=(np.concatenate((v,w),axis=0) for v,w in zip((x,y,train,validation),(a,b,c,d)))
    response,state,report=fit_regressor(x,y,train,validation,
        spec=ModelSpec(x.shape[-1],y.shape[-1],hidden=(96,96),ensemble=3,learning_rate=.001),
        updates=args.updates,checkpoint_path=args.output/"response_resume",resume=args.resume,device=args.device,
        metadata={"learned_role":"response_ensemble","layout":layout,
        "response_dataset_sha256":meta["tensor_sha256"],"uncertainty":"ensemble disagreement; uncalibrated",
        "held_out":"episode groups; mechanism values shared across splits"})
    response.save(args.output/"response",fit_report=report)
    reports["response"]=report
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"training_report.json").write_text(json.dumps(reports,indent=2)+"\n")
    print(json.dumps(reports,indent=2))


if __name__=="__main__":
    main()
