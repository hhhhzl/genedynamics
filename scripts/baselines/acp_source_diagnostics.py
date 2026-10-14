"""Offline ACP source sensor/label alignment and sampling/conditioning diagnostics.

All inputs are SHA-qualified saved source episodes; no simulator reset or target
interaction occurs. Ablations diagnose approximation error and do not authorize
deploying a candidate or changing a predeclared source validation gate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from genedynamics.learning.baselines.acp.task_training import SourceDemonstrations,build_policy
from genedynamics.learning.baselines.acp.trainer import deterministic_runtime
from scripts.baselines.acp_native import ROOT,_digest_file,_seed
from scripts.baselines.acp_source_precision import diagnosis,validation


def alignment(paths):
    rows=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as arrays:
            wrench=arrays["public.measured_wrench"][:-1,:3]
            norm=np.linalg.norm(wrench.astype(np.float64),axis=-1)
            native=np.where(norm<.5,5000.,np.where(norm>5,200.,5000.-4800.*(norm-.5)/4.5))
            labels=arrays["acp.stiffness_scalar"]
            rows.append({"episode":str(path),"sha256":_digest_file(path),"commands":len(labels),
                "current_public_force_native_stiffness_max_abs_error":float(np.max(np.abs(native-labels))),
                "one_step_shift_mean_abs_stiffness_error":float(np.mean(np.abs(native[1:]-labels[:-1]))),
                "low_stiffness_commands":int((labels<1000).sum()),"other_commands":int((labels>=1000).sum())})
    return rows


def condition_ablation(policy,data):
    indices=np.unique(np.linspace(0,len(data)-1,16).astype(int))
    obs=data.batch(indices)["obs"]
    reversed_obs={"sparse":{key:value.flip(0) for key,value in obs["sparse"].items()}}
    _seed(18680)
    with torch.inference_mode():real=policy.predict_action(obs)["sparse"].cpu()
    _seed(18680)
    with torch.inference_mode():swapped=policy.predict_action(reversed_obs)["sparse"].cpu()
    # Keep identical sampled noise; only the public conditioning changes.
    return {"source_indices":indices.tolist(),"identical_sampling_noise_seed":18680,
        "reference_first_translation_mean_change_m":float((real[:,0,:3]-swapped[:,0,:3]).norm(dim=-1).mean()),
        "virtual_first_translation_mean_change_m":float((real[:,0,9:12]-swapped[:,0,9:12]).norm(dim=-1).mean()),
        "stiffness_first_mean_change_N_per_m":float((real[:,0,18]-swapped[:,0,18]).abs().mean()),
        "interpretation":"Nonzero change establishes that the actor uses conditioning; it does not establish correct contact conditioning."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.checkpoint=args.checkpoint.resolve()
    config=yaml.safe_load(args.config.read_text())
    output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("diagnostics belong outside latex/")
    output.mkdir(parents=True,exist_ok=False)
    root=ROOT/config["source"].get("data_root",str(args.checkpoint.parent.relative_to(ROOT)))
    paths=[root/f"source_seed_{config['source']['collection_seeds'][i]}/episode.npz" for i in config["source"]["validation_collection_indices"]]
    dataset=SourceDemonstrations(paths,history_len=config["training"]["history_len"],horizon=config["training"]["horizon"])
    deterministic_runtime();torch.set_num_threads(1)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    policy=build_policy(payload["shape_meta"],payload["history_len"])
    policy.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
    policy.to(args.device).eval()
    report={"checkpoint":str(args.checkpoint),"checkpoint_sha256":_digest_file(args.checkpoint),
        "source_label_alignment":alignment(paths),"numerical_interface":diagnosis(policy,dataset),
        "conditioning_ablation":condition_ablation(policy,dataset),"source_sampling_ablations":{},
        "target_resets":0,"target_controls":0,"source_physical_calls":0,
        "scope":"reused development source validation, not final independent test; offline diagnostics only"}
    original=dict(policy.noise_scheduler.config)
    for name,steps,clip in (("native_16_clip",16,True),("diagnostic_50_clip",50,True),("diagnostic_16_no_clip",16,False)):
        policy.num_inference_steps=steps
        policy.noise_scheduler.register_to_config(clip_sample=clip)
        result=validation(policy,dataset,windows=48,repeats=2,prefix=4)
        report["source_sampling_ablations"][name]=result
        (output/"diagnostics_progress.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"variant":name,"metrics":result["metrics"]}),flush=True)
    policy.num_inference_steps=16;policy.noise_scheduler.register_to_config(clip_sample=original["clip_sample"])
    report["source_sha256"]=_digest_file(Path(__file__))
    (output/"diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
