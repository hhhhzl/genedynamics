"""Read-only paired native K/VT wrench and loading-intent audit on real source windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from genedynamics.learning.baselines.acp.task_training import SourceDemonstrations,build_policy
from genedynamics.learning.baselines.acp.trainer import deterministic_runtime
from genedynamics.learning.cross_embodiment.core.control_math import orientation_error
from genedynamics.solvers.single.acp import spatial
from genedynamics.solvers.single.acp.compliance import decode_stiffness_tool
from scripts.baselines.acp_native import ROOT,_digest_file,_seed
from scripts.baselines.acp_source_precision import _quantiles


def static_wrench(action):
    ref=spatial.pose9_to_SE3(action[...,:9]);vt=spatial.pose9_to_SE3(action[...,9:18])
    current=np.broadcast_to(np.eye(4),ref.shape)
    stiffness=decode_stiffness_tool(ref,vt,action[...,18],current)
    error=np.concatenate([vt[...,:3,3],np.stack([orientation_error(np,np.eye(3),r) for r in vt[...,:3,:3].reshape(-1,3,3)]).reshape(*vt.shape[:-2],3)],axis=-1)
    return ref,vt,np.einsum("...ij,...j->...i",stiffness,error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());output=args.output.resolve()
    if ROOT/"latex" in output.parents:parser.error("source diagnostics belong outside latex/")
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/config["source"]["data_root"]
    paths=[source/f"source_seed_{config['source']['collection_seeds'][i]}/episode.npz" for i in config["source"]["validation_collection_indices"]]
    dataset=SourceDemonstrations(paths,history_len=8,horizon=16)
    rotations=[];origins=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            for t in range(len(data["acp.reference_pose_world"])-16):
                rotations.append(data["public.tcp_rotation"][t]);origins.append((str(path),t,float(data["public.elapsed_seconds"][t,0])))
    if len(origins)!=len(dataset):raise AssertionError("source window mapping differs")
    deterministic_runtime();torch.set_num_threads(1)
    checkpoint=args.checkpoint.resolve();payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    policy=build_policy(payload["shape_meta"],payload["history_len"])
    policy.load_state_dict(payload["state_dicts"]["ema_model"],strict=True);policy.to(args.device).eval()
    indices=np.unique(np.linspace(0,len(dataset)-1,48).astype(int));rows=[]
    for repeat in range(2):
        for start in range(0,len(indices),16):
            selected=indices[start:start+16];_seed(18590+repeat*100+start)
            with torch.inference_mode():pred=policy.predict_action(dataset.batch(selected)["obs"])["sparse"].cpu().numpy()[:,:4]
            gt=dataset.actions[selected,:4].numpy()
            pref,pvt,pw=static_wrench(pred);gref,gvt,gw=static_wrench(gt)
            for j,index in enumerate(selected):
                rotation=rotations[index]
                for h in range(4):
                    pred_intent=float(pred[j,h,18]*(rotation@(pref[j,h,:3,3]-pvt[j,h,:3,3]))[2])
                    gt_intent=float(gt[j,h,18]*(rotation@(gref[j,h,:3,3]-gvt[j,h,:3,3]))[2])
                    force_error=pw[j,h,:3]-gw[j,h,:3]
                    rows.append({"window":int(index),"repeat":repeat,"waypoint":h,
                        "episode":origins[index][0],"control_index":origins[index][1],"elapsed_s":origins[index][2],
                        "force_wrench_error_N":float(np.linalg.norm(force_error)),
                        "normal_wrench_error_N":float(abs((rotation@force_error)[2])),
                        "torque_wrench_error_Nm":float(np.linalg.norm(pw[j,h,3:]-gw[j,h,3:])),
                        "virtual_translation_error_m":float(np.linalg.norm(pred[j,h,9:12]-gt[j,h,9:12])),
                        "stiffness_abs_error_N_per_m":float(abs(pred[j,h,18]-gt[j,h,18])),
                        "label_force_intent_N":gt_intent,"predicted_force_intent_N":pred_intent,
                        "loading_intent_mismatch":bool((gt_intent>2.5)!=(pred_intent>2.5))})
    report={"checkpoint":str(checkpoint),"checkpoint_sha256":_digest_file(checkpoint),"rows":rows,
        "metrics":{key:_quantiles([row[key] for row in rows]) for key in ("force_wrench_error_N","normal_wrench_error_N","torque_wrench_error_Nm")},
        "loading_intent_mismatch_fraction":float(np.mean([row["loading_intent_mismatch"] for row in rows])),
        "worst_normal_wrench_rows":sorted(rows,key=lambda row:row["normal_wrench_error_N"],reverse=True)[:12],
        "semantics":"Native anisotropic K times VT pose error at the same observed TCP; body/world rotation retained. Static expected wrench excludes damping, velocity feedback, saturation and plant dynamics. Loading intent is the public expert's force/k world-z displacement, threshold2.5N; not hidden teacher state.",
        "target_calls":0,"source_physical_calls":0,"policy_calls":2*int(np.ceil(len(indices)/16)),
        "source_sha256":_digest_file(Path(__file__))}
    (output/"wrench_diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"metrics":report["metrics"],"loading_intent_mismatch_fraction":report["loading_intent_mismatch_fraction"]},indent=2))


if __name__=="__main__":main()
