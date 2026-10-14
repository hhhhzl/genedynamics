"""Explicit state-history ACP adaptation trained only on qualified public source demonstrations.

This is a benchmark adaptation: no claim of native CLIP/7000-Hz force sensing.
The actual Mod1 U-Net diffusion learner, normalizer, labels and EMA are retained.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from diffusers import DDIMScheduler

from genedynamics.solvers.single.acp import spatial
from genedynamics.solvers.single.acp.compliance import VirtualTargetEstimator
from genedynamics.solvers.single.acp.models.normalizer import LinearNormalizer
from genedynamics.solvers.single.acp.policy import ACPDiffusionPolicy

PUBLIC_FIELDS = ("tcp_position", "tcp_rotation", "tcp_twist", "measured_wrench",
                 "goal_position", "goal_rotation", "elapsed_seconds")


def pose(position, rotation):
    result=np.zeros(np.asarray(position).shape[:-1]+(4,4),dtype=np.float64)
    result[..., :3,:3]=rotation
    result[..., :3,3]=position
    result[...,3,3]=1
    return result


def history_features(history,history_len):
    selected=history[-history_len:]
    selected=[selected[0]]*(history_len-len(selected))+selected
    current=selected[-1]
    base=pose(current["tcp_position"],current["tcp_rotation"])
    inverse=spatial.SE3_inv(base)
    poses=np.stack([pose(item["tcp_position"],item["tcp_rotation"]) for item in selected],axis=1)
    relative=spatial.SE3_to_pose9(inverse[:,None]@poses)
    rotation=np.swapaxes(np.asarray(current["tcp_rotation"]),-1,-2)
    def body_vectors(key):
        values=np.stack([item[key] for item in selected],axis=1)
        return np.concatenate([np.einsum("bij,bhj->bhi",rotation,values[...,:3]),
                               np.einsum("bij,bhj->bhi",rotation,values[...,3:])],axis=-1)
    goal=pose(current["goal_position"],current["goal_rotation"])
    return {"tcp_relative_pose":relative.astype(np.float32),
            "wrench_tool":body_vectors("measured_wrench").astype(np.float32),
            "twist_tool":body_vectors("tcp_twist").astype(np.float32),
            "goal_relative_pose":spatial.SE3_to_pose9(inverse@goal)[:,None].astype(np.float32),
            "elapsed_seconds":np.asarray(current["elapsed_seconds"],dtype=np.float32)[:,None]}


class StateHistoryEncoder(nn.Module):
    """Learned public TCP/force/twist history embedding, explicitly replacing native vision."""
    def __init__(self,history_len,feature_dim=64):
        super().__init__()
        self.history_len=history_len
        self.feature_dim=feature_dim
        self.history=nn.Sequential(nn.Linear(history_len*21,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.fusion=nn.Sequential(nn.Linear(138,128),nn.SiLU(),nn.Linear(128,feature_dim))

    def output_shape(self):
        return (self.feature_dim,)

    def forward(self,obs):
        temporal=torch.cat([obs["tcp_relative_pose"],obs["wrench_tool"],obs["twist_tool"]],dim=-1)
        features=self.history(temporal.flatten(start_dim=1))
        goal=torch.cat([obs["goal_relative_pose"].flatten(start_dim=1),obs["elapsed_seconds"].flatten(start_dim=1)],dim=-1)
        return self.fusion(torch.cat([features,goal],dim=-1))


def model_configuration(history_len=8,horizon=16,seed=18420):
    shapes={"tcp_relative_pose":9,"wrench_tool":6,"twist_tool":6,"goal_relative_pose":9,"elapsed_seconds":1}
    meta={"action":{"shape":[19]},"obs":{key:{"shape":[size],"type":"low_dim"} for key,size in shapes.items()},
          "sample":{"obs":{"sparse":{key:{"horizon":history_len if key in {"tcp_relative_pose","wrench_tool","twist_tool"} else 1,"down_sample_steps":1} for key in shapes}},
                    "action":{"sparse":{"horizon":horizon,"down_sample_steps":1}}}}
    cfg=OmegaConf.create({"optimizer":{"lr":3e-4,"betas":[.95,.999],"eps":1e-8,"weight_decay":1e-6},
                         "training":{"seed":seed,"gradient_accumulate_every":1,"use_ema":True,"lr_scheduler":"cosine","lr_warmup_steps":100},
                         "policy":{"obs_encoder":{"reduce_pretrained_lr":False}},
                         "ema":{"update_after_step":0,"inv_gamma":1.,"power":.75,"min_value":0.,"max_value":.9999}})
    return meta,cfg


def build_policy(meta,history_len,*,down_dims=(256,512,1024)):
    scheduler=DDIMScheduler(num_train_timesteps=50,beta_start=.0001,beta_end=.02,
                            beta_schedule="squaredcos_cap_v2",clip_sample=True,
                            set_alpha_to_one=True,steps_offset=0,prediction_type="epsilon")
    return ACPDiffusionPolicy(shape_meta=meta,noise_scheduler=scheduler,
             obs_encoder=StateHistoryEncoder(history_len),num_inference_steps=16,
             diffusion_step_embed_dim=32,down_dims=down_dims,kernel_size=5,n_groups=8,
             cond_predict_scale=True,input_pertub=.1)


class SourceDemonstrations:
    def __init__(self,paths,*,history_len=8,horizon=16):
        self.history_len,self.horizon=history_len,horizon
        observations,actions=[],[]
        self.receipts=[]
        for path in map(Path,paths):
            report=json.loads(path.with_name("report.json").read_text())
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if not report["valid_source_demo"] or digest != report["episode_sha256"]:
                raise ValueError("ACP adaptation requires SHA-verified successful source demonstrations")
            with np.load(path,allow_pickle=False) as data:
                public={key:np.array(data["public."+key],copy=True) for key in PUBLIC_FIELDS}
                reference=pose(data["public.reference_position"],data["public.reference_rotation"])
                recorded_acp="acp.reference_pose_world" in data
                if recorded_acp:
                    reference=data["acp.reference_pose_world"].copy()
                    recorded_virtual=data["acp.virtual_target_pose_world"].copy()
                    recorded_stiffness=data["acp.stiffness_scalar"].copy()
            tool=pose(public["tcp_position"],public["tcp_rotation"])
            virtual=tool.copy()
            stiffness=np.empty(len(tool))
            estimator=VirtualTargetEstimator(5000,200,.5,5,dim=3,characteristic_length=.02,vel_tol=999.002)
            for t in range(len(tool)):
                rotation=tool[t,:3,:3].T
                wrench=np.r_[rotation@public["measured_wrench"][t,:3],rotation@public["measured_wrench"][t,3:]]
                twist=np.r_[rotation@public["tcp_twist"][t,:3],rotation@public["tcp_twist"][t,3:]]
                stiffness[t],offset,_=estimator.update(wrench,twist)
                virtual[t,:3,3]+=tool[t,:3,:3]@offset
            if recorded_acp:
                virtual=recorded_virtual
                stiffness=recorded_stiffness
            for t in range(len(reference)-horizon):
                history=[{key:public[key][index:index+1] for key in PUBLIC_FIELDS}
                         for index in range(max(0,t-history_len+1),t+1)]
                observations.append({key:value[0] for key,value in history_features(history,history_len).items()})
                base=spatial.SE3_inv(tool[t])
                indices=np.arange(t,t+horizon)
                actions.append(np.concatenate([spatial.SE3_to_pose9(base@reference[indices]),
                    spatial.SE3_to_pose9(base@virtual[indices]),stiffness[indices,None]],axis=-1).astype(np.float32))
            self.receipts.append({"path":str(path),"sha256":digest,"public_observations":len(tool),
                "source_success":True,"labels":"recorded_executed_ACP_commands" if recorded_acp else "posthoc_native_virtual_target_estimates",
                "source_interaction_accounting":report["source_interaction_accounting"]})
        if not observations:
            raise ValueError("no source demonstration windows")
        self.observations={key:torch.from_numpy(np.stack([row[key] for row in observations])) for key in observations[0]}
        self.actions=torch.from_numpy(np.stack(actions))

    def __len__(self):
        return len(self.actions)

    def batch(self,indices):
        return {"obs":{"sparse":{key:value[indices] for key,value in self.observations.items()}},
                "action":{"sparse":self.actions[indices]}}

    def normalizer(self,*,rotation_residual=False):
        normalizer=LinearNormalizer()
        normalizer.fit({**self.observations,"action":self.actions})
        with torch.no_grad():
            for key,intervals in {"action":((3,9),(12,18)),"tcp_relative_pose":((3,9),),
                                   "goal_relative_pose":((3,9),)}.items():
                for start,end in intervals:
                    normalizer.params_dict[key]["scale"][start:end]=1
                    normalizer.params_dict[key]["offset"][start:end]=0
            if rotation_residual:
                identity=torch.tensor([1.,0.,0.,0.,1.,0.])
                rotations=torch.cat([self.actions[...,3:9],self.actions[...,12:18]],dim=0)
                scale=max(.01,float((rotations-identity).abs().max())*1.25)
                for start,end in ((3,9),(12,18)):
                    normalizer.params_dict["action"]["scale"][start:end]=1./scale
                    normalizer.params_dict["action"]["offset"][start:end]=-identity/scale
                self.rotation_residual_scale=scale
        return normalizer


def sampling_profile(policy,dataset,indices=(0,1,30,90,180,250,268,300,500)):
    from scipy.spatial.transform import Rotation
    device=next(policy.parameters()).device
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    policy.eval()
    for index in indices:
        if index>=len(dataset):continue
        torch.manual_seed(18499+index)
        if device.type=="cuda":torch.cuda.synchronize(device)
        started=time.perf_counter()
        with torch.inference_mode():
            predicted=policy.predict_action(dataset.batch([index])["obs"])["sparse"][0].cpu().numpy()
        if device.type=="cuda":torch.cuda.synchronize(device)
        expected=dataset.actions[index].numpy()
        row={"window":index,"sampling_seconds":time.perf_counter()-started,
             "reference_translation_error_m":float(np.linalg.norm(predicted[0,:3]-expected[0,:3])),
             "virtual_translation_error_m":float(np.linalg.norm(predicted[0,9:12]-expected[0,9:12]))}
        for name,offset in [("reference",0),("virtual",9)]:
            a=spatial.pose9_to_SE3(predicted[0,offset:offset+9])[:3,:3]
            b=spatial.pose9_to_SE3(expected[0,offset:offset+9])[:3,:3]
            row[name+"_rotation_error_rad"]=float(Rotation.from_matrix(a.T@b).magnitude())
        rows.append(row)
    return {"source_windows":rows,"device":str(device),
             "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
             "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,
             "rss_max_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "mean_sampling_seconds":float(np.mean([row["sampling_seconds"] for row in rows]))}
