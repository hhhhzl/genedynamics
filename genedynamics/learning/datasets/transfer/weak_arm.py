"""Versioned task-bound fixed-arm weak trajectories with complete deadlines.

No reward, success, mechanism, evaluator trace or private controller carry can
be serialized here. Physical receipts live beside, never inside, this buffer.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json

import numpy as np

from genedynamics.envs.adapters.transfer_task import FLAT_FIELDS
from genedynamics.learning.datasets.transfer.artifacts import atomic_json, canonical_bytes

META={"version","permission","robot","mode","seed","purpose","chart","controls","recording_end","task"}
VERSION="public-arm-weak-task-data-v2"


@dataclass(frozen=True)
class PublicArmWeakTrajectory:
    metadata: dict
    public: dict
    requests: np.ndarray

    def __post_init__(self):
        m=dict(self.metadata);r=np.array(self.requests,dtype=np.float32,copy=True)
        if set(m)!=META or m["version"]!=VERSION or m["permission"]!="B_log":
            raise ValueError("strict public target metadata required")
        if m["task"] not in {"T1","T2","T3"} or m["robot"]!="xarm7" or m["mode"] not in (0,1) or m["purpose"] not in {"fit","validation"}:
            raise ValueError("declared target body, mode and episode split required")
        if m["chart"]!="shared13" or m["recording_end"] not in {"terminal","budget_stop"}:
            raise ValueError("unknown recorded action chart/end")
        if isinstance(m["seed"],bool) or not isinstance(m["seed"],int) or m["seed"]<0:
            raise ValueError("nonnegative public reset seed required")
        n=m["controls"];width=13
        if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<={"T1":500,"T2":1000,"T3":750}[m["task"]] or r.shape!=(n,width) or not np.isfinite(r).all():
            raise ValueError("one finite command per observed20ms transition required")
        if set(self.public)!=set(FLAT_FIELDS):raise ValueError("public sensor schema only")
        expected=dict(joint_position=(7,),joint_velocity=(7,),tcp_position=(3,),tcp_rotation=(3,3),tcp_twist=(6,),
            measured_wrench=(6,),goal_position=(3,),goal_rotation=(3,3),previous_action=(13,),previous_request=(13,),
            reference_position=(3,),reference_rotation=(3,3),elapsed_seconds=(1,),body_position=(3,),body_twist=(6,),controller_feedback=(2,))
        p={k:np.array(v,dtype=np.float32,copy=True) for k,v in self.public.items()}
        if any(v.shape!=(n+1,*expected[k]) or not np.isfinite(v).all() for k,v in p.items()):
            raise ValueError("public observation before and after each command required")
        if not np.allclose(p["elapsed_seconds"].reshape(-1),np.arange(n+1)*.02,atol=1e-6):
            raise ValueError("complete causal cold-reset episode prefixes required")
        if np.any(p["previous_action"][0]) or np.any(p["previous_request"][0]):raise ValueError("fresh public reset required")
        if not np.array_equal(p["previous_request"][1:],r):raise ValueError("issued request receipt differs")
        for v in p.values():v.setflags(write=False)
        r.setflags(write=False)
        object.__setattr__(self,"metadata",MappingProxyType(m));object.__setattr__(self,"public",MappingProxyType(p));object.__setattr__(self,"requests",r)

    @property
    def digest(self):
        h=hashlib.sha256(canonical_bytes(dict(self.metadata)))
        for k,v in sorted({**self.public,"requests":self.requests}.items()):h.update(k.encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        return h.hexdigest()

    def prefix(self,controls):
        if isinstance(controls,bool) or not isinstance(controls,int) or not 1<=controls<=len(self.requests):raise ValueError("prefix exceeds recorded transitions")
        m=dict(self.metadata,controls=controls)
        if controls<len(self.requests):m["recording_end"]="budget_stop"
        return type(self)(m,{k:v[:controls+1] for k,v in self.public.items()},self.requests[:controls])

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(path/"public.npz",requests=self.requests,**{"public."+k:v for k,v in self.public.items()})
        atomic_json(path/"public.json",dict(self.metadata));(path/"public.sha256").write_text(self.digest+"\n")

    @classmethod
    def load(cls,path):
        path=Path(path)
        with np.load(path/"public.npz",allow_pickle=False) as z:
            if set(z.files)!={"requests"}|{"public."+k for k in FLAT_FIELDS}:raise ValueError("private/unknown arrays forbidden")
            row=cls(json.loads((path/"public.json").read_text()),{k:z["public."+k] for k in FLAT_FIELDS},z["requests"])
        if row.digest!=(path/"public.sha256").read_text().strip():raise ValueError("public target buffer SHA changed")
        return row

