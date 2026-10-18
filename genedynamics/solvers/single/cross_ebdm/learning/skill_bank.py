"""Frozen legal source-history queries from SHA-verified training episodes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from genedynamics.tasks.transfer.guide_observation import context_sequence
from genedynamics.tasks.transfer.specs import task_spec
from .response import response_windows,PUBLIC_FIELDS


def semantic_embedding(public):
    n=len(public['tcp_position'])
    return np.concatenate((np.asarray(public['tcp_position'])/.02,np.asarray(public['tcp_rotation']).reshape(n,9)/.1,
        np.asarray(public['tcp_twist'])/.1,np.asarray(public['measured_wrench'])/np.asarray([5.]*3+[1.]*3)),axis=1)


class PublicSkillBank:
    def __init__(self,entries,response_spec,*,task='T3'):
        self.spec=response_spec;self.task=task;arrays=[];embeddings=[];phases=[];modes=[];sources=[]
        for entry in entries:
            path=Path(entry['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry['sha256']:raise ValueError('frozen source skill episode changed')
            with np.load(path,allow_pickle=False) as z:p={k[7:]:z[k].copy() for k in z.files if k.startswith('public.')};a=z['actions'].copy()
            windows=response_windows(p,a,body='panda',task=task,calibration_context=[0,0,0],spec=response_spec);t=windows['ticks'];context=context_sequence(p,task_spec(task,robot='panda'),entry['mode'])
            arrays.append(dict(features=windows['features'],commands=windows['commands'],position=p['tcp_position'][t],rotation=p['tcp_rotation'][t],wrench=p['measured_wrench'][t]))
            embeddings.append(semantic_embedding(p)[t]);phases.append(np.argmax(context[t,22:26],axis=1));modes.append(np.full(len(t),entry['mode']))
            sources.append(dict(path=str(path),sha256=digest,mode=entry['mode'],windows=len(t)))
        if not arrays:raise ValueError('nonempty qualified training source bank required')
        self.arrays={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};self.embedding=np.concatenate(embeddings);self.phases=np.concatenate(phases);self.modes=np.concatenate(modes);self.groups={}
        for phase in range(4):
            for mode in (0,1):
                ids=np.flatnonzero((self.phases==phase)&(self.modes==mode))
                if len(ids):self.groups[phase,mode]=(ids,cKDTree(self.embedding[ids]))
        self.manifest=dict(version='public-source-history-query-v1',task=task,body='panda',source_training_only=True,entries=sources,
            query='nearest actual source history, same public phase and requested mode; no target joints/body/controller carry copied to source',
            response_spec=dict(history_len=response_spec.history_len,horizon=response_spec.horizon,maximum_joints=response_spec.maximum_joints))
        self.digest=hashlib.sha256(json.dumps(self.manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def query(self,public,phase,modes):
        if set(public)!=PUBLIC_FIELDS or len(phase)!=len(public['tcp_position']) or len(modes)!=len(phase):raise ValueError('one complete public query and request per world required')
        embed=semantic_embedding(public);chosen=[];distances=[]
        for e,p,m in zip(embed,phase,modes):
            if (int(p),int(m)) not in self.groups:raise ValueError('source query phase/mode absent from frozen source bank')
            ids,tree=self.groups[int(p),int(m)];distance,index=tree.query(e);chosen.append(ids[index]);distances.append(distance)
        return {k:v[chosen].copy() for k,v in self.arrays.items()},np.asarray(distances),np.asarray(chosen,dtype=np.int64)
