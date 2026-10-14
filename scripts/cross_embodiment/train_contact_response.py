"""Extend a frozen Panda task prior to public planar/floating response bodies."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from genedynamics.learning.cross_embodiment.contact_data import load_contact_dataset,pad_contact_carry,SEMANTIC_SLICE
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,fit_regressor
from genedynamics.learning.cross_embodiment.models import ModelSpec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-prior',type=Path,required=True); p.add_argument('--responses',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--updates',type=int,default=5000); p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cpu')
    a=p.parse_args(); source=FrozenRegressor.load(a.source_prior)
    if source.metadata.get('learned_role')!='source_prior':raise ValueError('frozen source prior required')
    datasets=[load_contact_dataset(path) for path in a.responses]; dimension=max(d['carry'].shape[-1] for d,m in datasets)
    tables=[]
    for d,m in datasets:
        if m['role']!='response_development':raise ValueError('only public development response data may train this model')
        b,k,h=d['command'].shape[:3]; carry=pad_contact_carry(d['carry'],dimension); delta=pad_contact_carry(d['delta'],dimension)
        body=np.broadcast_to(d['body'][:,None],(b,k,3)); context=(d['mechanism']-[1.,0.,.5])/[.25,4.,.5]
        x=np.concatenate((carry,d['command'].reshape(b,k,-1),body,np.broadcast_to(context[:,None],(b,k,3))),axis=-1)
        y=np.concatenate((delta,d['risk']),axis=-1); valid=d['valid'].any(-1)
        tables.append((x.reshape(b*k,-1),y.reshape(b*k,-1),(valid&(d['group'][:,None]!=0)).ravel(),(valid&(d['group'][:,None]==0)).ravel()))
    x,y,train,validation=(np.concatenate([t[i] for t in tables]).astype(np.float32 if i<2 else bool) for i in range(4))
    response,state,report=fit_regressor(x,y,train,validation,spec=ModelSpec(x.shape[-1],y.shape[-1],hidden=(96,96),ensemble=3),
        updates=a.updates,checkpoint_path=a.output/'response_resume',resume=a.resume,device=a.device,
        metadata={'learned_role':'response_ensemble','risk_kind':'contact','active_command_dim':13,'command_bounds':[.08]*3+[.4]*3+[1.]*6+[25.],
            'layout':{'carry_dim':dimension,'command_dim':h*13,'context_dim':3,'body_feature_dim':3,'semantic_slice':list(SEMANTIC_SLICE),'force_index':193},
            'response_datasets':[m['tensor_sha256'] for d,m in datasets],'source_prior_sha256':source.metadata['tensor_sha256'],
            'scope':'seen public response bodies; fixed/planar carry padded with zero stance-WBC extension',
            'body_intention':'known fixture vx=.005 until t=4s; not arbitrary learned locomotion',
            'risks':['force_peak','contact_loss','collision_or_support_failure'],'uncertainty':'uncalibrated ensemble disagreement'})
    response.save(a.output/'response',fit_report=report)
    # Byte-for-byte frozen inference artifact: Task 4 never trains a new prior.
    destination=a.output/'source_prior'
    if destination.exists():
        if FrozenRegressor.load(destination).metadata['tensor_sha256']!=source.metadata['tensor_sha256']:raise ValueError('existing prior mismatch')
    else:shutil.copytree(a.source_prior,destination)
    print(report)


if __name__=='__main__':main()
