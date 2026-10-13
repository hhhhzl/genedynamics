"""Evaluate frozen response on reserved public mechanism combinations; never fit."""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.data import load_dataset
from genedynamics.learning.cross_embodiment.learning_data import response_tables
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    model=FrozenRegressor.load(a.model); data,meta=load_dataset(a.dataset)
    if meta['role']!='response_validation':raise ValueError('reserved response_validation role required')
    x,y,train,valid,layout=response_tables(data,meta)
    if train.any() or not valid.any():raise ValueError('held-out data must contain validation rows only')
    prediction=np.concatenate([np.asarray(model.predict(x[valid][i:i+256])) for i in range(0,int(valid.sum()),256)],axis=1)
    mean=prediction.mean(0); start,end=layout['semantic_slice']
    report={'dataset_sha256':meta['tensor_sha256'],'model_sha256':model.metadata['tensor_sha256'],
        'role':meta['role'],'examples':int(valid.sum()),'normalized_mse':float(np.mean(((prediction-y[valid])/model.outputs.scale)**2)),
        'tcp_delta_rmse_m':float(np.sqrt(np.mean((mean[:,start:start+3]-y[valid,start:start+3])**2))),
        'disagreement':float(np.mean(prediction.std(0))),
        'scope':'reserved public mechanisms, diagnostic prediction only; not final hidden-target calibration or calibrated risk'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(report)


if __name__=='__main__':main()
