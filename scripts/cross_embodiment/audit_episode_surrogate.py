"""Qualify a small learned finite-policy response model independently of a twin.

The model predicts a complete episode signature/cost for a fixed candidate
bank and calibrated body. It is NOT the shared autoregressive dynamics model.
No real evaluation trace enters fitting or validation-driven parameter updates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.learned import fit_regressor
from genedynamics.learning.cross_embodiment.models import ModelSpec
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.run_theory_feasibility import publish


def features(parameters,mode):
    return np.asarray([parameters["scale"],parameters["stiffness"],
        parameters["bias"][0]/.01,parameters["bias"][1]/.01,float(mode==0),float(mode==1)],np.float32)


def errors(prediction,actual):
    return dict(signature_rmse=float(np.sqrt(np.mean((prediction[:,:24]-actual[:,:24])**2))),
        task_cost_rmse=float(np.sqrt(np.mean((prediction[:,24]-actual[:,24])**2))),
        failure_classification_error=float(np.mean((prediction[:,25]>.5)!=(actual[:,25]>.5))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--updates",type=int,default=3000)
    a=p.parse_args()
    search=json.loads((a.study/"calibrated_search/search.json").read_text())
    study=json.loads((a.study/"report.json").read_text())
    if not study["complete"]:raise ValueError("completed physical audit required")
    x=[];y=[];seeds=[]
    for candidate in search["history"]:
        for row in candidate["episodes"]:
            x.append(features(candidate["parameters"],row["mode"]))
            y.append([*row["response_signature"],row["costs"]["task"],float(not row["success"])])
            seeds.append(row["seed"])
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);seeds=np.asarray(seeds)
    train=seeds==min(seeds);validation=~train
    contract=dict(training="only calibrated model rollouts; no hidden target task traces",train_reset=int(min(seeds)),
        validation_resets=sorted(map(int,set(seeds[validation]))),updates=a.updates,model=dict(hidden=[32,32],ensemble=3),
        thresholds=dict(signature_rmse=.05,task_cost_rmse=.15,failure_classification_error=.1),
        code_fingerprint=code_fingerprint(),scope="finite candidate episode response; fixed calibrated xArm7 body/context; no autoregressive/global kernel qualification")
    if (a.output/"contract.json").exists():raise ValueError("immutable learned-response study already exists; use a new output")
    publish(a.output/"contract.json",contract)
    model,_,fit=fit_regressor(x,y,train,validation,spec=ModelSpec(6,26,hidden=(32,32),ensemble=3),seed=90,
        updates=a.updates,batch_size=24,metadata=dict(learned_role="finite_episode_response",
            config=study["contract"]["config"],calibration=study["calibration"]["estimated"],
            scope=contract["scope"],source_dataset=str(a.study/"calibrated_search")))
    model.save(a.output/"response",fit_report=fit)
    prediction=np.asarray(model.predict(x[validation])).mean(0)
    validation_report=errors(prediction,y[validation])
    supported=all(validation_report[k]<=v for k,v in contract["thresholds"].items())
    occupancy={}
    for name,evaluation in study["evaluations"].items():
        parameters=study["methods"][name] or dict(scale=1.,stiffness=0.,bias=[0.,0.])
        inputs=np.asarray([features(parameters,r["mode"]) for r in evaluation["episodes"]])
        actual=np.asarray([[*r["response_signature"],r["costs"]["task"],float(not r["success"])] for r in evaluation["episodes"]])
        prediction=np.asarray(model.predict(inputs))
        occupancy[name]=dict(errors(prediction.mean(0),actual),ensemble_disagreement=float(prediction.std(0).mean()))
    publish(a.output/"report.json",dict(complete=True,contract=contract,fit=fit,validation=validation_report,
        validation_supported=supported,physical_episode_occupancy_audit=occupancy,
        scope="separate learned finite-response diagnostic; no inheritance of twin success; ensemble disagreement is uncalibrated; no neural transport deployed"))
    print(dict(validation=validation_report,validation_supported=supported,occupancy=occupancy),flush=True)


if __name__=="__main__":main()
