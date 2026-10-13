"""Observed-state conditional response data from a probe-calibrated public model.

Candidate, reset, mode and time are distinct axes. Independent CPU processes
are a reference scheduler, not the future GPU provider. No target evaluation
or hidden target model is used by the collector.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.run_theory_feasibility import publish


def candidate_bank():
    return [dict(scale=s,stiffness=k,bias=list(b)) for s in (1.,2.5) for k in (0.,1.)
            for b in ((0.,0.),(-.004,.002),(-.008,.004))]


def make_transport(parameters,backend="native"):
    k=(parameters["stiffness"],)*2+(0.,)+(parameters["stiffness"],)*3
    if parameters.get("style")=="impedance":return ReferenceTransport.from_impedance(parameters["scale"],k,backend=backend)
    return ReferenceTransport.from_stage_parameters(parameters["scale"],parameters["bias"],
        alignment_stiffness=k,insertion_stiffness=k,backend=backend)


def collect_one(job):
    path,config,mechanism,prior_path,index,parameters,seed,mode,noise=job
    if (Path(path)/"metadata.json").exists():return load_bundle(path)[1]
    prior=PhaseGaussianPrior(FrozenRegressor.load(prior_path,backend="native"))
    row=rollout(ContactConfig(**config),ContactMechanismSpec(**mechanism),seed=seed,mode=mode,
        prior=prior,transport=make_transport(parameters),output=path,record_carry=True,observation_noise=noise)
    return dict(candidate=index,**row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/mechanism_revision/strict_phase_prior"))
    p.add_argument("--workers",type=int,default=4);p.add_argument("--train-groups",type=int,default=12)
    p.add_argument("--validation-groups",type=int,default=4);p.add_argument("--calibration-groups",type=int,default=4)
    p.add_argument("--test-only",action="store_true");p.add_argument("--test-start",type=int,default=40000);p.add_argument("--test-groups",type=int,default=8)
    a=p.parse_args(); study=json.loads((a.study/"report.json").read_text())
    if not study["complete"] or not study["calibration"]["validation"]["supported"]:raise ValueError("validated counted calibration required")
    config=study["contract"]["config"]; estimate=study["calibration"]["estimated"]
    bank=candidate_bank();split={"train":list(range(33000,33000+a.train_groups)),
        "validation":list(range(34000,34000+a.validation_groups)),"interval_calibration":list(range(35000,35000+a.calibration_groups))}
    if a.test_only:split={"test":list(range(a.test_start,a.test_start+a.test_groups))}
    producer=inspect.getsource(rollout)
    contract=dict(schema=2,config=config,estimated=estimate,candidates=bank,split=split,
        source_sha256=load_bundle(a.prior)[1]["tensor_sha256"],
        calibration_sha256=hashlib.sha256((a.study/"calibration.json").read_bytes()).hexdigest(),
        target_task_training_interactions=0,model_episode_budget=sum(map(len,split.values()))*len(bank)*2,
        rollout_sha256=hashlib.sha256(producer.encode()).hexdigest(),
        dataset_role="post_calibration_model_development",noise_scope="explicit source stream; no observation corruption")
    if (a.output/"contract.json").exists():
        if json.loads((a.output/"contract.json").read_text())!=contract:raise ValueError("immutable dataset contract mismatch")
    else:publish(a.output/"contract.json",contract)
    jobs=[(str(a.output/role/f"c{i}"/f"s{seed}_m{mode}"),config,estimate,str(a.prior),i,param,seed,mode,None)
          for role,seeds in split.items() for seed in seeds for i,param in enumerate(bank) for mode in (0,1)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(collect_one,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%24==0:print(dict(model_episodes=len(rows),total=len(jobs),last_status=row["status"]),flush=True)
    publish(a.output/"report.json",dict(complete=True,contract=contract,episodes=len(rows),
        valid_control_steps=sum(round(r["seconds"]/config["control_dt"]) for r in rows),
        target_task_training_interactions=0))


if __name__=="__main__":main()
