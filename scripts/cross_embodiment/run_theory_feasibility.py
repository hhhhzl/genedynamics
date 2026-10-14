"""Resumable CPU study: matched mating, counted calibration and strong search.

This is a development study, not the final paper split. The same finite pool
of 48 calibrated-twin rollouts supports both controller-search and anchored
transport objectives. Equality/defeat by a strong baseline remains a result.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np

from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec,ContactChunk
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from genedynamics.learning.cross_embodiment.structured_response import fit_contact_twin,validate_twin
from genedynamics.learning.cross_embodiment.research_audit import ResearchAudit
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
from scripts.cross_embodiment.validate_mechanism_chain import summarize


def publish(path,value):
    from genedynamics.experiments.artifacts import atomic_json
    atomic_json(path,value)


def costs(row,source):
    """Predeclared bounded components, reported separately from success."""
    error=np.linalg.norm(np.asarray(row["final_position"])-[.44,-.2,.580])/.002
    pose=min(error,1.)
    force=min(row["final_force_rmse"]/5.,1.)
    task=.5*(not row["success"])+.25*pose+.25*force
    response=float(np.mean((np.asarray(row["response_signature"])-source)**2)/4.)
    mode=float(row["observed_mode"]!=row["mode"])
    return dict(task=float(task),response=response,mode=mode,time=float(row["seconds"]/15.),
                joint=float(.6*task+.25*response+.1*mode+.05*row["seconds"]/15.))


def transport(parameters,backend="jax"):
    k=(parameters["stiffness"],)*2+(0.,)+(parameters["stiffness"],)*3
    return ReferenceTransport.from_stage_parameters(parameters["scale"],parameters["bias"],
        alignment_stiffness=k,insertion_stiffness=k,backend=backend)


def noise_records(records,seed,position_std,force_std):
    """Probe READOUT noise, explicitly not a closed-loop sensor-noise study."""
    random=np.random.default_rng(seed);result=[]
    for record in records:
        trace=record.result.trace
        observed=trace._replace(tcp_position=trace.tcp_position+random.normal(0,position_std,trace.tcp_position.shape),
                               normal_force=trace.normal_force+random.normal(0,force_std,trace.normal_force.shape))
        result.append(replace(record,result=ContactChunk(record.result.state,observed)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--backend",choices=("jax","native"),default="native")
    p.add_argument("--device",choices=("cpu",),default="cpu",help="structured physical study is CPU only")
    p.add_argument("--robot",choices=("panda","xarm7","ur5e"),default="xarm7")
    p.add_argument("--gain",type=float,default=.65);p.add_argument("--delay",type=int,default=3)
    p.add_argument("--friction",type=float,default=.4);p.add_argument("--time-constant",type=float,default=.02)
    p.add_argument("--payload",type=float,default=.5,help="diagnostic truth; identification still assumes known .5 kg")
    p.add_argument("--calibration-seed",type=int,default=4100)
    p.add_argument("--test-seed-start",type=int,default=24000);p.add_argument("--reset-seeds",type=int,default=5)
    p.add_argument("--train-seed-start",type=int,default=310)
    p.add_argument("--position-noise",type=float,default=0.);p.add_argument("--force-noise",type=float,default=0.)
    p.add_argument("--calibrate-only",action="store_true")
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    config=mating_contact_config(a.robot);truth=ContactMechanismSpec(a.gain,a.delay,a.payload,a.friction,a.time_constant)
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior,backend=a.backend,device=a.device))
    contract={"config":asdict(config),"diagnostic_truth":asdict(truth),"source_sha256":prior.regressor.metadata["tensor_sha256"],
        "calibration_seed":a.calibration_seed,"test_seeds":list(range(a.test_seed_start,a.test_seed_start+a.reset_seeds)),
        "train_seeds":[a.train_seed_start,a.train_seed_start+1],"source_seeds":[a.train_seed_start,a.train_seed_start+1],"probe_readout_noise_std":[a.position_noise,a.force_noise],
        "source_law":"frozen legacy Panda phase prior; new geometry/decoder qualified separately",
        "backend":a.backend,"device":a.device,"physics_backend":"mujoco",
        "role":"development study; independent resets after fixed candidate search, not final paper evidence",
        "protocol":"matched_keyway_signed_wrench_axial_support_v1","code_fingerprint":code_fingerprint()}
    if (out/"contract.json").exists():
        old=json.loads((out/"contract.json").read_text())
        if old!=contract:raise ValueError("study resume contract/code mismatch")
    else:publish(out/"contract.json",contract)
    if not (out/"calibration.json").exists():
        records,ledger=collect_probes(config,truth,out/"probes",seed=a.calibration_seed,tangential=True)
        observed=noise_records(records,a.calibration_seed+999,a.position_noise,a.force_noise)
        estimate,fit=fit_contact_twin(config,observed[:3])
        validation=validate_twin(config,estimate,observed[3:])
        publish(out/"calibration.json",dict(estimated=asdict(estimate),fit=fit,validation=validation,budget=ledger,
            probe_readout_noise_std=[a.position_noise,a.force_noise],noise_scope="readout only; physical low-level force feedback remains exact"))
    calibration=json.loads((out/"calibration.json").read_text())
    if a.calibrate_only:return
    if not calibration["validation"]["supported"]:
        publish(out/"report.json",dict(complete=True,status="unsupported_response",calibration=calibration,
            attempted_task_resets=a.reset_seeds*2,successes=0,scope="no target task training; calibration rejection counted as unsupported failure"))
        return
    estimate=ContactMechanismSpec(**calibration["estimated"])
    source={}
    for seed in contract["source_seeds"]:
        for mode in (0,1):
            path=out/"source"/f"s{seed}_m{mode}"
            row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(mating_contact_config(),ContactMechanismSpec(),seed=seed,mode=mode,prior=prior,output=path)
            if not row["success"] or row["observed_mode"]!=mode:raise RuntimeError("source physical support rejected; do not train target")
            source[seed,mode]=np.asarray(row["response_signature"])
    candidates=[dict(scale=s,stiffness=k,bias=b) for s in (1.,2.5) for k in (0.,1.) for b in ((0.,0.),(-.004,.002),(-.008,.004))]
    def search(context,name):
        path=out/name
        if (path/"search.json").exists():return json.loads((path/"search.json").read_text())
        history=[]
        for index,param in enumerate(candidates):
            from genedynamics.experiments.artifacts import active_policy
            policy=active_policy()
            if policy:policy.check(16*1024**2)
            rows=[]
            for seed in contract["train_seeds"]:
                for mode in (0,1):
                    artifact=path/f"c{index}"/f"s{seed}_m{mode}"
                    row=load_bundle(artifact)[1] if (artifact/"metadata.json").exists() else rollout(config,context,seed=seed,mode=mode,prior=prior,transport=transport(param,backend=a.backend),output=artifact)
                    rows.append(dict(**row,costs=costs(row,source[seed,mode])))
            history.append(dict(parameters=param,episodes=rows,means={k:float(np.mean([r["costs"][k] for r in rows])) for k in rows[0]["costs"]}))
            print(dict(stage=name,candidate=index,successes=sum(r["success"] for r in rows),costs=history[-1]["means"]),flush=True)
        winners={objective:min(range(len(history)),key=lambda i:history[i]["means"][objective]) for objective in ("task","joint")}
        result=dict(history=history,winners=winners,twin_rollouts=48,valid_control_steps=sum(round(r["seconds"]/.02) for h in history for r in h["episodes"]),
            scope="same finite candidate pool for task-only controller search and source-response-anchored transport; not global optimization")
        publish(path/"search.json",result);return result
    fitted=search(estimate,"calibrated_search");nominal=search(ContactMechanismSpec(),"nominal_search")
    methods={"direct":None,"calibrated_transport":candidates[fitted["winners"]["joint"]],
             "same_model_controller_search":candidates[fitted["winners"]["task"]],
             "nominal_response_transport":candidates[nominal["winners"]["joint"]]}
    for name,param in methods.items():
        if param is not None:transport(param,backend=a.backend).save(out/"frozen"/name)
    freeze={"contract_sha256":hashlib.sha256((out/"contract.json").read_bytes()).hexdigest(),"methods":methods,
        "estimated":asdict(estimate),"test_seeds":contract["test_seeds"],"adaptation":"no new target task probes or target task fitting"}
    from genedynamics.experiments.artifacts import canonical_json
    freeze=canonical_json(freeze)
    if (out/"frozen_evaluation.json").exists() and json.loads((out/"frozen_evaluation.json").read_text())!=freeze:raise ValueError("frozen evaluation mismatch")
    publish(out/"frozen_evaluation.json",freeze)
    evaluations={};model_costs={}
    mean_source={m:np.mean([source[s,m] for s in contract["source_seeds"]],axis=0) for m in (0,1)}
    for name,param in methods.items():
        rows=[];predicted=[]
        for seed in contract["test_seeds"]:
            for mode in (0,1):
                deployed=None if param is None else ReferenceTransport.load(out/"frozen"/name,backend=a.backend,device=a.device)
                for mechanism,role,buffer in ((truth,"evaluation",rows),(estimate,"occupancy_audit",predicted)):
                    path=out/role/name/f"s{seed}_m{mode}"
                    row=load_bundle(path)[1] if (path/"metadata.json").exists() else rollout(config,mechanism,seed=seed,mode=mode,prior=prior,transport=deployed,output=path)
                    buffer.append(dict(**row,costs=costs(row,mean_source[mode])))
        evaluations[name]=dict(summary=summarize(rows),episodes=rows)
        model_costs[name]=predicted
        print(dict(stage="evaluation",method=name,summary=summarize(rows)),flush=True)
        publish(out/"report.json",dict(complete=False,contract=contract,calibration=calibration,evaluations=evaluations))
    names=list(methods)
    clustered=lambda table:[[np.mean([r["costs"]["joint"] for r in table[name] if r["seed"]==s]) for name in names] for s in contract["test_seeds"]]
    actual_table={name:evaluations[name]["episodes"] for name in names}
    certificate=ResearchAudit().cost_certificate(clustered(actual_table),clustered(model_costs))
    laws={objective:ResearchAudit().mode_law(
        [np.mean([r["costs"][objective] for r in fitted["history"][fitted["winners"][objective]]["episodes"] if r["mode"]==m]) for m in (0,1)],
        [.5,.5],[all(r["success"] for r in fitted["history"][fitted["winners"][objective]]["episodes"] if r["mode"]==m) for m in (0,1)]) for objective in ("task","joint")}
    publish(out/"report.json",dict(complete=True,contract=contract,calibration=calibration,methods=methods,evaluations=evaluations,
        occupancy_audit=model_costs,finite_cost_certificate=certificate,model_supported_mode_laws=laws,
        same_model_controller_equals_transport=methods["same_model_controller_search"]==methods["calibrated_transport"],
        scope="finite structured development study; mode laws are model-derived and not yet a sampled-law physical evaluation; no neural/stochastic/mobile superiority claim"))


if __name__=="__main__":main()
