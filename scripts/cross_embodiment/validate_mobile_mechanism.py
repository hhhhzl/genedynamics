"""Counted coupled calibration, frozen Panda prior and supported mobile insertion."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path
import json
import platform
import numpy as np
from genedynamics.learning.cross_embodiment import RandomRequest
from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactConfig,ContactCommands,ContactChunk,MobileContactMechanismSpec,mating_contact_config
from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv,ProbeRecord
from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.mobile_response import fit_mobile_context,validate_mobile_response
from scripts.cross_embodiment.calibrate_contact_twin import Observer
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.collect_conditional_response import make_transport
from scripts.cross_embodiment.pilot_mobile_phase import body_intent
from scripts.cross_embodiment.run_theory_feasibility import publish


PRIOR="tmp/cross_embodiment/mechanism_revision/strict_phase_prior"


def restore_probe(config,path,index):
    """Read observed state with a nominal model, without target reset/stepping."""
    import mujoco
    from genedynamics.envs.domains.manipulation.cross_embodiment.backends.floating_contact import FloatingState,FloatingTrace
    d,m=load_bundle(path);runner=ContactRunner(config=config)
    data=mujoco.MjData(runner._impl.model)
    data.qpos[:]=d["qpos"];data.qvel[:]=d["qvel"];data.qacc_warmstart[:]=0.
    mujoco.mj_forward(runner._impl.model,data)
    state=FloatingState((data,),*(d["initial_"+name] for name in FloatingState._fields[1:]))
    trace=FloatingTrace(*(d["trace_"+name] for name in FloatingTrace._fields))
    return ProbeRecord(index,m["purpose"],Observer.observe(runner,state),ContactCommands(d["skill"],d["body"]),ContactChunk(state,trace),state)


def collect(config,truth,out,seed):
    runner=ContactRunner(config=config,mechanism=truth)
    gate=HiddenTargetEnv(runner,Observer(),ProbeBudget(5,str(out/"ledger.json")));records=[]
    from genedynamics.envs.domains.manipulation.cross_embodiment.backends.floating_contact import FloatingTrace
    for i in range(5):
        path=out/f"probe_{i}";purpose="fit" if i<3 else "validation";initial="jammed" if i in (1,3) else "clear"
        if i<len(gate.budget.attempts):
            if not (path/"metadata.json").exists():raise RuntimeError("charged mobile probe has no committed record; cannot repeat for free")
            records.append(restore_probe(config,path,i));continue
        skill=np.zeros((1,100,13),np.float32);t=np.arange(100)*config.control_dt
        skill[0,:,0]=.004*np.sin(2*np.pi*t);skill[0,:,1]=.002*np.sin(3*np.pi*t);skill[0,:,5]=.03*np.sin(2*np.pi*t)
        skill[...,6:12]=1.
        if initial=="jammed":skill[...,12]=3.+i*.2
        body=np.zeros((1,100,3),np.float32);body[0,:,0]=.006*np.sin((1.+.1*i)*np.pi*t);body[0,:,2]=.01*np.sin(2*np.pi*t)
        record=gate.probe(RandomRequest(seed+i,(seed*2+i,)),ContactCommands(skill,body),purpose=purpose,initial=initial);records.append(record)
        save_bundle(path,dict(skill=skill,body=body,qpos=record.initial_state.physics[0].qpos,qvel=record.initial_state.physics[0].qvel,
            **{"initial_"+n:getattr(record.initial_state,n) for n in record.initial_state._fields if n!="physics"},
            **{"trace_"+n:getattr(record.result.trace,n) for n in FloatingTrace._fields}),dict(purpose=purpose,status=int(record.result.state.status[0])))
    gate.seal();return records,gate.budget.snapshot()


def episode(job):
    path,cfg,mech,param,seed,mode=job;path=Path(path)
    if (path/"metadata.json").exists():return load_bundle(path)[1]
    prior=PhaseGaussianPrior(FrozenRegressor.load(PRIOR,backend="native"))
    row=rollout(ContactConfig(**cfg),MobileContactMechanismSpec(**mech),seed=seed,mode=mode,prior=prior,transport=make_transport(param),
        output=path,record_carry=True,body_selector=body_intent)
    d,m=load_bundle(path);valid=d["valid"]
    body=dict(min_support=int(d["support_count"][valid].min()),max_tilt=float(d["base_tilt"][valid].max()),
        max_qp_residual=float(d["wbc_residual"][valid].max()),base_displacement=float(np.linalg.norm(d["base_position"][0,-1,:2]-d["base_position"][0,0,:2])))
    row.update(body_metrics=body,joint_success=bool(row["success"] and body["min_support"]>=2 and body["max_tilt"]<=.5 and body["max_qp_residual"]<=.01))
    save_bundle(path,d,dict(m,**row));return row


def score(row):
    task=.5*(not row["joint_success"])+.25*min(np.linalg.norm(np.asarray(row["final_position"])-[.44,-.2,.58])/.002,1.)+.25*min(row["final_force_rmse"]/5.,1.)
    return dict(task=float(task),joint=float(.8*task+.15*(row["observed_mode"]!=row["mode"])+.05*row["seconds"]/15.))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--body-gain",type=float,default=.95);p.add_argument("--arm-gain",type=float,default=1.);p.add_argument("--delay",type=int,default=1)
    p.add_argument("--seed-start",type=int,default=55000);a=p.parse_args();out=a.output
    config=replace(mating_contact_config("xarm7"),robot="go2_z1",mobile=True,environment="mobile_keyed_insertion_recovery",reset_task_noise=0.,reset_joint_noise=.0001,approach_speed=.01,approach_integral_limit=.03)
    truth=MobileContactMechanismSpec(a.arm_gain,a.delay,.5,.4,.02,a.body_gain)
    bank=[dict(style="impedance",scale=s,stiffness=k,bias=[0.,0.]) for s in (1.,1.5,2.) for k in (0.,1.)]
    family=[MobileContactMechanismSpec(g,d,.5,.4,.02,b) for g in (.9,1.) for d in (0,1) for b in (.9,.95,1.)]
    contract=dict(config=asdict(config),diagnostic_truth=asdict(truth),family=[asdict(x) for x in family],candidates=bank,
        source_prior_sha256=load_bundle(PRIOR)[1]["tensor_sha256"],train_seeds=[a.seed_start,a.seed_start+1],test_seeds=list(range(a.seed_start+100,a.seed_start+105)),
        target_probe_budget=5,target_task_training_interactions=0,model_episode_budget_per_search=24,
        thresholds=dict(success_fraction=.95,mode_fraction=1.,min_support=2,max_tilt=.5,max_qp_residual=.01),
        body_intent="world vx .005 m/s for first 4 s, real floating stance, no walking",runtime=platform.platform())
    if (out/"contract.json").exists():
        if json.loads((out/"contract.json").read_text())!=contract:raise ValueError("mobile frozen contract mismatch")
    else:publish(out/"contract.json",contract)
    if not (out/"calibration.json").exists():
        records,ledger=collect(config,truth,out/"probes",a.seed_start+200)
        estimate,fit=fit_mobile_context(config,records[:3],family)
        arm,arm_fit=fit_mobile_context(config,records[:3],[m for m in family if m.body_servo_gain==1.])
        validation=validate_mobile_response(config,estimate,records[3:]);arm_validation=validate_mobile_response(config,arm,records[3:])
        publish(out/"calibration.json",dict(estimated=asdict(estimate),fit=fit,validation=validation,arm_only_estimated=asdict(arm),arm_only_fit=arm_fit,arm_only_validation=arm_validation,budget=ledger))
    calibration=json.loads((out/"calibration.json").read_text())
    if not calibration["validation"]["supported"]:raise RuntimeError("coupled mobile response validation rejected")
    print(dict(stage="mobile_calibration",estimated=calibration["estimated"],validation=calibration["validation"],arm_only_validation=calibration["arm_only_validation"]),flush=True)
    contexts=dict(coupled=calibration["estimated"],arm_only=calibration["arm_only_estimated"],nominal=asdict(MobileContactMechanismSpec()))
    jobs=[(str(out/f"{role}/c{i}/s{s}_m{m}"),asdict(config),context,param,s,m) for role,context in contexts.items() for i,param in enumerate(bank) for s in contract["train_seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n,_ in enumerate(pool.map(episode,jobs),1):
            if n%12==0:print(dict(stage="mobile_model_search",episodes=n,total=len(jobs)),flush=True)
    selected={}
    for role in contexts:
        rows=[]
        for i,param in enumerate(bank):
            scores=[score(load_bundle(out/f"{role}/c{i}/s{s}_m{m}")[1]) for s in contract["train_seeds"] for m in (0,1)]
            rows.append({k:float(np.mean([x[k] for x in scores])) for k in ("task","joint")})
        selected[role]={k:min(range(len(bank)),key=lambda i:rows[i][k]) for k in ("task","joint")};publish(out/f"{role}/search.json",dict(scores=rows,winners=selected[role],model_episodes=24))
    methods=dict(direct=0,coupled=selected["coupled"]["joint"],task_only=selected["coupled"]["task"],arm_only=selected["arm_only"]["joint"],nominal=selected["nominal"]["joint"])
    publish(out/"frozen.json",dict(methods=methods,parameters={n:bank[i] for n,i in methods.items()},source_prior_sha256=contract["source_prior_sha256"]))
    jobs=[(str(out/f"evaluation/c{i}/s{s}_m{m}"),asdict(config),asdict(truth),bank[i],s,m) for i in sorted(set(methods.values())) for s in contract["test_seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(episode,jobs))
    results={}
    for name,i in methods.items():
        rows=[load_bundle(out/f"evaluation/c{i}/s{s}_m{m}")[1] for s in contract["test_seeds"] for m in (0,1)]
        results[name]=dict(attempted=len(rows),successes=sum(r["joint_success"] for r in rows),mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows),episodes=rows)
    e=results["coupled"]
    publish(out/"report.json",dict(complete=True,passed=e["successes"]>=.95*e["attempted"] and e["mode_matches"]==e["attempted"],
        contract=contract,calibration=calibration,methods=methods,evaluations=results,distinct_physical_target_evaluation_episodes=len(jobs),
        target_task_training_interactions=0,scope="calibrated known Go2+Z1 stance adapter; Task3 Panda prior unchanged; no navigation, walking, wheeled or humanoid qualification"))
    print({n:{k:r[k] for k in ("successes","attempted","mode_matches")} for n,r in results.items()},flush=True)


if __name__=="__main__":main()
