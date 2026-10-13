"""CPU 3D ObstacleTransfer: frozen learned prior, probes, twin and heldout test."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment import CrossEmbodiment,ExecutionSpec,MechanismSpec,ObstacleTransferConfig,RandomRequest,SkillSpec
from genedynamics.learning.cross_embodiment.contracts import ControllerState,StepTrace,ChunkResult
from genedynamics.learning.cross_embodiment.source_query import SourceQueryAdapter
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,GaussianSkillPrior
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from genedynamics.learning.cross_embodiment.learning_data import online_prior_condition
from genedynamics.learning.cross_embodiment.data import observed_route_modes
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv,ProbeRecord
from genedynamics.learning.cross_embodiment.obstacle_response import fit_obstacle_context,validate_obstacle_response
from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
from scripts.cross_embodiment.run_theory_feasibility import publish


def execution():return ExecutionSpec(backend="numpy",physics_backend="mujoco")


def collect_probes(task,skill,truth,output):
    runner=CrossEmbodiment(task=task,skill=skill,mechanism=truth,execution=execution())
    observer=SourceQueryAdapter(task=replace(task,robot="panda"),skill=skill,execution=execution())
    gate=HiddenTargetEnv(runner,observer,ProbeBudget(5,str(output/"ledger.json")));records=[]
    for i in range(5):
        path=output/f"probe_{i}";purpose="fit" if i<3 else "validation"
        if i<len(gate.budget.attempts):
            if not (path/"metadata.json").exists():raise RuntimeError("charged obstacle probe lost; cannot replay for free")
            d,m=load_bundle(path);state=runner.reset_batch(RandomRequest(46000+i,(460000+i,)))
            import mujoco
            state.physics[0].qpos[:]=d["initial_qpos"];state.physics[0].qvel[:]=d["initial_qvel"];mujoco.mj_forward(runner._impl.model,state.physics[0])
            state=state._replace(controller=ControllerState(*(d["initial_"+n] for n in ControllerState._fields)))
            trace=StepTrace(*(d["trace_"+n] for n in StepTrace._fields))
            record=ProbeRecord(i,purpose,observer.observe(runner,state),d["commands"],ChunkResult(state,trace),state)
        else:
            command=np.zeros((1,100,13),np.float32);t=np.arange(100)*task.control_dt
            command[0,:,0]=.015*np.sin((2+i*.2)*np.pi*t);command[0,:,1]=.01*np.sin((3+i*.1)*np.pi*t)
            command[0,:,2]=.008*np.sin(2*np.pi*t);command[0,:,5]=.08*np.sin(3*np.pi*t)
            record=gate.probe(RandomRequest(46000+i,(460000+i,)),command,purpose=purpose)
            save_bundle(path,dict(commands=command,initial_qpos=record.initial_state.physics[0].qpos.copy(),initial_qvel=record.initial_state.physics[0].qvel.copy(),
                **{"initial_"+n:getattr(record.initial_state.controller,n) for n in ControllerState._fields},
                **{"trace_"+n:getattr(record.result.trace,n) for n in StepTrace._fields}),dict(purpose=purpose))
        records.append(record)
    gate.seal();return records,gate.budget.snapshot()


def rollout_job(job):
    path,task_values,mechanism_values,prior_path,stiffness,seed,mode=job;path=Path(path)
    if (path/"metadata.json").exists():return load_bundle(path)[1]
    task=ObstacleTransferConfig(**task_values);skill=SkillSpec();mechanism=MechanismSpec(**mechanism_values)
    runner=CrossEmbodiment(task=task,skill=skill,mechanism=mechanism,execution=execution())
    observer=SourceQueryAdapter(task=replace(task,robot="panda"),skill=skill,execution=execution())
    prior=GaussianSkillPrior(FrozenRegressor.load(prior_path,backend="native"))
    mapping=ReferenceTransport.from_impedance(1.,(stiffness,)*6,backend="native")
    request=RandomRequest(seed,(seed*2+mode,));state=runner.reset_batch(request);traces=[];previous=None;support_fail=False
    for k in range(task.timeout_steps//skill.horizon):
        semantic=observer.observe(runner,state);query=observer.build(semantic,request,previous=previous);previous=query.virtual_state
        if not query.support.all():support_fail=True;state=state._replace(status=np.array([7]));break
        condition=online_prior_condition(query.semantics,np.asarray(task.goal_position),rotation(task.goal_yaw),state.steps*task.control_dt,np)
        noise=np.random.default_rng(np.random.SeedSequence([seed,k,726])).normal(size=(1,skill.horizon,prior.command_dim)).astype(np.float32)
        command=prior.sample(condition,np.array([mode],np.int32),noise);command=mapping.apply(command)
        result=runner.collect_chunk(state,command);state=result.state;traces.append(result.trace)
        if (state.status!=0).all():break
    if not traces:raise RuntimeError("source projection failed at declared source-supported reset")
    data={n:np.concatenate([getattr(t,n) for t in traces],axis=1) for n in StepTrace._fields}
    observed=int(observed_route_modes(data["tcp_position"][:,None],data["valid"][:,None])[0])
    report=dict(success=bool(state.status[0]==1),status=int(state.status[0]),seed=seed,mode=mode,observed_mode=observed,
        final_position_error=float(data["position_error"][0,-1]),final_orientation_error=float(data["orientation_error"][0,-1]),
        clearance_lower_bound=float(data["clearance_lower_bound"].min()),max_joint_speed=float(data["max_joint_speed"].max()),
        seconds=float(state.steps[0]*task.control_dt),source_support_failure=support_fail)
    report["task_cost"]=float(.5*(not report["success"])+.25*min(report["final_position_error"]/task.position_tolerance,1.)+.25*min(report["final_orientation_error"]/task.orientation_tolerance,1.))
    save_bundle(path,data,dict(report,config=task_values,diagnostic_mechanism=mechanism_values,role="frozen learned source, finite impedance map",source_prior_sha256=prior.regressor.metadata["tensor_sha256"]))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/learned_cpu_v2/source_prior"))
    p.add_argument("--deadline",type=float,default=30.);p.add_argument("--seed-start",type=int,default=48000)
    a=p.parse_args();out=a.output;task=ObstacleTransferConfig(robot="xarm7",timeout_seconds=a.deadline,reset_joint_noise=.0005);skill=SkillSpec();truth=MechanismSpec(.65,3,.5)
    bank=[0.,.25,.5,.75,1.];contract=dict(config=asdict(task),diagnostic_truth=asdict(truth),stiffness_candidates=bank,
        source_prior_sha256=load_bundle(a.prior)[1]["tensor_sha256"],source_seeds=[a.seed_start,a.seed_start+1],train_seeds=[a.seed_start+100,a.seed_start+101],test_seeds=list(range(a.seed_start+200,a.seed_start+208)),
        thresholds=dict(success_fraction=.95,mode_fraction=1.),probe_budget=5,target_task_training_interactions=0,
        model_episodes_per_search=20,protocol=f"3D over/around asymmetric load transfer, common {a.deadline:g} s deadline, unchanged collision/hold/pose/speed criteria")
    contract=json.loads(json.dumps(contract))
    if (out/"contract.json").exists():
        if json.loads((out/"contract.json").read_text())!=contract:raise ValueError("obstacle contract mismatch")
    else:publish(out/"contract.json",contract)
    source_jobs=[(str(out/f"source/s{s}_m{m}"),asdict(replace(task,robot="panda")),asdict(MechanismSpec()),str(a.prior),0.,s,m) for s in contract["source_seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:source=list(pool.map(rollout_job,source_jobs))
    if not all(r["success"] and r["observed_mode"]==r["mode"] for r in source):raise RuntimeError("frozen obstacle source qualification failed")
    if not (out/"calibration.json").exists():
        records,ledger=collect_probes(task,skill,truth,out/"probes");estimate,fit=fit_obstacle_context(task,skill,records[:3])
        validation=validate_obstacle_response(task,skill,estimate,records[3:]);nominal=validate_obstacle_response(task,skill,MechanismSpec(),records[3:])
        publish(out/"calibration.json",dict(estimated=asdict(estimate),fit=fit,validation=validation,nominal_validation=nominal,budget=ledger))
    calibration=json.loads((out/"calibration.json").read_text())
    if not calibration["validation"]["supported"]:raise RuntimeError("obstacle calibration validation failed")
    jobs=[(str(out/f"{role}/c{i}/s{s}_m{m}"),asdict(task),mech,str(a.prior),k,s,m)
        for role,mech in (("calibrated_search",calibration["estimated"]),("nominal_search",asdict(MechanismSpec()))) for i,k in enumerate(bank) for s in contract["train_seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(rollout_job,jobs))
    selected={}
    for role in ("calibrated_search","nominal_search"):
        values=[]
        for i,k in enumerate(bank):
            rows=[load_bundle(out/f"{role}/c{i}/s{s}_m{m}")[1] for s in contract["train_seeds"] for m in (0,1)]
            task_cost=float(np.mean([r["task_cost"] for r in rows]));mode_error=float(np.mean([r["mode"]!=r["observed_mode"] for r in rows]))
            values.append(dict(stiffness=k,task=task_cost,joint=.8*task_cost+.15*mode_error+.05*float(np.mean([r["seconds"]/a.deadline for r in rows]))))
        selected[role]={key:min(range(len(bank)),key=lambda i:values[i][key]) for key in ("task","joint")}
        publish(out/f"{role}/search.json",dict(history=values,winners=selected[role],model_episodes=20))
    methods=dict(direct=0,calibrated=selected["calibrated_search"]["joint"],task_only=selected["calibrated_search"]["task"],nominal=selected["nominal_search"]["joint"])
    publish(out/"frozen.json",dict(methods=methods,stiffness={name:bank[i] for name,i in methods.items()},source_prior_sha256=contract["source_prior_sha256"]))
    jobs=[(str(out/f"evaluation/c{i}/s{s}_m{m}"),asdict(task),asdict(truth),str(a.prior),bank[i],s,m) for i in sorted(set(methods.values())) for s in contract["test_seeds"] for m in (0,1)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(rollout_job,jobs))
    evaluations={}
    for name,i in methods.items():
        rows=[load_bundle(out/f"evaluation/c{i}/s{s}_m{m}")[1] for s in contract["test_seeds"] for m in (0,1)]
        evaluations[name]=dict(attempted=len(rows),successes=sum(r["success"] for r in rows),mode_matches=sum(r["observed_mode"]==r["mode"] for r in rows),episodes=rows)
    e=evaluations["calibrated"]
    publish(out/"report.json",dict(complete=True,contract=contract,calibration=calibration,methods=methods,evaluations=evaluations,
        source_successes=4,passed=e["successes"]>=.95*e["attempted"] and e["mode_matches"]==e["attempted"],
        distinct_physical_target_evaluation_episodes=len(jobs),target_task_training_interactions=0,
        scope="CPU full 3D motion mechanism chain; source frozen, same-model task-only and nominal search budget matched; no unconditional superiority claim"))
    print({k:{x:v[x] for x in ("successes","attempted","mode_matches")} for k,v in evaluations.items()},flush=True)


if __name__=="__main__":main()
