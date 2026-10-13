"""Actual MuJoCo CPU evaluation of the small learned ObstacleTransfer chain.

All baselines share resets, source samples and the 13D decoder. Outcomes include
failures. This is a functional experiment, not evidence of method superiority.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.learning.cross_embodiment import CrossEmbodiment,ExecutionSpec,MechanismSpec,ObstacleTransferConfig,RandomRequest,SkillSpec
from genedynamics.learning.cross_embodiment.source_query import SourceQueryAdapter,TaskSemantics
from genedynamics.learning.cross_embodiment.data import load_dataset,observed_route_modes
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,GaussianSkillPrior
from genedynamics.learning.cross_embodiment.learning_data import online_prior_condition,online_response_carry,response_carry
from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv,fit_context
from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
from genedynamics.learning.cross_embodiment.transport import StochasticTransport,TransportSpec,TransportBatch
from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation


def probe_library(horizon,seed):
    rng=np.random.default_rng(seed)
    while True:
        command=np.zeros((1,horizon,13),np.float32)
        command[...,:3]=rng.uniform(-.05,.05,(1,1,3))
        command[...,3:6]=rng.uniform(-.15,.15,(1,1,3))
        command[...,6:12]=rng.uniform(-.3,.3,(1,1,6))
        yield command


def calibrate(response,task,skill,execution,mechanism,budget_count,output):
    runner=CrossEmbodiment(execution=execution,task=task,skill=skill,mechanism=mechanism)
    observer=SourceQueryAdapter(execution=execution,task=replace(task,robot="panda"),skill=skill)
    path=output/f"budget_{budget_count}.json"
    if path.exists(): raise FileExistsError(f"use a fresh output directory: {path}")
    budget=ProbeBudget(budget_count,str(path)); gate=HiddenTargetEnv(runner,observer,budget)
    inputs=[]; targets=[]; weights=[]
    library=probe_library(skill.horizon,100)
    for i in range(budget_count):
        purpose="validation" if i%10>=7 else "fit"
        command=next(library)
        record=gate.probe(RandomRequest(101,(i+1000,)),command,purpose=purpose)
        before=online_response_carry(record.initial_state,record.observation,np)
        after_semantics=observer.observe(runner,record.result.state)
        after=online_response_carry(record.result.state,after_semantics,np)
        trace=record.result.trace
        risks=np.stack((trace.clearance_lower_bound.min(axis=1),trace.max_joint_speed.max(axis=1),trace.saturation.max(axis=1)),axis=-1)
        inputs.append(np.concatenate((before,command.reshape(1,-1),np.array([[0.,1.]])),axis=-1)[0])
        targets.append(np.concatenate((after-before,risks),axis=-1)[0])
        weights.append(float(purpose=="fit" and np.all(np.isfinite(after))))
    gate.seal()
    report={"budget":budget.snapshot(),"fit_examples":int(sum(weights)),"validation_examples":budget_count-int(sum(weights))}
    context=jnp.zeros(3,jnp.float32)
    if budget_count:
        x,y,w=map(lambda a:np.asarray(a,np.float32),(inputs,targets,weights))
        context,loss=fit_context(response,x,y,w,updates=300)
        def error(c,mask):
            prediction=np.asarray(response.predict(jnp.concatenate((jnp.asarray(x),jnp.broadcast_to(c,(len(x),3))),axis=-1)))
            mask=mask.astype(bool)
            return float(np.mean(((prediction[:,mask]-y[mask])/response.outputs.scale)**2)) if mask.any() else None
        report.update({"context":np.asarray(context).tolist(),"validation_before":error(jnp.zeros(3),1-w),
                       "validation_after":error(context,1-w),"fit_loss_final":float(loss[-1])})
        save_bundle(output/f"probes_{budget_count}",{"inputs":x,"targets":y,"fit_mask":w,"context":context},
                    {"artifact_kind":"budgeted_probes","ledger":budget.snapshot(),"hidden_truth_in_features":False})
    return context,report


def train_transport(prior,response,observer,skill,task,updates,data_path,output):
    data,meta=load_dataset(data_path)
    valid=data["trace.valid"].all(axis=-1)&(data["episode.robot"][:,None]==1)&(data["episode.split"][:,None]==0)
    indices=np.argwhere(valid)
    chosen=np.random.default_rng(31).choice(len(indices),size=min(96,len(indices)),replace=False)
    e,k=indices[chosen].T
    semantics=TaskSemantics(*(data["chunk.semantic."+name][e,k] for name in TaskSemantics._fields))
    query=observer.build(semantics,RandomRequest(32,tuple(range(2000,2000+len(e)))))
    source=online_response_carry(query.virtual_state,query.semantics,np)
    target=response_carry(data)[e,k]
    condition=online_prior_condition(query.semantics,np.asarray(task.goal_position),rotation(task.goal_yaw),
                                     data["chunk.start_steps"][e,k]*task.control_dt,jnp)
    mode=jnp.asarray(data["episode.requested_route"][e],dtype=jnp.int32)
    noise=jax.random.normal(jax.random.PRNGKey(33),(len(e),skill.horizon,12))
    commands=prior.sample(condition,mode,noise)
    truth=data["diagnostic.mechanism_truth"][e]
    context=(truth-np.array([1.,0.,.5]))/np.array([.25,4.,.5])
    batch=TransportBatch(jnp.asarray(source),jnp.asarray(target),commands,jnp.asarray(context,dtype=jnp.float32),jnp.asarray(query.support,dtype=jnp.float32))
    transport=StochasticTransport(response,spec=TransportSpec(horizon=skill.horizon,microbatch_size=16))
    state=transport.initialize(34)
    rng=np.random.default_rng(35); losses=[]
    for offset in range(0,updates,20):
        n=min(20,updates-offset)
        ids=jnp.asarray(rng.integers(0,len(e),size=(n,32)))
        batches=jax.tree.map(lambda x:x[ids],batch)
        noises=jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(36),offset),(n,32,skill.horizon,12))
        state,loss=transport.train_chunk(state,batches,noises)
        transport.synchronize(loss); losses.extend(np.asarray(loss).tolist())
    transport._impl.model.save(output/"transport",state.parameters,metadata={"artifact_role":"stochastic_transport",
        "transport_spec":transport.spec.__dict__,"training_data_sha256":meta["tensor_sha256"]})
    return transport,state.parameters,{"updates":updates,"source_supported":int(np.asarray(query.support).sum()),
        "training_examples":len(e),"first_loss":losses[0] if losses else None,"last_loss":losses[-1] if losses else None}


def evaluate(name,prior,transport,parameters,response,context,task,skill,execution,mechanism,output):
    runner=CrossEmbodiment(execution=execution,task=task,skill=skill,mechanism=mechanism)
    observer=SourceQueryAdapter(execution=execution,task=replace(task,robot="panda"),skill=skill)
    request=RandomRequest(41,(3000,3001)); state=runner.reset_batch(request)
    mode=jnp.array([0,1],jnp.int32); traces=[]; support_events=[]; unsupported=np.zeros(2,bool); query_previous=None
    from genedynamics.learning.cross_embodiment.contracts import Status
    for k in range(task.timeout_steps//skill.horizon):
        semantics=observer.observe(runner,state)
        query=observer.build(semantics,request,previous=query_previous); query_previous=query.virtual_state
        support=np.asarray(query.support)
        newly=(~support)&(np.asarray(state.status)==0)
        support_events.append(newly)
        unsupported|=newly
        state=state._replace(status=np.where(newly,Status.SOURCE_SUPPORT_FAILURE,state.status).astype(np.int32))
        condition=online_prior_condition(query.semantics,jnp.asarray(task.goal_position),jnp.asarray(rotation(task.goal_yaw)),
                                         jnp.asarray(state.steps*task.control_dt),jnp)
        # Stateless independent streams: changing transport never advances prior RNG.
        prior_noise=jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(42),k),(2,skill.horizon,12))
        source_commands=prior.sample(condition,mode,prior_noise)
        commands=source_commands
        if name!="direct":
            selected_context=jnp.zeros(3) if name=="uncalibrated" else context
            batch=TransportBatch(jnp.asarray(online_response_carry(query.virtual_state,query.semantics,np)),
                jnp.asarray(online_response_carry(state,semantics,np)),source_commands,
                jnp.broadcast_to(selected_context,(2,3)),jnp.asarray(support,dtype=jnp.float32))
            noise=jnp.zeros((2,skill.horizon,12)) if name=="deterministic" else jax.random.normal(
                jax.random.fold_in(jax.random.PRNGKey(43),k),(2,skill.horizon,12))
            commands=transport.apply(parameters,batch,noise)
        result=runner.evaluate_chunk(state,np.asarray(commands)); state=result.state; traces.append(result.trace)
        if np.all(np.asarray(state.status)!=0): break
    tensors={field:np.stack([np.asarray(getattr(trace,field)) for trace in traces],axis=1) for field in traces[0]._fields}
    tensors["source_support_terminal"]=np.stack(support_events,axis=1)
    observed=observed_route_modes(tensors["tcp_position"],tensors["valid"])
    report={"baseline":name,"attempted":2,"status":np.asarray(state.status).tolist(),
            "successes":int((np.asarray(state.status)==1).sum()),"source_support_failures":int(unsupported.sum()),
            "requested_modes":[0,1],"observed_modes":observed.tolist(),"valid_steps":int(tensors["valid"].sum()),
            "final_position_error":tensors["position_error"][:,-1,-1].tolist(),
            "terminal_events":(tensors["terminal"].sum(axis=(1,2))+tensors["source_support_terminal"].sum(axis=1)).tolist(),"gpu_qualification":"not_run"}
    save_bundle(output/name,tensors,{"artifact_kind":"learned_physical_evaluation",**report,"scene":runner.describe()})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,default=Path("tmp/cross_embodiment/learned_cpu"))
    parser.add_argument("--response-data",type=Path,default=Path("tmp/cross_embodiment/obstacle_transfer_response_full"))
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/learned_closed_loop"))
    parser.add_argument("--budgets",type=int,nargs="+",default=[0,10,30,100])
    parser.add_argument("--transport-updates",type=int,default=300)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    prior=GaussianSkillPrior(FrozenRegressor.load(args.models/"source_prior"))
    response=FrozenRegressor.load(args.models/"response")
    task=ObstacleTransferConfig(robot="xarm7",timeout_seconds=30.,reset_joint_noise=0.)
    skill=SkillSpec(horizon=prior.horizon); execution=ExecutionSpec(backend="numpy",physics_backend="mujoco")
    # Withheld from the four public development combinations.
    mechanism=MechanismSpec(.87,3,.7)
    contexts={}; calibration={}
    for budget in args.budgets:
        contexts[budget],calibration[budget]=calibrate(response,task,skill,execution,mechanism,budget,args.output)
    observer=SourceQueryAdapter(execution=execution,task=replace(task,robot="panda"),skill=skill)
    transport,parameters,training=train_transport(prior,response,observer,skill,task,args.transport_updates,args.response_data,args.output)
    selected=max(args.budgets); context=contexts[selected]
    reports=[evaluate(name,prior,transport,parameters,response,context,task,skill,execution,mechanism,args.output)
             for name in ("direct","uncalibrated","deterministic","full")]
    report={"calibration":calibration,"transport_training":training,"evaluation":reports,
            "scope":"CPU functional learned chain; two episodes per baseline, no statistical superiority claim"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
