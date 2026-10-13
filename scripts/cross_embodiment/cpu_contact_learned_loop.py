"""Common CPU learned loop on fixed and planar-mobile contact embodiments.

Source contact prior is frozen nominal Panda. Only budgeted probes calibrate a
withheld target mechanism. Physical failures remain in baseline denominators.
"""
import argparse
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as jp
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactRunner,ContactCommands
from genedynamics.learning.cross_embodiment.contact_query import ContactSourceQuery,ContactSemantics,observe_contact
from genedynamics.learning.cross_embodiment.contact_data import contact_carry,pad_contact_carry,body_descriptor,load_contact_dataset,SEMANTIC_SLICE
from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
from genedynamics.learning.cross_embodiment import MechanismSpec,RandomRequest
from genedynamics.learning.cross_embodiment.learned import GaussianSkillPrior,FrozenRegressor
from genedynamics.learning.cross_embodiment.learning_data import online_prior_condition
from genedynamics.learning.cross_embodiment.transport import StochasticTransport,TransportSpec,TransportBatch
from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv,fit_context
from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle,load_bundle
from genedynamics.learning.cross_embodiment.random_streams import jax_normal,RandomStream
from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation


class Observer:
    @staticmethod
    def observe(runner,state): return observe_contact(state,runner,np)


def calibrate(response,config,mechanism,budget_count,output,resume=False):
    runner=ContactRunner(config=config,mechanism=mechanism)
    ledger=output/"probe_ledger.json"
    if ledger.exists() and not resume: raise FileExistsError("use --resume or a fresh output; calibration probes must not be replayed for free")
    gate=HiddenTargetEnv(runner,Observer(),ProbeBudget(budget_count,str(ledger)))
    rows=[]; cache=output/"probe_records"
    contract={"response_sha256":response.metadata["tensor_sha256"],"config":asdict(config),"mechanism":asdict(mechanism)}
    contract=json.loads(json.dumps(contract))
    contract_path=output/"calibration_contract"
    if resume and (contract_path/"metadata.json").exists():
        _,old_contract=load_bundle(contract_path)
        if old_contract.get("contract")!=contract: raise ValueError("calibration resume contract mismatch")
    elif not resume:
        save_bundle(contract_path,{}, {"contract":contract})
    if resume and (cache/"metadata.json").exists():
        old,provenance=load_bundle(cache)
        if provenance.get("contract")!=contract: raise ValueError("probe response/config/diagnostic mechanism mismatch")
        rows=list(zip(old["inputs"],old["targets"],old["fit_mask"]))
    for i in range(budget_count):
        if i<len(gate.budget.attempts): continue
        rng=np.random.default_rng(110+i)
        commands=np.zeros((1,10,13),np.float32)
        commands[...,:3]=rng.uniform(-.005,.005,(1,1,3))
        commands[...,3:6]=rng.uniform(-.02,.02,(1,1,3))
        commands[...,6:12]=rng.uniform(-.2,.2,(1,1,6))
        commands[...,12]=rng.uniform(2.,6.)
        purpose="validation" if i%10>=7 else "fit"
        record=gate.probe(RandomRequest(111,(50000+i,)),commands,purpose=purpose)
        before=pad_contact_carry(contact_carry(record.initial_state,runner),response.metadata["layout"]["carry_dim"])
        after=pad_contact_carry(contact_carry(record.result.state,runner),response.metadata["layout"]["carry_dim"])
        trace=record.result.trace
        risks=np.array([[trace.force_peak.max(),trace.contact_lost.max(),trace.forbidden_collision.max()]],np.float32)
        x=np.concatenate((before,commands.reshape(1,-1),body_descriptor(config)[None]),axis=-1)[0]
        y=np.concatenate((after-before,risks),axis=-1)[0]
        rows.append((x,y,float(purpose=="fit")))
        save_bundle(cache,{name:np.asarray([r[j] for r in rows],np.float32) for j,name in enumerate(("inputs","targets","fit_mask"))},
                    {"contract":contract,"charged_attempts":len(gate.budget.attempts)})
    gate.seal(); context=jp.zeros(3,jp.float32)
    report={"ledger":gate.budget.snapshot(),"calibration_validation":None}
    if rows:
        x,y,w=(np.asarray([r[i] for r in rows],np.float32) for i in range(3))
        context,loss=fit_context(response,x,y,w,updates=200)
        mask=w==0
        def error(c):
            p=np.asarray(response.predict(jp.concatenate((jp.asarray(x),jp.broadcast_to(c,(len(x),3))),axis=-1)))
            return float(np.mean(((p[:,mask]-y[mask])/response.outputs.scale)**2)) if mask.any() else None
        report["calibration_validation"]={"before":error(jp.zeros(3)),"after":error(context)}
        save_bundle(output/"probes",{"inputs":x,"targets":y,"fit_mask":w,"context":context},report)
    return context,report


def train(prior,response,query,config,path,updates,output,resume=False,calibrated_context=None):
    data,meta=load_contact_dataset(path)
    data["carry"]=pad_contact_carry(data["carry"],response.metadata["layout"]["carry_dim"])
    eligible=data["valid"].all(axis=-1)&(data["group"][:,None]!=0)&(data["body"][:,0,None]!=0)
    candidates=np.argwhere(eligible)
    chosen=np.random.default_rng(112).choice(len(candidates),min(96,len(candidates)),replace=False)
    e,k=candidates[chosen].T; carry=data["carry"][e,k]; start,end=SEMANTIC_SLICE
    semantic=ContactSemantics(carry[:,start:start+3],carry[:,start+3:start+12].reshape(-1,3,3),
        carry[:,start+12:end],carry[:,193],(carry[:,193]>.3).astype(np.int32))
    result=query.build(semantic,RandomRequest(113,tuple(range(51000,51000+len(e)))))
    source=pad_contact_carry(contact_carry(result.state,query.source),response.metadata["layout"]["carry_dim"])
    condition=online_prior_condition(semantic,jp.asarray(config.goal),jp.asarray(rotation()),jp.asarray(k*.2),jp)
    command=prior.sample(condition,jp.asarray(data["mode"][e],dtype=jp.int32),
        jax.random.normal(jax.random.PRNGKey(114),(len(e),10,13)))
    context=(data["mechanism"][e]-np.array([1.,0.,.5]))/np.array([.25,4.,.5])
    batch=TransportBatch(jp.asarray(source),jp.asarray(carry),command,jp.asarray(context,dtype=jp.float32),
        jp.asarray(result.support,dtype=jp.float32),jp.broadcast_to(jp.array([0.,0.,1.]),(len(e),3)),jp.asarray(data["body"][e]))
    transport=StochasticTransport(response,spec=TransportSpec(horizon=10,microbatch_size=16,residual_scale=.05,change_weight=.1))
    state=transport.initialize(115); history=[]; rng=np.random.default_rng(116)
    from genedynamics.learning.cross_embodiment.checkpoint import save_training_checkpoint,load_training_checkpoint
    checkpoint=output/"transport_resume"; start=0
    if resume and (checkpoint/"metadata.json").exists():
        state,_,_,saved=load_training_checkpoint(checkpoint,transport._impl.model)
        if saved.get("response_sha256")!=response.metadata["tensor_sha256"] or saved.get("data_sha256")!=meta["tensor_sha256"]:
            raise ValueError("transport resume provenance mismatch")
        start=int(np.asarray(state.step)); history=saved["loss_history"]
    if start>updates or (start%20 and start!=updates): raise ValueError("resume must preserve transport chunk boundaries")
    for offset in range(0,updates,20):
        length=min(20,updates-offset); ids=jp.asarray(rng.integers(0,len(e),(length,32)))
        if offset<start: continue
        batches=jax.tree.map(lambda x:x[ids],batch)
        noises=jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(117),offset),(length,32,10,13))
        state,loss=transport.train_chunk(state,batches,noises); transport.synchronize(loss); history.extend(np.asarray(loss).tolist())
        ledger_path=output/"probe_ledger.json"
        ledger=ProbeBudget(json.loads(ledger_path.read_text())["limit"],str(ledger_path)) if ledger_path.exists() else ProbeBudget(0)
        save_training_checkpoint(checkpoint,state,context=np.zeros(3,np.float32) if calibrated_context is None else calibrated_context,multipliers=np.array([transport.spec.risk_multiplier],np.float32),
            rng_seed=117,rng_counter=offset+length,budget=ledger,
            metadata={"response_sha256":response.metadata["tensor_sha256"],"data_sha256":meta["tensor_sha256"],"loss_history":history})
    transport.save(output/"transport",state.parameters)
    return transport,state.parameters,{"examples":len(e),"source_supported":int(np.sum(result.support)),
        "first_loss":history[0],"last_loss":history[-1],"updates":updates,"public_data_sha256":meta["tensor_sha256"]}


def evaluate(name,prior,response,transport,parameters,context,config,mechanism,output):
    runner=ContactRunner(config=config,mechanism=mechanism); query=ContactSourceQuery(config)
    request=RandomRequest(121,(52000,52001)); state=runner.reset_batch(request,initial="jammed" if config.insertion else "clear")
    histories=[]; support_history=[]; support_events=[]; previous=None; body_history=[]
    for k in range(config.timeout_steps//10):
        semantic=observe_contact(state,runner,np)
        projection=query.build(semantic,request,previous=previous); previous=projection.state
        failed=~projection.support&(state.status==0)
        state=state._replace(status=np.where(failed,7,state.status)); support_history.append(projection.support); support_events.append(failed)
        condition=online_prior_condition(semantic,jp.asarray(config.goal),jp.asarray(rotation()),jp.asarray(state.steps*config.control_dt),jp)
        source=prior.sample(condition,jp.array([0,1],jp.int32),jax_normal(122,jp.asarray(request.sample_ids,jp.uint32),k,RandomStream.PRIOR_SKILL,(10,13)))
        command=source
        if name!="direct":
            selected=jp.zeros(3) if name=="uncalibrated" else context
            d=response.metadata["layout"]["carry_dim"]
            batch=TransportBatch(jp.asarray(pad_contact_carry(contact_carry(projection.state,query.source),d)),jp.asarray(pad_contact_carry(contact_carry(state,runner),d)),
                source,jp.broadcast_to(selected,(2,3)),jp.asarray(projection.support,dtype=jp.float32),
                jp.broadcast_to(jp.array([0.,0.,1.]),(2,3)),jp.broadcast_to(jp.asarray(body_descriptor(config)),(2,3)))
            noise=jp.zeros((2,10,13)) if name=="deterministic" else jax_normal(123,jp.asarray(request.sample_ids,jp.uint32),k,RandomStream.TRANSPORT,(10,13))
            command=transport.apply(parameters,batch,noise)
        body=np.zeros((2,10,3),np.float32)
        if config.mobile and k<20: body[...,0]=.005
        result=runner.collect_chunk(state,ContactCommands(np.asarray(command),body)); state=result.state; histories.append(result.trace)
        if config.mobile: body_history.append(np.stack([d.qpos[:3] for d in state.physics]))
        if np.all(state.status!=0): break
    tensors={name:np.stack([getattr(trace,name) for trace in histories],axis=1) for name in histories[0]._fields}
    tensors["source_support"]=np.stack(support_history,axis=1); tensors["source_support_terminal"]=np.stack(support_events,axis=1)
    if body_history: tensors["body_qpos"]=np.stack(body_history,axis=1)
    observed=classify_contact_modes(tensors["tcp_position"],tensors["normal_force"],tensors["valid"],insertion=config.insertion)
    report={"baseline":name,"environment":config.environment,"robot":config.robot,"attempted":2,"status":state.status.tolist(),
        "successes":int((state.status==1).sum()),"observed_modes":observed.tolist(),"requested_modes":[0,1],
        "force_peak":float(tensors["force_peak"].max()),"valid_steps":int(tensors["valid"].sum()),
        "terminal_events":(tensors["terminal"].sum(axis=(1,2))+tensors["source_support_terminal"].sum(axis=1)).tolist(),
        "gpu_qualification":"not_run","diagnostic_mechanism":[mechanism.servo_gain,mechanism.delay_steps,mechanism.payload_mass],
        "source_prior_sha256":prior.regressor.metadata["tensor_sha256"],"response_sha256":response.metadata["tensor_sha256"],
        "evaluation_role":"exploratory CPU qualification; not an untouched statistical test set"}
    save_bundle(output/name,tensors,{**report,"scene":runner.describe()})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,required=True)
    parser.add_argument("--response-data",type=Path,required=True)
    parser.add_argument("--task",default="guided_contact_alignment")
    parser.add_argument("--robot",default="xarm7")
    parser.add_argument("--budget",type=int,default=10)
    parser.add_argument("--updates",type=int,default=200)
    parser.add_argument("--gain",type=float,default=.95)
    parser.add_argument("--delay",type=int,default=1)
    parser.add_argument("--mass",type=float,default=.55)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    config=ContactConfig(environment=args.task,robot=args.robot,mobile=args.task.startswith("mobile"))
    prior=GaussianSkillPrior(FrozenRegressor.load(args.models/"source_prior")); response=FrozenRegressor.load(args.models/"response")
    mechanism=MechanismSpec(args.gain,args.delay,args.mass)
    context,calibration=calibrate(response,config,mechanism,args.budget,args.output,resume=args.resume)
    query=ContactSourceQuery(config)
    transport,parameters,training=train(prior,response,query,config,args.response_data,args.updates,args.output,resume=args.resume,calibrated_context=context)
    evaluations=[evaluate(name,prior,response,transport,parameters,context,config,mechanism,args.output)
                 for name in ("direct","uncalibrated","deterministic","full")]
    report={"calibration":calibration,"transport_training":training,"evaluation":evaluations,
            "scope":"CPU small learned contact chain, failures retained; no method superiority claim"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print({"training":training,"evaluation":evaluations})


if __name__=="__main__": main()
