"""Counted free-space/contact probes and an independent structured-twin check."""
import argparse
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from genedynamics.learning.cross_embodiment import RandomRequest,MechanismSpec
from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactConfig,ContactTrace,ContactChunk
from genedynamics.learning.cross_embodiment.calibration import HiddenTargetEnv,ProbeRecord
from genedynamics.learning.cross_embodiment.core.budget import ProbeBudget
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle,load_bundle
from genedynamics.learning.cross_embodiment.structured_response import fit_twin_context,validate_twin,response_features
from genedynamics.learning.cross_embodiment.contact_query import observe_contact


class Observer:
    @staticmethod
    def observe(runner,state):return observe_contact(state,runner,np)


def probe_policy(kind,amplitude,*,tangential=False):
    def policy(observation,state,index):
        command=np.zeros((1,10,13),np.float32)
        if kind=="free":
            command[...,:2]=[.025*(-1 if index//2%2 else 1),.012*(-1 if index//3%2 else 1)]
            command[...,5]=.12*(-1 if index//2%2 else 1)
        else:
            command[...,12]=amplitude*(1. if index<6 else .65)
            if tangential:
                command[...,0]=.003*(-1 if index//2%2 else 1)
                command[...,1]=.002*(-1 if index//3%2 else 1)
                command[...,5]=.025*(-1 if index//2%2 else 1)
        return command
    return policy


def collect_probes(config,mechanism,output,*,design="contact",seed=400,tangential=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    gate=HiddenTargetEnv(ContactRunner(config=config,mechanism=mechanism),Observer(),ProbeBudget(5,str(output/"ledger.json")))
    records=[]
    def initial_pose(kind):
        if kind=="free" and config.guide_profile=="spatial_dual_channel":return "raised"
        return "jammed" if kind=="loaded" and config.insertion else "clear"
    contract={"config":asdict(config),"design":design,"seed":seed,"diagnostic_mechanism":asdict(mechanism),"probe_protocol":"excited_contact_tangent_v3" if tangential else "excited_free_contact_v2"}
    contract_path=output/"contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract: raise ValueError("probe resume contract mismatch")
    else: contract_path.write_text(json.dumps(contract,sort_keys=True)+"\n")
    kinds=[("free",0.),("loaded",3.),("loaded",7.),("loaded",4.),("free",0.)]
    if design=="free": kinds=[("free",0.)]*3+kinds[3:]
    for i,(kind,amplitude) in enumerate(kinds):
        path=output/f"probe_{i}"
        if i<len(gate.budget.attempts):
            if not (path/"metadata.json").exists():
                raise RuntimeError("charged probe lacks committed record; it cannot be replayed for free")
            data,meta=load_bundle(path)
            runner=ContactRunner(config=config)
            state=runner.reset_batch(RandomRequest(seed+i,(seed*10+i,)),initial=initial_pose(kind))
            state.physics[0].qpos[:]=data["initial_qpos"];state.physics[0].qvel[:]=data["initial_qvel"]
            import mujoco
            mujoco.mj_forward(runner._impl.model,state.physics[0])
            state=state._replace(**{n:data["initial_"+n] for n in state._fields if n!="physics"})
            trace=ContactTrace(*(data["trace_"+n] for n in ContactTrace._fields))
            records.append(ProbeRecord(i,meta["purpose"],Observer.observe(runner,state),data["commands"],ContactChunk(state,trace),state))
            continue
        record=gate.probe_policy(RandomRequest(seed+i,(seed*10+i,)),probe_policy(kind,amplitude,tangential=tangential),chunks=10,
                                initial=initial_pose(kind),purpose="fit" if i<3 else "validation")
        records.append(record)
        save_bundle(output/f"probe_{i}",{"commands":record.commands,"response":response_features(record.result.trace),
            **{"trace_"+n:getattr(record.result.trace,n) for n in ContactTrace._fields},
            **{"initial_"+n:getattr(record.initial_state,n) for n in record.initial_state._fields if n!="physics"},
            "valid":record.result.trace.valid,"initial_qpos":record.initial_state.physics[0].qpos.copy(),
            "initial_qvel":record.initial_state.physics[0].qvel.copy()},
            {"purpose":record.purpose,"kind":kind,"control_steps":int(record.result.trace.valid.sum()),
             "contact_steps":int(((record.result.trace.normal_force>.3)&record.result.trace.valid).sum()),
             "peak_force":float(record.result.trace.force_peak.max()),"final_status":int(record.result.state.status[0])})
    gate.seal()
    return records,gate.budget.snapshot()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--robot",default="xarm7");p.add_argument("--gain",type=float,default=.65);p.add_argument("--delay",type=int,default=3)
    p.add_argument("--design",choices=("free","contact"),default="contact");p.add_argument("--seed",type=int,default=400)
    a=p.parse_args();config=ContactConfig(environment="keyed_insertion_recovery",robot=a.robot,reset_task_noise=.002)
    records,ledger=collect_probes(config,MechanismSpec(a.gain,a.delay,.5),a.output,design=a.design,seed=a.seed)
    context,fit=fit_twin_context(config,records[:3])
    validation=validate_twin(config,context,records[3:]);nominal=validate_twin(config,MechanismSpec(),records[3:])
    report={"estimated":asdict(context),"fit":fit,"validation":validation,"nominal_validation":nominal,"budget":ledger,
            "diagnostic_truth":{"gain":a.gain,"delay":a.delay,"mass":.5},"role":"development mechanism diagnostic; validation excluded from fitting"}
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print({k:v for k,v in report.items() if k!="fit"},flush=True)


if __name__=="__main__":main()
