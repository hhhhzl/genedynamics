"""Run actual arm loading/sliding/release and keyed withdrawal/retry fixtures."""
import argparse
from pathlib import Path
import json
import numpy as np
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactRunner,ContactCommands
from genedynamics.learning.cross_embodiment import RandomRequest
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=["numpy","jax"],default="numpy")
    parser.add_argument("--robot",choices=["panda","xarm7","ur5e","go2_z1"],default="panda")
    parser.add_argument("--task",choices=["guided_contact_alignment","keyed_insertion_recovery","mobile_keyed_insertion_recovery"],default="guided_contact_alignment")
    parser.add_argument("--initial",choices=["clear","jammed"],default="clear")
    parser.add_argument("--route",choices=["x_then_y","y_then_x"],default="x_then_y")
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/contact_smoke"))
    args=parser.parse_args()
    config=ContactConfig(environment=args.task,robot=args.robot,mobile=args.task.startswith("mobile"))
    runner=ContactRunner(config=config,backend=args.backend)
    state=runner.reset_batch(RandomRequest(61,(6000,)),initial=args.initial)
    histories=[]
    for k in range(config.timeout_steps//10):
        t=k*10*config.control_dt
        command=np.zeros((1,10,13),np.float32)
        # Public geometric diagnostic teacher. No claim of a learned contact prior.
        if config.insertion and args.initial=="jammed" and t<2.:
            command[...,2]=.015
            command[...,5]=-.06
        else:
            command[...,12]=config.desired_force
            if config.insertion:
                position=np.asarray(state.reference_position)[0]
                command[...,:2]=np.clip(2.*(np.array(config.goal[:2])-position[:2]),-.02,.02)
            elif t>1.:
                position=np.asarray(state.reference_position)[0]
                waypoint=np.array(config.goal[:2])
                if t<3.: waypoint[1 if args.route=="x_then_y" else 0]=(-.20 if args.route=="x_then_y" else .38)
                command[...,:2]=np.clip(2.*(waypoint-position[:2]),-.04,.04)
        if config.mobile:
            body=np.zeros((1,10,3),np.float32)
            if t<4.: body[...,0]=.005
            command=ContactCommands(command,body)
        result=runner.collect_chunk(state,command); runner.synchronize(result)
        state=result.state; histories.append(result.trace)
        if np.all(np.asarray(state.status)!=0): break
    tensors={name:np.stack([np.asarray(getattr(trace,name)) for trace in histories],axis=1) for name in histories[0]._fields}
    if config.mobile:
        tensors["final_body_qpos"]=np.asarray(state.physics.qpos[:,:3]) if hasattr(state.physics,"qpos") else np.stack([d.qpos[:3] for d in state.physics])
    report={"task":config.environment,"robot":config.robot,"initial":args.initial,"route":args.route,
        "status":np.asarray(state.status).tolist(),"force_peak":float(tensors["force_peak"].max()),
        "contact_steps":int(tensors["allowed_contact"].sum()),"terminal_events":int(tensors["terminal"].sum()),
        "final_tcp":tensors["tcp_position"][0,-1,-1].tolist(),"final_normal_force":float(tensors["normal_force"][0,-1,-1]),
        "scope":"physical scripted contact qualification; contact learning chain not yet qualified"}
    save_bundle(args.output,tensors,{**report,"scene":runner.describe()})
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
