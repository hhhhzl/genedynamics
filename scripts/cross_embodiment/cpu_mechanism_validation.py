"""CPU research gates: event-conditioned source, budgeted twin identification.

This is the explicitly declared structured-physics-twin route, separate from
the legacy unconstrained MLP-context experiment. No hidden truth is passed to
the source prior, calibration selector, transport or command decoder.
"""
import argparse
from pathlib import Path
import json
import time
import numpy as np

from genedynamics.learning.cross_embodiment import MechanismSpec,RandomRequest
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactRunner,ContactCommands,research_contact_config
from genedynamics.learning.cross_embodiment.contact_query import observe_contact
from genedynamics.learning.cross_embodiment.contact_behavior import (
    behavior_reset,advance_behavior,teacher_reference,reference_commands,PhaseGaussianPrior,keyway_alignment)
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle,load_bundle


def policy_step(runner,state,memory,prior=None,seed=0,observation_noise=None):
    observation=observe_contact(state,runner,np)
    measured_force=state.measured_force
    if observation_noise is not None:
        from genedynamics.learning.cross_embodiment.core.control_math import integrate_rotation
        sigma=np.asarray(observation_noise,dtype=float)
        if sigma.shape!=(3,) or not np.isfinite(sigma).all() or np.any(sigma<0):
            raise ValueError("observation noise requires nonnegative position/rotation/force standard deviations")
        random=np.random.default_rng(np.random.SeedSequence([seed,int(state.steps[0]),8293]))
        position=observation.tcp_position+random.normal(size=observation.tcp_position.shape)*sigma[0]
        angular=random.normal(size=(len(state.sample_id),3))*sigma[1]
        rotation=np.stack([integrate_rotation(np,r,w,1.) for r,w in zip(observation.tcp_rotation,angular)])
        observation=observation._replace(tcp_position=position,tcp_rotation=rotation)
        measured_force=measured_force+random.normal(size=measured_force.shape)*sigma[2]
    aligned=keyway_alignment(observation.tcp_position,observation.tcp_rotation,runner.config.mating_clearance) if runner.config.socket_profile=="matched_keyway" else None
    if runner.config.guide_profile=="spatial_dual_channel":
        aligned=np.where(memory.mode==0,observation.tcp_position[:,0]>.455,observation.tcp_position[:,1]>-.181)
    memory=advance_behavior(memory,observation.tcp_position,observation.tcp_rotation,
                            measured_force,insertion=runner.config.insertion,dt=10*runner.config.control_dt,goal_z=runner.config.goal[2],
                            alignment_mask=aligned)
    noise=np.random.default_rng(np.random.SeedSequence([seed,int(state.steps[0])])).normal(size=(len(state.sample_id),5)).astype(np.float32)
    reference=teacher_reference(memory,insertion=runner.config.insertion,goal_z=runner.config.goal[2]) if prior is None else np.asarray(prior.reference(memory,noise))
    if runner.config.guide_profile=="spatial_dual_channel":
        # Stop accumulating approach error before the terminal outlet. The
        # integrated low-level reference still has to return to the goal;
        # simply commanding zero velocity would hold an overshot reference.
        capture=(memory.phase==3)&(np.max(np.abs(observation.tcp_position[:,:2]-np.asarray(runner.config.goal)[:2]),axis=-1)<.003)
        integral=memory.tracking_integral.copy();integral[capture,:2]=0.
        memory=memory._replace(tracking_integral=integral)
    command=reference_commands(reference,state.reference_position,state.reference_rotation,
                               observation.tcp_position,observation.tcp_rotation,integral_bias=memory.tracking_integral,dt=runner.config.control_dt)
    return command,memory,reference


def rollout(config,mechanism,*,seed,mode,prior=None,force_scale=1.,angular_scale=1.,transport=None,output=None,command_selector=None,record_carry=False,observation_noise=None,body_selector=None,record_events=False,artifact_context=None,record_snapshots=False):
    runner=ContactRunner(config=config,mechanism=mechanism)
    if record_events:
        if type(runner._impl).__name__!="MujocoContact":raise NotImplementedError("event diagnostics require fixed-body numpy provider")
        runner._impl.record_events=True
    state=runner.reset_batch(RandomRequest(seed,(seed*2+mode,)),initial="jammed" if config.insertion else "clear")
    memory=behavior_reset(1,[mode]); traces=[]; phases=[]; references=[]
    carries=[]; deltas=[]; raw_commands=[]; events=[]
    snapshots=[]
    start=time.monotonic()
    for _ in range(config.timeout_steps//10):
        if record_snapshots:
            if type(runner._impl).__name__!="MujocoContact":raise NotImplementedError("snapshots require fixed-body numpy reference")
            snapshots.append({**{n:np.array(getattr(state,n),copy=True) for n in state._fields if n!="physics"},
                **{"physics."+n:np.array(getattr(state.physics[0],n),copy=True) for n in ("qpos","qvel","qacc_warmstart","ctrl","time")}})
        selected=transport if command_selector is None else command_selector(runner,state,memory,prior,seed)
        command,memory,reference=policy_step(runner,state,memory,prior,seed,observation_noise)
        if selected is not None: command=np.array(selected.apply(command,phase=memory.phase),copy=True)
        command[...,12]*=force_scale; command[...,3:6]*=angular_scale
        if record_carry:
            from genedynamics.learning.cross_embodiment.contact_data import contact_carry
            before=contact_carry(state,runner)
        executed=command if body_selector is None else ContactCommands(command,body_selector(runner,state,memory,command))
        result=runner.collect_chunk(state,executed)
        if record_events:events.append(runner._impl.last_event_trace)
        if record_carry:
            carries.append(before[0]); deltas.append((contact_carry(result.state,runner)-before)[0]);raw_commands.append(command[0])
        state=result.state
        traces.append(result.trace); phases.append(memory.phase.copy()); references.append(reference)
        if np.all(state.status!=0): break
    trace={n:np.concatenate([getattr(t,n) for t in traces],axis=1) for n in traces[0]._fields}
    force=trace["normal_force"][trace["valid"]&(trace["normal_force"]>.3)]
    valid_force=trace["normal_force"][0,trace["valid"][0]]
    final=trace["tcp_position"][0,-1]
    from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
    from genedynamics.learning.cross_embodiment.core.control_math import orientation_distance
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    observed=classify_contact_modes(trace["tcp_position"],trace["normal_force"],trace["valid"],insertion=config.insertion)
    report={"status":int(state.status[0]),"success":bool(state.status[0]==1),"seed":seed,"mode":mode,
        "force_peak":float(trace["force_peak"].max()),"lateral_force_peak":float(trace["lateral_force_peak"].max()),"loaded_force_rmse":float(np.sqrt(np.mean((force-config.desired_force)**2))) if len(force) else None,
        "final_position":final.tolist(),"seconds":float(state.steps[0]*config.control_dt),
        "observed_mode":int(observed[0]),"final_force":float(trace["normal_force"][0,-1]),
        "orientation_error":float(orientation_distance(np,trace["tcp_rotation"][0,-1],rotation())),
        "final_force_rmse":float(np.sqrt(np.mean((valid_force[-15:]-config.desired_force)**2))),
        "wall_seconds":time.monotonic()-start,"phase_counts":np.bincount(np.asarray(phases).ravel(),minlength=4).tolist()}
    from genedynamics.learning.cross_embodiment.research_audit import realized_signature
    report["response_signature"]=realized_signature(trace,phases,config)
    if hasattr(runner._impl,"last_qp_diagnostic"):report["last_qp_diagnostic"]=runner._impl.last_qp_diagnostic
    if output is not None:
        from genedynamics.learning.cross_embodiment.data import code_fingerprint
        extra={} if not record_carry else dict(carry=np.asarray(carries),delta=np.asarray(deltas),raw_command=np.asarray(raw_commands))
        if record_events:extra.update({"event."+name:np.concatenate([e[name] for e in events],axis=1) for name in events[0]})
        if record_snapshots:extra.update({"snapshot."+name:np.stack([s[name] for s in snapshots]) for name in snapshots[0]})
        save_bundle(output,{**trace,"phase":np.asarray(phases),"reference":np.asarray(references),**extra},
                    {**report,"config":config.__dict__,"diagnostic_mechanism":mechanism.__dict__,"code_fingerprint":code_fingerprint(),"role":"development diagnostic",
                     "artifact_context":artifact_context,
                     "observation_noise_std":observation_noise,"observation_noise_scope":"task feedback at 0.2 s; high-rate impedance/wrench feedback unchanged" if observation_noise is not None else None})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--robot",default="panda")
    p.add_argument("--research",action="store_true")
    p.add_argument("--task",default="keyed_insertion_recovery")
    p.add_argument("--gain",type=float,default=1.)
    p.add_argument("--delay",type=int,default=0)
    p.add_argument("--noise",type=float,default=.002)
    p.add_argument("--seeds",type=int,nargs="+",default=[10,11])
    p.add_argument("--force-scale",type=float,default=1.)
    p.add_argument("--prior",type=Path)
    p.add_argument("--backend",choices=("jax","native"),default="native")
    p.add_argument("--device",choices=("cpu",),default="cpu")
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    prior=None
    if args.prior:
        from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
        prior=PhaseGaussianPrior(FrozenRegressor.load(args.prior,backend=args.backend,device=args.device))
    config=ContactConfig(environment=args.task,robot=args.robot,mobile=args.task.startswith("mobile"),reset_task_noise=args.noise,timeout_seconds=30.)
    if args.research: config=research_contact_config(args.robot)
    mechanism=MechanismSpec(args.gain,args.delay,.5)
    results=[]
    for seed in args.seeds:
        for mode in (0,1):
            r=rollout(config,mechanism,seed=seed,mode=mode,prior=prior,force_scale=args.force_scale,output=args.output/f"s{seed}_m{mode}")
            results.append(r); print(r,flush=True)
    (args.output/"report.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__": main()
