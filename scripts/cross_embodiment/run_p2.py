"""Bounded GPU protocol qualification and model-only finite-study collection.

MuJoCo CPU remains the target reference. A GPU model is never silently
substituted for a target or for the independent CPU model-validation gate.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from genedynamics.experiments.artifacts import ArtifactPolicy, POLICY_ENV, atomic_json, canonical_json
from genedynamics.experiments.process import run_stage
from scripts.research.environment import ROOT, environment


def prepare(gpu=False):
    runtime=Path(os.environ.get('GENEDYNAMICS_RUNTIME',ROOT/'.runtime/cross_ebdm')).resolve()
    os.environ.update(environment(runtime))
    os.environ.update(JAX_PLATFORMS='cuda,cpu' if gpu else 'cpu',MUJOCO_GL='egl',
                      OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',JAX_DEFAULT_MATMUL_PRECISION='highest',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
    os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(path,value):
    value=canonical_json(value)
    if Path(path).exists():
        if read(path)!=value:raise ValueError(f'immutable P2 contract changed: {path}')
    else:atomic_json(path,value)
    return value


def cases(parent):
    """Declared historical development fixtures, never a new test split."""
    bank=parent/'cross_ebdm_recovery_distribution'
    feasibility=parent/'cross_ebdm_recovery_feasibility'
    selected=read(feasibility/'level_xarm7_derated/seed_7100/frozen.json')['methods']['full']['index']
    matched=bank/'level_xarm7_matched/seed_8100/frozen_maps'
    derated=feasibility/'level_xarm7_derated/seed_7100/frozen_maps'/f'c{selected}'
    def case(name,robot,gain,delay,seed,mode,mapping,expected):
        return dict(name=name,robot=robot,gain=gain,delay=delay,seed=seed,mode=mode,
                    mapping=None if mapping is None else str(mapping.resolve()),expected_status=expected)
    return [case('source_mode0','panda',1.,0,66100200,0,None,1),
            case('source_mode1','panda',1.,0,66100200,1,None,1),
            case('matched_success','xarm7',1.,0,67100303,0,matched/'c4',1),
            case('matched_late_loss','xarm7',1.,0,67100303,0,matched/'c15',8),
            case('matched_prefix_force','xarm7',1.,0,67100303,1,matched/'c17',9),
            case('matched_direct_loss','xarm7',1.,0,66100200,0,None,8),
            case('matched_direct_timeout','xarm7',1.,0,66100200,1,None,5),
            case('derated_mode0','xarm7',.65,3,66100200,0,derated,1),
            case('derated_mode1','xarm7',.65,3,66100200,1,derated,1)]


def freeze_cpu(contract):
    import numpy as np
    from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec
    from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    from scripts.cross_embodiment.cpu_mechanism_validation import rollout
    root=Path(contract['config']['output_dir']);prior=PhaseGaussianPrior(FrozenRegressor.load(contract['prior']['path'],backend='native'))
    reuse=contract['config']['method_params'].get('cpu_fixture_root')
    if reuse:
        reference=Path(reuse);origin=read(reference/'workflow_contract.json')
        if origin['cases']!=contract['cases'] or origin['prior']!=contract['prior'] or origin['assets']!=contract['assets']:
            raise ValueError('reused CPU fixture protocol mismatch')
        stable=[n for n in contract['code']['files'] if n.startswith('genedynamics/') or n=='scripts/cross_embodiment/cpu_mechanism_validation.py']
        if any(origin['code']['files'].get(n)!=contract['code']['files'][n] for n in stable):
            raise ValueError('CPU fixture numerical sources changed')
        records=[]
        for case in contract['cases']:
            data,meta=load_bundle(reference/'cpu_fixtures'/case['name'])
            attempt=read(reference/'cpu_fixtures'/case['name']/'attempt.json')
            if meta['code_fingerprint']['aggregate']!=origin['code']['aggregate'] or attempt['status']!='complete' or attempt['tensor_sha256']!=meta['tensor_sha256']:
                raise ValueError('reused CPU fixture provenance mismatch')
            records.append(dict(name=case['name'],status=meta['status'],tensor_sha256=meta['tensor_sha256'],
                wall_seconds=meta['wall_seconds'],snapshots=len(data['snapshot.steps'])))
        report=dict(complete=True,records=records,new_target_diagnostic_episodes=0,new_source_episodes=0,new_probes=0,
            reused_target_diagnostic_episodes=7,reused_source_episodes=2,reference=str(reference),
            reference_contract_sha256=digest(reference/'workflow_contract.json'))
        atomic_json(root/'cpu_report.json',report);return report
    records=[]
    for case in contract['cases']:
        path=root/'cpu_fixtures'/case['name'];attempt=path/'attempt.json'
        if not (path/'metadata.json').exists():
            if attempt.exists():raise RuntimeError('incomplete charged CPU fixture; no free replay')
            atomic_json(attempt,dict(status='pending',request=case,role='source' if case['robot']=='panda' else 'target_development_qualification'))
            mapping=None if case['mapping'] is None else ReferenceTransport.load(case['mapping'],backend='native')
            rollout(mating_contact_config(case['robot']),ContactMechanismSpec(case['gain'],case['delay']),
                    seed=case['seed'],mode=case['mode'],prior=prior,transport=mapping,output=path,
                    record_events=True,record_carry=True,record_snapshots=True,
                    artifact_context=dict(protocol='p2_cpu_fixture_v1',request=case))
            _,meta=load_bundle(path)
            atomic_json(attempt,dict(status='complete',request=case,tensor_sha256=meta['tensor_sha256']))
        data,meta=load_bundle(path)
        if meta['code_fingerprint']['aggregate']!=contract['code']['aggregate']:raise ValueError('CPU fixture source changed')
        if meta['status']!=case['expected_status']:raise RuntimeError('frozen historical CPU status did not reproduce')
        records.append(dict(name=case['name'],status=meta['status'],tensor_sha256=meta['tensor_sha256'],
                            wall_seconds=meta['wall_seconds'],snapshots=len(data['snapshot.steps'])))
    report=dict(complete=True,records=records,new_target_diagnostic_episodes=7,new_source_episodes=2,new_probes=0)
    atomic_json(root/'cpu_report.json',report)
    return report


def gpu_report(trace,phases,status,steps,case,config,wall):
    import numpy as np
    from genedynamics.learning.cross_embodiment.contact_reference import classify_contact_modes
    from genedynamics.learning.cross_embodiment.core.control_math import orientation_distance
    from genedynamics.envs.domains.manipulation.cross_embodiment.workcell import rotation
    from genedynamics.learning.cross_embodiment.research_audit import realized_signature
    count=int(trace['valid'].sum());last=max(0,count-1)
    loaded=trace['normal_force'][trace['valid']&(trace['normal_force']>.3)]
    valid=trace['normal_force'][0,trace['valid'][0]]
    return dict(status=int(status),success=bool(status==1),seed=case['seed'],mode=case['mode'],
        observed_mode=int(classify_contact_modes(trace['tcp_position'],trace['normal_force'],trace['valid'],insertion=True)[0]),
        force_peak=float(trace['force_peak'].max()),lateral_force_peak=float(trace['lateral_force_peak'].max()),
        final_position=trace['tcp_position'][0,last].tolist(),final_force=float(trace['normal_force'][0,last]),
        seconds=float(steps*config.control_dt),orientation_error=float(orientation_distance(np,trace['tcp_rotation'][0,last],rotation())),
        final_force_rmse=float(np.sqrt(np.mean((valid[-15:]-config.desired_force)**2))) if len(valid) else 5.,
        loaded_force_rmse=float(np.sqrt(np.mean((loaded-config.desired_force)**2))) if len(loaded) else None,
        phase_counts=np.bincount(phases[:(count+9)//10].ravel(),minlength=4).tolist(),
        response_signature=realized_signature(trace,phases,config),wall_seconds=wall,
        physics_provider='mjx-jax',physics_dtype='float64',target_reference_provider='mujoco-numpy')


def gpu_qualification(contract):
    import numpy as np
    import jax
    from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactRunner,ContactMechanismSpec
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
    from genedynamics.learning.cross_embodiment.backends.jax.phase_contact import JaxPhaseContactPolicy,affine_bank,replay_noise
    from genedynamics.learning.cross_embodiment.contracts import RandomRequest
    root=Path(contract['config']['output_dir']);params=contract['config']['method_params']
    cpu_root=Path(params.get('cpu_fixture_root',root))
    device=params['gpu_device'];prior=PhaseGaussianPrior(FrozenRegressor.load(contract['prior']['path'],backend='jax',device=device))
    groups={}
    for case in contract['cases']:groups.setdefault((case['robot'],case['gain'],case['delay']),[]).append(case)
    records=[];operator=[];benchmarks=[];batch_checks=[]
    for (robot,gain,delay),group in groups.items():
        config=mating_contact_config(robot);runner=ContactRunner(config=config,mechanism=ContactMechanismSpec(gain,delay),backend='jax',device=device)
        policy=JaxPhaseContactPolicy(runner,prior);dev=runner._impl.device
        # Fixed shape, declared repeats are benchmark/model calls, not new target access.
        padded=(group*((params['qualification_batch']+len(group)-1)//len(group)))[:params['qualification_batch']]
        seeds=[c['seed'] for c in padded];modes=[c['mode'] for c in padded]
        state=policy.reset_episodes(seeds,modes);memory=policy.initial_memory(modes)
        weights,biases=affine_bank([c['mapping'] for c in padded])
        noise,weights,biases=jax.device_put((replay_noise(seeds,75),weights,biases),dev)
        start=time.monotonic();result=policy.run_chunk(state,memory,noise,weights,biases);runner.synchronize(result);cold=time.monotonic()-start
        start=time.monotonic()
        with jax.transfer_guard('disallow'):
            warm=policy.run_chunk(state,memory,noise,weights,biases);runner.synchronize(warm)
        seconds=time.monotonic()-start
        cold_host=jax.device_get(result);host=jax.device_get(warm)
        deterministic=all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(cold_host),jax.tree.leaves(host)))
        batch_checks.append(dict(robot=robot,gain=gain,delay=delay,cold_warm_equal=deterministic,transfer_guard=True))
        for i,case in enumerate(group):
            data={n:np.asarray(getattr(host.trace,n)[i:i+1]) for n in host.trace._fields}
            phases=np.asarray(host.phases[:,i:i+1]);report=gpu_report(data,phases,host.state.status[i],host.state.steps[i],case,config,seconds/len(padded))
            cpu,meta=load_bundle(cpu_root/'cpu_fixtures'/case['name'])
            comparison=dict(name=case['name'],cpu_status=meta['status'],gpu_status=report['status'],
                status_matches=meta['status']==report['status'],terminal_seconds_error=abs(meta['seconds']-report['seconds']),
                final_position_error_m=float(np.max(np.abs(np.asarray(meta['final_position'])-report['final_position']))),
                peak_force_error_N=abs(meta['force_peak']-report['force_peak']))
            limits=contract['thresholds'];comparison['long_reference_passed']=(comparison['status_matches'] and
                comparison['terminal_seconds_error']<=limits['terminal_seconds'] and
                comparison['final_position_error_m']<=limits['final_position_m'] and
                comparison['peak_force_error_N']<=max(limits['peak_force_abs_N'],limits['peak_force_relative']*meta['force_peak']))
            save_bundle(root/'gpu_fixtures'/case['name'],{**data,'phase':phases,'reference':np.asarray(host.references[:,i:i+1]),
                        'raw_command':np.asarray(host.raw_commands[i:i+1]),
                        **{'event.'+n:np.asarray(v[i:i+1]) for n,v in host.events.items()}},
                        dict(**report,config=asdict(config),diagnostic_mechanism=asdict(ContactMechanismSpec(gain,delay)),
                             code_fingerprint=contract['code'],artifact_context=dict(protocol='p2_gpu_fixture_v1',request=case)))
            records.append(comparison)
            # Same-state tests decouple operator errors from accumulated contact divergence.
            indices=sorted(set([0,min(2,len(cpu['snapshot.steps'])-1),len(cpu['snapshot.steps'])-1]))
            for k in indices:
                snap={n.removeprefix('snapshot.'):v[k] for n,v in cpu.items() if n.startswith('snapshot.')}
                imported=runner._impl.import_snapshot(snap);runner.synchronize(imported)
                measure=jax.jit(jax.vmap(runner._impl._measure),device=dev)
                wrench,peak,touch,bad=jax.device_get(measure(imported.physics))
                import mujoco
                native=mujoco.MjData(runner._impl.reference.model)
                for n in ('qpos','qvel','qacc_warmstart','ctrl'):getattr(native,n)[:]=snap['physics.'+n]
                native.time=float(snap['physics.time']);mujoco.mj_forward(runner._impl.reference.model,native)
                expected=runner._impl.reference._measure(native)
                axial=jax.device_get(jax.jit(jax.vmap(runner._impl._axial_support),device=dev)(imported.physics))[0]
                force_error=float(np.max(np.abs(wrench[0,:3]-expected[0][:3])))
                moment_error=float(np.max(np.abs(wrench[0,3:]-expected[0][3:])))
                # Independently apply the NumPy reference's contact-point
                # wrench convention to MJX's own solved contact forces. This
                # separates a sign/frame/TCP implementation error from a
                # difference between the two contact solvers.
                from mujoco.mjx._src import support
                force_query=jax.jit(lambda d:support.contact_force_dim(runner._impl.model,d,3)[0],device=dev)
                single=jax.tree.map(lambda a:a[0],imported.physics)
                forces=np.asarray(force_query(single));indices=support.contact_force_dim(runner._impl.model,single,3)[1]
                contact=jax.device_get(single.contact);tcp=np.asarray(single.site_xpos[runner._impl.sid]);reduced=np.zeros(6)
                for force,index in zip(forces,indices):
                    pair=tuple(map(int,contact.geom[index]))
                    if min(pair)<0 or contact.dist[index]>0 or tuple(sorted(pair)) not in runner._impl.reference.allowed:continue
                    world=(force.reshape(2,3)@contact.frame[index]).ravel()
                    world*=1. if runner._impl.reference.model.geom(pair[1]).name in {'contact_peg','contact_key'} else -1.
                    reduced+=np.r_[world[:3],world[3:]+np.cross(contact.pos[index]-tcp,world[:3])]
                semantics_error=float(np.max(np.abs(reduced-wrench[0])))
                # Geometry and solved force are reported separately. MJX is an
                # approximate response model if its solver changes a long task.
                operator.append(dict(name=case['name'],window=k,force_error_N=force_error,moment_error_Nm=moment_error,
                    wrench_semantics_max_error=semantics_error,wrench_semantics_passed=semantics_error<=limits['wrench_reduction_abs'],
                    axial_matches=bool(axial)==runner._impl.reference._axial_support(native),
                    forbidden_matches=bool(bad[0])==bool(expected[3]),
                    passed=force_error<=limits['checkpoint_force_N'] and moment_error<=limits['checkpoint_moment_Nm'] and
                           bool(axial)==runner._impl.reference._axial_support(native) and bool(bad[0])==bool(expected[3])))
        # Chunk scheduling/masks must preserve the exact GPU law, independently
        # of CPU/MJX physics equivalence.
        runner._impl.microbatch_size=params['qualification_batch']//2
        sliced=policy.run_chunk(state,memory,noise,weights,biases);runner.synchronize(sliced)
        sliced=jax.device_get(sliced)
        differences=[float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.)) for a,b in zip(jax.tree.leaves(host),jax.tree.leaves(sliced))]
        batch_checks[-1]['microbatch_max_error']=max(differences)
        batch_checks[-1]['microbatch_passed']=max(differences)<=limits['microbatch_abs']
        runner._impl.microbatch_size=None
        request=RandomRequest(991,(301,302));reset=runner.reset_batch(request,initial='jammed')
        command=jax.device_put(np.zeros((2,2,13),np.float32),dev)
        advanced=runner.collect_chunk(reset,command);runner.synchronize(advanced)
        partial=runner.reset_batch(request,previous=advanced.state,mask=np.array([True,False]),initial='jammed');runner.synchronize(partial)
        batch_checks[-1]['partial_reset_passed']=int(partial.steps[0])==0 and int(partial.steps[1])==2 and np.array_equal(np.asarray(partial.physics.qpos[1]),np.asarray(advanced.state.physics.qpos[1]))
        reordered=runner.reset_batch(RandomRequest(991,(302,301)),initial='jammed');runner.synchronize(reordered)
        batch_checks[-1]['reset_reordering_passed']=np.array_equal(np.asarray(reset.physics.qpos),np.asarray(reordered.physics.qpos)[::-1])
        steps=np.asarray(host.state.steps);terminal=np.asarray(host.trace.terminal);valid=np.asarray(host.trace.valid)
        batch_checks[-1]['terminal_masks_passed']=bool(np.array_equal(valid.sum(axis=1),steps) and
            np.array_equal(terminal.sum(axis=1),(np.asarray(host.state.status)!=0).astype(int)) and
            all(np.array_equal(v,np.arange(len(v))<n) for v,n in zip(valid,steps)))
        batch_checks[-1]['delay_queue_passed']=bool(np.all(np.asarray(advanced.trace.command)==0.)) if delay>=2 else True
        stats=dev.memory_stats() or {}
        cpu_seconds=sum(read(cpu_root/'cpu_fixtures'/c['name']/'metadata.json')['wall_seconds'] for c in padded)
        benchmarks.append(dict(robot=robot,gain=gain,delay=delay,batch=len(padded),windows=75,
            cold_compile_and_run_seconds=cold,warm_seconds=seconds,warm_episodes_per_second=len(padded)/seconds,
            valid_control_steps=int(np.asarray(host.trace.valid).sum()),cpu_serial_fixture_seconds=cpu_seconds,
            speedup_over_serial_cpu=cpu_seconds/seconds,memory_stats=stats,
            scope='full fixed-horizon model execution; CPU comparison includes reference IO/controller, not parallel CPU throughput'))
        atomic_json(root/'gpu_progress.json',dict(records=records,operators=operator,batch_checks=batch_checks,benchmarks=benchmarks))
    semantic=all(r['wrench_semantics_passed'] and r['axial_matches'] and r['forbidden_matches'] for r in operator)
    batching=all(r['cold_warm_equal'] and r['microbatch_passed'] and r['partial_reset_passed'] and
                 r['reset_reordering_passed'] and r['terminal_masks_passed'] and r['delay_queue_passed'] for r in batch_checks)
    report=dict(complete=True,device=str(prior.regressor.model._impl.device),records=records,operator_checks=operator,
                batch_checks=batch_checks,benchmarks=benchmarks,semantic_operator_qualified=semantic,
                same_state_cpu_force_qualified=all(r['passed'] for r in operator),
                batch_execution_qualified=batching,cpu_reference_equivalent=all(r['long_reference_passed'] for r in records),
                model_search_qualified=semantic and batching,
                target_reference_substitution_qualified=semantic and batching and all(r['long_reference_passed'] for r in records),
                required_cpu_model_validation=True,new_target_evaluation_episodes=0,new_probes=0,
                gpu_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    atomic_json(root/'gpu_report.json',report)
    print({k:report[k] for k in ('semantic_operator_qualified','batch_execution_qualified','cpu_reference_equivalent','model_search_qualified')},flush=True)
    return report


def p2(config,resume=False):
    prepare();from genedynamics.learning.cross_embodiment.data import code_fingerprint
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    from scripts.cross_embodiment.run_pilot import asset_manifest
    root=config.output_dir;params=config.method_params
    root.mkdir(parents=True,exist_ok=True)
    artifact=ArtifactPolicy(str(root),params['max_artifact_bytes'],params['min_free_bytes'])
    policy_path=root/'artifact_policy.json';atomic_json(policy_path,asdict(artifact),enforce_policy=False);os.environ[POLICY_ENV]=str(policy_path)
    _,prior=load_bundle(params['prior']);fixtures=cases(Path(params['p1_parent']))
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv'],text=True)
    contract=dict(config=config.to_dict(),code=code_fingerprint(),cases=fixtures,
        prior=dict(path=str(Path(params['prior']).resolve()),sha256=prior['tensor_sha256']),assets=asset_manifest(),gpu_inventory=inventory,
        versions={n:importlib.metadata.version(n) for n in ('numpy','mujoco','mujoco-mjx','jax','jaxlib')},
        mappings={c['mapping']:digest(Path(c['mapping'])/'metadata.json') for c in fixtures if c['mapping']},
        thresholds=dict(terminal_seconds=.4,final_position_m=.0005,peak_force_abs_N=.5,peak_force_relative=.1,
                        checkpoint_force_N=.15,checkpoint_moment_Nm=.015,microbatch_abs=1e-7,wrench_reduction_abs=1e-9),
        scope='development qualification; frozen CPU target reference; GPU model search requires independent CPU validation')
    import zipfile
    from genedynamics.experiments.artifacts import artifact_write
    overlay=root/'_environment/source-overlay.zip'
    if not overlay.exists():
        names=set(contract['code']['files'])
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/'configs/arm/keyed_insertion_recovery').rglob('*.yaml'))
        names.update(('requirements/cross-embodiment.lock.txt','scripts/research/environment.py',
                      'genedynamics/experiments/artifacts.py','genedynamics/experiments/process.py','genedynamics/experiments/runner.py'))
        with artifact_write(overlay,sum((ROOT/n).stat().st_size for n in names)+65536):
            overlay.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(overlay,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in sorted(names):archive.write(ROOT/name,name)
    with zipfile.ZipFile(overlay) as archive:
        if any(hashlib.sha256(archive.read(n)).hexdigest()!=h for n,h in contract['code']['files'].items()):
            raise ValueError('P2 retained source overlay changed')
    contract['source_overlay_sha256']=digest(overlay)
    contract['runtime']=dict(python=sys.version,requirements_lock_sha256=digest(ROOT/'requirements/cross-embodiment.lock.txt'),
                            native_library_sha256=digest(Path(os.environ['CROSS_EMBODIMENT_NATIVE_LIBRARY'])))
    if (root/'workflow_contract.json').exists() and not resume:raise FileExistsError('P2 exists; use --resume with unchanged contract')
    immutable(root/'workflow_contract.json',contract)
    if resume and (root/'report.json').exists():return read(root/'report.json')
    for stage,gpu in (('cpu_fixtures',False),('gpu_qualification',True)):
        complete=root/('cpu_report.json' if not gpu else 'gpu_report.json')
        if complete.exists() and read(complete).get('complete'):continue
        row=run_stage(stage,[sys.executable,'-m',__spec__.name,'--worker-contract',str(root/'workflow_contract.json'),'--stage',stage],root/'_logs',timeout=params['stage_timeout_seconds'])
        if row['status']!='passed':raise RuntimeError(f'P2 {stage} failed; inspect bounded log')
    cpu=read(root/'cpu_report.json');gpu=read(root/'gpu_report.json')
    report=dict(complete=True,cpu=cpu,gpu=gpu,status='qualified_model_search' if gpu['model_search_qualified'] else 'qualification_failed',
                target_reference_substitution_qualified=gpu['target_reference_substitution_qualified'])
    atomic_json(root/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path,nargs='?');parser.add_argument('--development-root',type=Path)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--worker-contract',type=Path)
    parser.add_argument('--stage',choices=('cpu_fixtures','gpu_qualification'))
    args=parser.parse_args();prepare(args.stage=='gpu_qualification')
    if args.worker_contract:
        from genedynamics.learning.cross_embodiment.data import code_fingerprint
        contract=read(args.worker_contract)
        if contract['code']['aggregate']!=code_fingerprint()['aggregate']:raise ValueError('P2 worker source changed')
        if args.stage=='cpu_fixtures':freeze_cpu(contract)
        else:
            import jax
            with jax.experimental.enable_x64(),jax.default_matmul_precision('highest'):gpu_qualification(contract)
        return
    from genedynamics.experiments.framework.config import ExperimentConfig
    config=ExperimentConfig.from_yaml(args.config)
    if args.development_root:config.use_development_output_root(args.development_root)
    if config.validate():parser.error('; '.join(config.validate()))
    print(json.dumps(p2(config,resume=args.resume),indent=2))


if __name__=='__main__':main()
