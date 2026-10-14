"""Schedule independent frozen model requests; no new target access.

Nominal requests identical except for output path may reuse a committed
physical result. The alias proof and physical-call counts remain explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from genedynamics.experiments.artifacts import atomic_json
from scripts.cross_embodiment.run_refinement import read,checksum


def aliases(root):
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
    manifest=read(root/'_manifests/search.json');groups={}
    for job in manifest:
        if job['role']!='training_nominal':continue
        identity=checksum({k:v for k,v in job.items() if k!='path'})
        groups.setdefault(identity,[]).append(job)
    pairs=[]
    for identity,jobs in groups.items():
        if len(jobs)!=2:raise ValueError('nominal alias must have exactly two identical condition requests')
        source,target=sorted(jobs,key=lambda j:j['path'])
        if {k:v for k,v in source.items() if k!='path'}!={k:v for k,v in target.items() if k!='path'}:
            raise ValueError('physical model inputs differ; cannot alias')
        pairs.append((source,target))
    plan=root/'_nominal_aliases';plan.mkdir(exist_ok=True)
    atomic_json(plan/'plan.json',dict(pairs=[dict(source=s,target=t) for s,t in pairs],
        source_manifest_sha256=hashlib.sha256((root/'_manifests/search.json').read_bytes()).hexdigest(),
        rule='exact equality of every request field except output path; nominal/public model only'))
    pending=list(pairs);records=[];physical_targets=[];started=time.monotonic()
    while pending:
        left=[]
        for source,target in pending:
            src,dst=Path(source['path']),Path(target['path'])
            if not (src/'metadata.json').exists() or not (src/'attempt.json').exists() or read(src/'attempt.json')['status']!='complete':
                left.append((source,target));continue
            data,meta=load_bundle(src)
            if meta['artifact_context']['request_sha256']!=checksum(source) or meta.get('physics_provider')!='mjx-jax':
                raise ValueError('alias source provenance mismatch')
            if (dst/'metadata.json').exists():
                _,other=load_bundle(dst)
                if other['artifact_context']['request_sha256']!=checksum(target):raise ValueError('alias target changed')
                existing=other.get('physical_alias')
                if existing:
                    if existing['source']!=str(src) or existing['source_tensor_sha256']!=meta['tensor_sha256'] or other['tensor_sha256']!=meta['tensor_sha256']:
                        raise ValueError('existing nominal alias source changed')
                    records.append(dict(target=str(dst),**existing))
                else:physical_targets.append(str(dst))
                continue
            if (dst/'attempt.json').exists():
                # Never replace a charged request. Wait for its original
                # producer and count it as physical rather than as an alias.
                left.append((source,target));continue
            alias=dict(source=str(src),source_request_sha256=checksum(source),source_tensor_sha256=meta['tensor_sha256'],
                       input_equality_except_path=True,new_physical_model_calls=0)
            atomic_json(dst/'attempt.json',dict(status='pending_alias',request_sha256=checksum(target),physical_alias=alias))
            metadata={k:v for k,v in meta.items() if k not in ('schema_version','linear_weight_layout','tensor_specs','tensor_sha256','tensor_file')}
            metadata['artifact_context']=dict(meta['artifact_context'],request_sha256=checksum(target))
            metadata['physical_alias']=alias
            save_bundle(dst,data,metadata);_,saved=load_bundle(dst)
            if saved['tensor_sha256']!=meta['tensor_sha256']:raise ValueError('identical model alias changed tensor bytes')
            atomic_json(dst/'attempt.json',dict(status='complete',request_sha256=checksum(target),
                tensor_sha256=saved['tensor_sha256'],role=target['role'],physical_alias=alias))
            records.append(dict(target=str(dst),**alias))
        pending=left
        atomic_json(plan/'report.json',dict(complete=not pending,logical_aliases=len(records),new_physical_model_calls=0,
            remaining=len(pending),records=records,original_physical_targets=physical_targets,
            analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        if pending:
            if time.monotonic()-started>14400:raise TimeoutError('bounded nominal alias wait expired')
            time.sleep(2)


def shard(root,replay):
    from genedynamics.experiments.process import run_stage
    if not read(replay)['passed']:raise ValueError('second physical GPU replay not qualified')
    jobs=[j for j in read(root/'_manifests/search.json') if j['role']=='training_calibrated' and 'level_xarm7_matched' in Path(j['path']).parts]
    if len(jobs)!=1152:raise ValueError('unexpected independent matched calibration shard size')
    if any((Path(j['path'])/'attempt.json').exists() for j in jobs):raise RuntimeError('shard already claimed')
    output=root/'_parallel_gpu0';output.mkdir(exist_ok=True)
    manifest=output/'manifest.json';atomic_json(manifest,jobs)
    plan=dict(physical_gpu=0,cuda_visible_devices='0',logical_device='gpu:0',episodes=len(jobs),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        replay_report=str(replay),replay_report_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
        main_worker_behavior='uses cached complete requests with exact context/code checks; no duplicate execution',
        scope='three independent calibration instances, each executed on a single card; no distributed learner')
    atomic_json(output/'plan.json',plan)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    stage=run_stage('gpu0_calibrated_models',[sys.executable,'-m','scripts.cross_embodiment.gpu_finite_collection',
        '--manifest',str(manifest),'--root',str(output),'--batch-size','64','--device','gpu:0'],output/'_logs',env=env,timeout=14400)
    if stage['status']!='passed':raise RuntimeError('independent GPU0 shard failed')
    atomic_json(output/'scheduling-report.json',dict(complete=True,plan=plan,stage=stage,
        collection_report_sha256=hashlib.sha256((output/'gpu_collection_report.json').read_bytes()).hexdigest()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('aliases','shard'),required=True);p.add_argument('--replay',type=Path)
    args=p.parse_args()
    if args.mode=='aliases':aliases(args.root)
    else:shard(args.root,args.replay)


if __name__=='__main__':main()
