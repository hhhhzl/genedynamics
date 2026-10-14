"""Consolidate both retained protocols and their revised theoretical claims."""
import argparse
from pathlib import Path
import os
import numpy as np

from genedynamics.experiments.artifacts import POLICY_ENV,atomic_json
from scripts.cross_embodiment.run_refinement import read,checksum
from scripts.paper.cross_ebdm.summarize_refinement import text_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,required=True)
    args=p.parse_args();root=args.validation_root.resolve()
    names=['cross_ebdm_feasibility','cross_ebdm_distribution',
           'cross_ebdm_recovery_feasibility','cross_ebdm_recovery_distribution']
    reports={};audits={}
    for name in names:
        reports[name]=read(root/name/'scientific_report.json')
        audits[name]=read(root/name/'_reports/independent-audit.json')
        if not reports[name]['complete'] or not audits[name]['passed']:
            raise ValueError('all retained protocols must be complete and independently audited')
    diagnostic=read(root/'cross_ebdm_diagnostics/diagnostic-report.json')
    replay=read(root/'cross_ebdm_diagnostics/replay-report.json')
    prefix=read(root/'cross_ebdm_diagnostics/prefix-model-report.json')
    output=root/names[-1];os.environ[POLICY_ENV]=str(output/'artifact_policy.json')
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    from scripts.cross_embodiment.audit_validation_closure import inspect
    diagnostic_checks=[]
    for manifest in (root/'cross_ebdm_diagnostics/_manifests').glob('*.json'):
        for job in read(manifest):
            data,meta=load_bundle(job['path'])
            expected=dict(request_sha256=checksum(job),role=job['role'],protocol=job.get('protocol','finite_refinement_v1'))
            if meta['artifact_context']!=expected or meta['code_fingerprint']['aggregate']!=job['code']:
                raise ValueError('diagnostic collection provenance mismatch')
            checked=inspect(Path(job['path']))
            if checked['reported']!=checked['accepted'] or not checked['mode_classification_matches']:
                raise ValueError('diagnostic physical acceptance mismatch')
            diagnostic_checks.append(dict(path=job['path'],role=job['role'],tensor_sha256=meta['tensor_sha256']))
    total_target=sum(r['actual_target_evaluation_episodes'] for r in reports.values())+replay['diagnostic_target_episodes']
    total_probes=sum(r['new_probes'] for r in reports.values())
    new_model=sum(sum(v for k,v in a['role_counts'].items() if k not in ('evaluation',)) for a in audits.values())+prefix['model_episodes']
    latest=reports[names[-1]]
    claims=[dict(claim='Calibration predicts the declared finite class',status='controlled_finite_evidence',
                 evidence=f'P1.5 {diagnostic["status_matches"]}/{diagnostic["episodes"]} paired status matches',
                 revision='Separate calibration error from candidate approximation and task feasibility.'),
            dict(claim='Suffix-only maps can repair prefix failures',status='refuted_in_the_declared_stage_subclass',
                 evidence=f'6/6 identical failing prefixes; {prefix["recovery_successes"]}/18 recovery models succeed on one known reset',
                 revision='Add a prefix-invariance success upper bound; restrict its empirical scope to this model diagnostic.'),
            dict(claim='Response anchoring exceeds same-class task-only control',status='not_demonstrated_task_only_matches_or_exceeds_success',
                 evidence=latest['totals'],revision='Replace dominance language with same-class task-budget and response tradeoff bounds.'),
            dict(claim='Matching marginals certifies source coupling',status='false',
                 evidence='Balanced coupling permutation preserves target traces and frequencies while changing source correspondence.',
                 revision='Analyze the joint source/target law separately from target marginal and task success.'),
            dict(claim='Keeping every source mode is always preferable',status='false_when_the_algorithm_stops_unsupported_mass',
                 evidence='Compare fixed-law and support-reweighted policies, with every refusal retained.',
                 revision='State support/stop-mass and TV/KL tradeoffs; empirical support is not global infeasibility.'),
            dict(claim='Extra transport noise is required',status='not_demonstrated',
                 evidence='Fixed small-noise perturbation only, no trained same-capacity stochastic-kernel comparison.',
                 revision='For separable expected costs, random mixtures cannot beat the best deterministic conditional action.')]
    program=dict(complete=True,protocols={n:dict(stage=r['stage'],totals=r['totals'],
                 full_cells_qualified=r['all_full_cells_qualified'],increment=r['transport_incremental_value'],
                 distinct_target_episodes=r['actual_target_evaluation_episodes'],new_probes=r['new_probes'],audit=audits[n])
                 for n,r in reports.items()},new_target_task_episodes=total_target,new_probes=total_probes,
                 model_and_source_episodes=new_model,diagnostic_target_episodes=replay['diagnostic_target_episodes'],
                 model_count_scope='manifested task/source episodes; calibration fitting short probe-prediction calls are retained in fit records and excluded from this count',
                 independent_calibrations=6,claims=claims,
                 diagnostic_audit=dict(passed=True,episodes=len(diagnostic_checks),records=diagnostic_checks),
                 scope='P1.5/P1.6/P1.7 development program; v2 designed from v1, same six calibrations; not independent confirmation or GPU/real-robot evidence')
    atomic_json(output/'_reports/program-report.json',program)
    lines=['# P1.5–P1.7 completed development program','',
           '| Protocol | Condition | Full | Task-only | Full + reweight | Task-only + reweight |',
           '| --- | --- | --- | --- | --- | --- |']
    for name,r in reports.items():
        for suite in sorted({v['suite'] for v in r['replicates']}):
            values=[]
            for method in ('full','task_only','full_reweighted','task_only_reweighted'):
                entries=[v['results'][method]['summary'] for v in r['replicates'] if v['suite']==suite and method in v['results']]
                values.append(f'{sum(e["successes"] for e in entries)}/{sum(e["attempted"] for e in entries)}' if entries else '—')
            lines.append('| '+name+' | '+suite+' | '+' | '.join(values)+' |')
    lines+=['',f'New counted probes: {total_probes}. New distinct target task episodes: {total_target}, including {replay["diagnostic_target_episodes"]} diagnostic replays.',
            'Six independent calibrations total; v2 explicitly reuses them. All rejections remain in requested denominators.',
            '', '| Claim | Result | Theory change |','| --- | --- | --- |']
    lines += ['| '+c['claim']+' | '+c['status']+' | '+c['revision']+' |' for c in claims]
    lines += ['',program['scope']]
    text_file(output/'_reports/program-report.md','\n'.join(lines)+'\n')
    # Actual paired model diagnostic: six identical prefixes versus recovery.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for index in (0,1,2,3,4,5,6,12,18):
        row=prefix['records'][index];d,m=load_bundle(row['path']);v=d['valid'][0].astype(bool)
        t=np.arange(1,v.sum()+1)*m['config']['control_dt'];color='#ac3b38' if index<6 else '#245b78'
        axes[0].plot(t,d['force_peak'][0,v],color=color,alpha=.35 if index<6 else .8,
                     label='Suffix-only (six identical)' if index==0 else 'Recovery enabled' if index==6 else None)
        axes[1].plot(t,d['tcp_position'][0,v,2],color=color,alpha=.35 if index<6 else .8)
    axes[0].axhline(25,color='black',ls='--',lw=1);axes[0].set_ylabel('Substep peak total contact force (N)')
    axes[1].set_ylabel('TCP z (m)')
    for ax in axes:ax.set_xlabel('Time (s)')
    axes[0].legend(fontsize=8);fig.suptitle('One known reset, calibrated-model diagnosis; no new target access')
    fig.tight_layout();target=output/'_reports/prefix-diagnosis.png'
    from genedynamics.experiments.artifacts import artifact_write
    with artifact_write(target,4*1024**2):fig.savefig(target,dpi=180)
    plt.close(fig)
    print({k:program[k] for k in ('complete','new_target_task_episodes','new_probes','independent_calibrations')})


if __name__=='__main__':main()
