"""Read-only P2 provenance, CPU/GPU discrepancy and independent task audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.audit_validation_closure import inspect
from scripts.cross_embodiment.run_refinement import read,checksum,splits
from scripts.cross_embodiment.run_theory_feasibility import costs
from scripts.paper.cross_ebdm.report_exports import (
    add_report_export_argument, preflight_report_exports,
)


def qualification(root):
    contract=read(root/'workflow_contract.json');report=read(root/'gpu_report.json')
    cpu_root=Path(contract['config']['method_params'].get('cpu_fixture_root',root))
    origin=read(cpu_root/'workflow_contract.json');errors=[];records=[]
    with zipfile.ZipFile(root/'_environment/source-overlay.zip') as archive:
        for name,digest in contract['code']['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:errors.append(dict(error='source overlay changed',path=name))
    for case in contract['cases']:
        for provider,parent,code in (('cpu',cpu_root,origin['code']['aggregate']),('gpu',root,contract['code']['aggregate'])):
            path=parent/(provider+'_fixtures')/case['name'];data,meta=load_bundle(path)
            row=inspect(path)
            if meta['code_fingerprint']['aggregate']!=code or row['reported']!=row['accepted'] or not row['mode_classification_matches']:
                errors.append(dict(error='qualification trace/provenance mismatch',path=str(path)))
            valid=data['valid'][0].astype(bool);steps=round(meta['seconds']/meta['config']['control_dt'])
            if valid.sum()!=steps or data['terminal'].sum()!=int(meta['status']!=0):
                errors.append(dict(error='qualification mask mismatch',path=str(path)))
            records.append(dict(name=case['name'],provider=provider,status=meta['status'],tensor_sha256=meta['tensor_sha256']))
    task_errors=[]
    for case in contract['cases']:
        cpu=load_bundle(cpu_root/'cpu_fixtures'/case['name'])[1]
        gpu=load_bundle(root/'gpu_fixtures'/case['name'])[1]
        a,b=[costs(m,np.zeros(24))['task'] for m in (cpu,gpu)]
        task_errors.append(dict(name=case['name'],cpu_task_cost=a,gpu_task_cost=b,absolute_error=abs(a-b)))
    result=dict(passed=not errors,records=records,errors=errors,traces=len(records),
                model_search_qualified=report['model_search_qualified'],
                cpu_reference_equivalent=report['cpu_reference_equivalent'],
                observed_engine_task_errors=task_errors,max_observed_engine_task_error=max(r['absolute_error'] for r in task_errors),
                engine_error_scope='nine selected historical development fixtures; not uniform or population error bound',
                scope='artifact/acceptance consistency; engine-equivalence and scientific gates remain separate')
    admission=root/'model_search_admission.json'
    if admission.exists():
        value=read(admission)
        if value['original_qualification_report_sha256']!=hashlib.sha256((root/'gpu_report.json').read_bytes()).hexdigest():
            raise ValueError('original qualification report changed after restricted admission')
        for row in value['witness_checks']:
            data,meta=load_bundle(root/'_reports/witness_checkpoints'/f'{row["name"]}_k{row["window"]}')
            valid=np.min(data['geom'],axis=1)>=0
            axial=bool(np.any(valid&data['allowed']&(data['dist']<=0)&(np.abs(data['normal'][:,2])>.7)))
            bad=bool(np.any(valid&~data['allowed']&(data['dist'] < -1e-4)))
            if axial!=row['axial'] or bad!=row['forbidden'] or not row['passed']:raise ValueError('restricted witness audit mismatch')
        result['restricted_model_search_admitted']=value['model_search_qualified']
        result['restricted_admission_sha256']=hashlib.sha256(admission.read_bytes()).hexdigest()
    atomic_json(root/'_reports/independent-audit.json',result)
    if errors:raise RuntimeError('P2 qualification artifact audit failed')
    return result


def independent_task_band(root):
    """No candidate reselection: inspect the held-out CPU model-validation set."""
    contract=read(root/'workflow_contract.json');report=read(root/'scientific_report.json')
    params=contract['config']['method_params'];rows=[]
    for replicate in report['replicates']:
        folder=root/('level_'+replicate['suite'])/f'seed_{replicate["seed"]}'
        methods=replicate['methods'];conditional=params['stage']=='1.7'
        full=methods['full']['indices'] if conditional else [methods['full']['index']]*2
        task=methods['task_only']['indices'] if conditional else [methods['task_only']['index']]*2
        differences=[];groups=[]
        for seed in splits(replicate['seed'],params)['model_validation']:
            group=[]
            for mode in (0,1):
                if full[mode] is None or task[mode] is None:continue
                paths=[folder/'model_validation_calibrated'/f'c{index}'/f's{seed}_m{mode}' for index in (full[mode],task[mode])]
                values=[costs(load_bundle(path)[1],np.zeros(24))['task'] for path in paths]
                group.append(values[0]-values[1])
            if len(group)==2:groups.append(seed);differences.append(float(np.mean(group)))
        mean=float(np.mean(differences)) if differences else None
        radius=float(np.sqrt(2*np.log(1/.05)/len(differences))) if differences else None
        same=full==task
        rows.append(dict(suite=replicate['suite'],calibration_seed=replicate['seed'],clusters=len(differences),
            full_minus_task_cpu_model_mean=mean,training_task_slack=params['task_slack'],
            empirical_band_holds=None if mean is None else mean<=params['task_slack'],
            same_frozen_maps=same,structural_task_equality=same,
            paired_hoeffding_radius=radius,certified_noninferiority=same or (False if mean is None else mean+radius<=params['task_slack']),
            certificate_basis='identical frozen maps and source law' if same else 'finite paired Hoeffding bound',
            reset_groups=groups,differences=differences,
            scope='selected maps before feasibility rejection; independent CPU calibrated model, not target or uniform class certificate'))
    atomic_json(root/'_reports/cpu-model-task-band.json',dict(records=rows,
        scope='read-only held-out analysis; never changes frozen selection/support or target schedule'))
    return rows


def audit_gpu_search(root):
    contract=read(root/'workflow_contract.json');manifest=read(root/'_manifests/search.json');errors=[];statuses=Counter()
    providers=Counter();code=contract['code']['aggregate'];jobs={j['path']:j for j in manifest};aliases=0
    for job in manifest:
        data,meta=load_bundle(job['path']);context=dict(request_sha256=checksum(job),role=job['role'],protocol=job.get('protocol','finite_refinement_v1'))
        attempt=read(Path(job['path'])/'attempt.json');row=inspect(job['path'])
        if (meta.get('artifact_context')!=context or meta['code_fingerprint']['aggregate']!=code or
                attempt['status']!='complete' or attempt['tensor_sha256']!=meta['tensor_sha256'] or
                row['reported']!=row['accepted'] or not row['mode_classification_matches'] or meta.get('physics_provider')!='mjx-jax'):
            errors.append(dict(error='GPU model trace/request mismatch',path=job['path']))
        statuses[str(meta['status'])]+=1;providers[meta.get('physics_provider','unknown')]+=1
        alias=meta.get('physical_alias')
        if alias:
            aliases+=1;source=jobs.get(alias['source'])
            if (source is None or source['role']!='training_nominal' or job['role']!='training_nominal' or
                    {k:v for k,v in source.items() if k!='path'}!={k:v for k,v in job.items() if k!='path'}):
                errors.append(dict(error='nominal alias model inputs differ',path=job['path']));continue
            _,original=load_bundle(alias['source'])
            if (original['tensor_sha256']!=meta['tensor_sha256'] or original['tensor_sha256']!=alias['source_tensor_sha256'] or
                    read(Path(alias['source'])/'attempt.json')['status']!='complete'):
                errors.append(dict(error='nominal alias source changed',path=job['path']))
    production=[root/'gpu_collection_report.json',root/'_parallel_gpu0/gpu_collection_report.json']
    calls=sum(read(p)['new_physical_model_calls'] for p in production if p.exists())
    padding=sum(read(p)['padding_model_calls'] for p in production if p.exists())
    if calls-padding!=len(manifest)-aliases:errors.append(dict(error='GPU physical-call accounting mismatch'))
    result=dict(passed=not errors,episodes=len(manifest),physical_model_calls=calls,padding_model_calls=padding,
        logical_nominal_aliases=aliases,statuses=dict(statuses),providers=dict(providers),errors=errors)
    atomic_json(root/'_reports/gpu-search-audit.json',result)
    if errors:raise RuntimeError('GPU model search audit failed')
    return result


def common_execution(root):
    """Separate response comparison on common deployments from stopping cost."""
    science=read(root/'scientific_report.json');pairs=[]
    for replicate in science['replicates']:
        left=replicate['results']['full']['episodes'];right=replicate['results']['task_only']['episodes']
        def identity(r):return (r['seed'],r['source_mode'] if 'source_mode' in r else r['mode'])
        lookup={identity(r):r for r in right}
        for full in left:
            task=lookup[identity(full)];f=not full.get('unsupported');t=not task.get('unsupported')
            pairs.append(dict(suite=replicate['suite'],calibration_seed=replicate['seed'],reset=full['seed'],
                source_mode=identity(full)[1],full_executed=f,task_executed=t,
                full_cost=full['costs']['task'] if f else 1.,task_cost=task['costs']['task'] if t else 1.,
                response_difference=full['costs']['response']-task['costs']['response'] if f and t else None,
                same_physical_trace=f and t and full['path']==task['path']))
    common=[r for r in pairs if r['full_executed'] and r['task_executed']]
    extra_full=sum(not r['full_executed'] and r['task_executed'] for r in pairs)
    common_cost=sum(r['full_cost']-r['task_cost'] for r in common)/len(pairs)
    gap=sum(r['full_cost']-r['task_cost'] for r in pairs)/len(pairs)
    report=dict(requests=len(pairs),common_executed=len(common),common_identical_traces=sum(r['same_physical_trace'] for r in common),
        additional_full_rejections=extra_full,additional_full_rejection_mass=extra_full/len(pairs),
        additional_task_rejections=sum(r['full_executed'] and not r['task_executed'] for r in pairs),
        full_minus_task_cost_all_requests=gap,common_execution_cost_gap_weighted=common_cost,
        common_execution_response_mean_gap=float(np.mean([r['response_difference'] for r in common])) if common else None,
        empirical_stopping_bound=common_cost+extra_full/len(pairs),stopping_bound_holds=gap<=common_cost+extra_full/len(pairs)+1e-12,
        scope='same original source law; empirical paired bookkeeping, not a population certificate; response compared only where both execute',records=pairs)
    if not report['stopping_bound_holds']:raise ValueError('stopping-cost bookkeeping mismatch')
    atomic_json(root/'_reports/common-execution-audit.json',report);return report


def figures(qual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contract=read(qual/'workflow_contract.json');cpu=Path(contract['config']['method_params'].get('cpu_fixture_root',qual))
    fig,axes=plt.subplots(2,1,figsize=(7,5.5),constrained_layout=True)
    for ax,name in zip(axes,('matched_prefix_force','matched_late_loss')):
        metric='force_peak' if name=='matched_prefix_force' else 'normal_force'
        for label,parent,provider in (('CPU target reference',cpu,'cpu'),('GPU model',qual,'gpu')):
            data,meta=load_bundle(parent/(provider+'_fixtures')/name);valid=data['valid'][0].astype(bool)
            t=np.arange(valid.sum())*meta['config']['control_dt']
            ax.plot(t,data[metric][0,valid],label=f'{label}: status {meta["status"]}',lw=1.)
        if metric=='force_peak':ax.axhline(25.,ls='--',c='grey',lw=.8)
        else:
            ax.axhline(2.5,ls='--',c='grey',lw=.8);ax.axhline(7.5,ls='--',c='grey',lw=.8)
        ax.set_title(name.replace('_',' '));ax.set_ylabel('Peak contact force (N)' if metric=='force_peak' else 'TCP axial force (N)');ax.legend(fontsize=8)
    axes[-1].set_xlabel('Control time (s)');folder=qual/'_reports';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'cpu-gpu-contact-discrepancy.png',dpi=180);plt.close(fig)


def program(root):
    task=root/'results/arm/keyed_insertion_recovery';qual=task/'qualification/cross_ebdm_gpu'
    q=qualification(qual);figures(qual);result=dict(qualification=q)
    lines=['# P2 retained evidence','',
           f'Original model-search gate: {q["model_search_qualified"]}. Restricted model-search admission: {q.get("restricted_model_search_admitted",False)}. CPU reference equivalent: {q["cpu_reference_equivalent"]}.','',
           '| Fixture | CPU status | GPU status | Long reference passed |','| --- | --- | --- | --- |']
    gpu=read(qual/'gpu_report.json')
    for row in gpu['records']:lines.append(f'| {row["name"]} | {row["cpu_status"]} | {row["gpu_status"]} | {row["long_reference_passed"]} |')
    lines+=['','Independent CPU model validation and CPU target evaluation are required.','']
    for name in ('cross_ebdm_gpu_coverage','cross_ebdm_gpu_distribution'):
        path=task/'validation'/name
        if not (path/'scientific_report.json').exists():continue
        from scripts.paper.cross_ebdm.summarize_refinement import audit,tables
        audit_result=audit(path);science=read(path/'scientific_report.json');tables(path,science,None)
        bands=independent_task_band(path);search=audit_gpu_search(path) if name.endswith('coverage') else None
        common=common_execution(path)
        result[name]=dict(audit=audit_result,totals=science['totals'],new_probes=science['new_probes'],
            actual_target_episodes=science['actual_target_evaluation_episodes'],task_bands=bands,gpu_search_audit=search,
            common_execution=common)
        result[name]['transport_incremental_task_value']='not_demonstrated' if science['totals']['full']['successes']<=science['totals']['task_only']['successes'] else 'requires_independent_statistical_confirmation'
        result[name]['all_full_requests_successful_with_source_mode']=(science['totals']['full']['successes']==science['totals']['full']['attempted'] and
            science['totals']['full']['mode_matches']==science['totals']['full']['attempted'])
        lines+=['',f'## {name}','','| Method | Success / attempted | Rejected |','| --- | --- | --- |']
        for method,value in science['totals'].items():lines.append(f'| {method} | {value["successes"]}/{value["attempted"]} | {value["unsupported"]} |')
        lines+=['',f'New probes: {science["new_probes"]}; distinct CPU target episodes: {science["actual_target_evaluation_episodes"]}.',
                'Frozen selection is unchanged by this read-only audit. Development results, conditional on reused calibrations.']
    result['scientific_scope']='GPU model search plus CPU confirmation; no automatic transport superiority, population guarantee or real-robot evidence'
    result['benchmark_scope_correction']=dict(
        cpu='native controller/physics wall time; scene construction, reset, prior loading and artifact IO excluded',
        gpu='compiled full fixed-horizon controller/physics kernel; reset, input preparation and artifact IO excluded',
        padding='GPU executes masked fixed-horizon padding; CPU stops at terminal; compare valid steps as well as episodes',
        comparison='CPU reference is serial, not the 12-worker CPU collector',
        correction='original immutable benchmark scope text mentions CPU IO; the recorded CPU timer excludes IO')
    coverage=task/'validation/cross_ebdm_gpu_coverage'
    workers=[]
    for folder,stage in ((coverage,coverage/'_logs/gpu_model_search.stage.json'),
                         (coverage/'_parallel_gpu0',coverage/'_parallel_gpu0/_logs/gpu0_calibrated_models.stage.json')):
        if (folder/'gpu_collection_report.json').exists() and stage.exists():
            collection=read(folder/'gpu_collection_report.json');timing=read(stage)
            kernels=sum(b['seconds'] for b in collection['batches'])
            workers.append(dict(path=str(folder),physical_calls=collection['new_physical_model_calls'],
                worker_stage_seconds=timing['seconds'],compile_and_kernel_seconds=kernels,
                other_stage_seconds=timing['seconds']-kernels,
                scope='other time includes reset/input preparation, IO, dispatch and accounting; not separately profiled'))
    result['production_worker_timing']=workers
    result['formal_scaling_ready']=False
    result['performance_claim']='throughput measured; equivalent target-physics acceleration not established'
    destination=qual/'_reports';atomic_json(destination/'program-report.json',result)
    (destination/'program-report.md').write_text('\n'.join(lines)+'\n')
    return result


def report_export_paths(available):
    generated=['sections/current_gpu_evidence.tex','tables/p2_gpu_qualification.tex','figures/cpu_gpu_contact_discrepancy.png']
    for name,filename in [('cross_ebdm_gpu_coverage','p2_coverage_results.tex'),
                          ('cross_ebdm_gpu_distribution','p2_law_results.tex')]:
        if name in available:generated.append('tables/'+filename)
    return generated


def paper(root,result,destination):
    """Export audited P2 development reports outside the manuscript tree."""
    import shutil
    generated=report_export_paths(result)
    destination=preflight_report_exports(destination,generated+['report-export-provenance.json'])
    for folder in ('tables','sections','figures'):
        (destination/folder).mkdir(parents=True,exist_ok=True)
    qual=root/'results/arm/keyed_insertion_recovery/qualification/cross_ebdm_gpu'
    report=read(qual/'gpu_report.json');tables=destination/'tables'
    lines=['% Generated from retained P2 qualification and independent audits.',
           r'\begin{center}\small',r'\begin{tabular}{lrrc}',r'\toprule',
           r'Fixture & CPU & GPU & Long reference \\',r'\midrule']
    for row in report['records']:
        lines.append(row['name'].replace('_',r'\_')+f' & {row["cpu_status"]} & {row["gpu_status"]} & '+('Pass' if row['long_reference_passed'] else 'Fail')+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
    (tables/'p2_gpu_qualification.tex').write_text('\n'.join(lines)+'\n')
    section=[r'\subsection{P2: GPU Model Execution and CPU Confirmation}',
        'P2 使用固定基 Panda/xArm7、匹配键槽与 TCP 有符号 wrench，保留同一',
        'phase prior、反馈 decoder、控制时钟、力限制与成功判据。接触物理和控制',
        'carry 使用 FP64，冻结模型与命令接口使用 FP32。时间 scan 与环境 vmap',
        '形成设备内闭环；IK reset 与工件读写是明确的主机边界。稳态运行通过',
        '禁止隐式设备传输检查；三个条件的完整与 microbatch 结果逐元素一致。',
        '',r'\paragraph{工程语义与物理等价分别验收。}',
        '九个历史开发案例用于工程诊断，其中两个为源身体；它们不是新统计测试集。',
        '27 个保留检查点上，工具侧符号、TCP 力矩平移及 MJX 自身接触集合上的',
        'witness 定义通过独立检查。原 CPU 接触集合与长时闭环等价验收未通过。',
        r'\input{tables/p2_gpu_qualification.tex}',
        '状态 1/8/9/5 分别表示成功、轴向支撑丢失、力超限和超时。只有两个源',
        '案例满足全部预声明长轨迹容差；有四个案例改变了终止状态。双精度',
        '不能消除接触解算差异及其通过阶段/终止事件累积的影响。',
        r'\begin{figure}[ht]\centering',
        r'\includegraphics[width=.88\linewidth]{figures/cpu_gpu_contact_discrepancy.png}',
        r'\caption{同一冻结 prior/map/reset 的 CPU 与 GPU 接触诊断。上图为峰值',
        '接触力，下图为 TCP 轴向力；虚线为原有约束或保持区间。差异不是方法收益。}',
        r'\end{figure}',
        '因此原验收失败保留；另行给予强制独立 CPU 验证的受限模型搜索资格。',
        'GPU 输出不替代 CPU 目标物理，也不自动继承 CPU 校准 twin 的预测准确率。',
        f'上述固定诊断中，有界任务成本的最大实测 CPU/GPU 差为 {result["qualification"]["max_observed_engine_task_error"]:.5f}；',
        '这不是统一误差上界，也不是随机样本总体估计。',
        '',r'\paragraph{扩大覆盖与冻结确认。}',
        '继续引用三份校准重复乘两个条件的已计费 buffer，新增 probe 为零；',
        '没有增加独立校准数。训练 reset 增至 8，独立 CPU 模型验证增至 16，',
        '目标评估与条件 law 各使用 12 个新 reset clusters，均与前轮开发分离。',
        'GPU 搜索完整方法与强 task-only 的同一有限候选池；选中映射经过 CPU',
        '验证后冻结，拒绝仍计入全部请求，最终目标评估使用原 CPU 环境。']
    for name,title,filename in [('cross_ebdm_gpu_coverage','共同映射','p2_coverage_results.tex'),
                                ('cross_ebdm_gpu_distribution','条件 law','p2_law_results.tex')]:
        if name not in result:continue
        science=read(root/'results/arm/keyed_insertion_recovery/validation'/name/'scientific_report.json')
        tab=[r'\begin{center}\small',r'\begin{tabular}{lrrr}',r'\toprule',
             r'Method & Derated & Matched & Rejected \\',r'\midrule']
        for method,total in science['totals'].items():
            values=[]
            for suite in ('xarm7_derated','xarm7_matched'):
                selected=[r['results'][method]['summary'] for r in science['replicates'] if r['suite']==suite]
                values.append(f'{sum(s["successes"] for s in selected)}/{sum(s["attempted"] for s in selected)}')
            tab.append(method.replace('_',r'\_')+' & '+' & '.join(values)+f' & {total["unsupported"]}'+r'\\')
        tab += [r'\bottomrule',r'\end{tabular}',r'\end{center}']
        (tables/filename).write_text('\n'.join(tab)+'\n')
        full=science['totals']['full'];task=science['totals']['task_only']
        section += ['',f'{title}的完整方法为 ${full["successes"]}/{full["attempted"]}$，',
            f'强 task-only 为 ${task["successes"]}/{task["attempted"]}$；',r'\input{tables/'+filename+'}',
            '这是全部请求的分母，包含验证拒绝；不把仅部署子集或不同版本的最好格拼成主表。']
        common=result[name]['common_execution']
        section += [f'共同部署的 {common["common_executed"]} 个请求中，{common["common_identical_traces"]} 个使用同一物理轨迹。',
            f'完整方法额外拒绝 {common["additional_full_rejections"]} 个请求；全部请求的平均任务成本差为',
            f'$+{common["full_minus_task_cost_all_requests"]:.5f}$，共同部署子集的平均响应成本差为',
            f'${common["common_execution_response_mean_gap"]:.5f}$。两种分母不能混用。']
        if name.endswith('distribution'):
            section += ['重加权的任务成功伴随模式质量改变；固定 mode0 诊断为 $144/144$，',
                '但只保留一个源模式，不能据此确认最佳单模态控制器或 transport 的独立优势。',
                '打乱 coupling 保持相同物理成功数而原始源对应为零；固定额外噪声没有改善成功数。']
    section += ['', '当前结果属于开发证据。训练预算带、独立模型验证、最终任务收益与',
        '模式质量分别报告；相同冻结映射的结构相等不构成独立增益。后端误差、',
        '采样误差、拒绝质量及不同模式权重的成本项已在有限理论中显式区分。']
    (destination/'sections/current_gpu_evidence.tex').write_text('\n'.join(section)+'\n')
    shutil.copyfile(qual/'_reports/cpu-gpu-contact-discrepancy.png',destination/'figures/cpu_gpu_contact_discrepancy.png')
    atomic_json(destination/'report-export-provenance.json',dict(source_program_report_sha256=hashlib.sha256((qual/'_reports/program-report.json').read_bytes()).hexdigest(),
        generated_files={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in generated},
        scope='P2 development-report export outside the manuscript tree'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-root',type=Path,required=True)
    add_report_export_argument(p)
    args=p.parse_args()
    if args.report_export_dir is not None:
        validation=args.development_root/'results/arm/keyed_insertion_recovery/validation'
        available=[name for name in ('cross_ebdm_gpu_coverage','cross_ebdm_gpu_distribution')
                   if (validation/name/'scientific_report.json').exists()]
        preflight_report_exports(args.report_export_dir,report_export_paths(available)+['report-export-provenance.json'])
    result=program(args.development_root)
    if args.report_export_dir:paper(args.development_root,result,args.report_export_dir)
    print(json.dumps({k:(v.get('totals',v.get('passed')) if isinstance(v,dict) else v) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
