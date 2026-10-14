"""Versioned P1.5--P1.7 studies using task/category/method/level/seed paths.

Only calibrated/public models generate search episodes. Actual targets are
accessed through counted probes, declared diagnostics and frozen evaluations.
All methods, including rejection and ties, remain in the denominators.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

from genedynamics.experiments.artifacts import ArtifactPolicy, POLICY_ENV, atomic_json, canonical_json
from genedynamics.experiments.process import run_stage
from genedynamics.learning.cross_embodiment.contact import ContactConfig, ContactMechanismSpec, mating_contact_config
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from genedynamics.learning.cross_embodiment.finite_transport import (
    refinement_candidates, make_mapping, response_cost, select_finite, conditional_selection,
    supported_weights, selection_stability_interval)
from scripts.cross_embodiment.run_pilot import prepare_environment, asset_manifest, digest


def read(path):
    return json.loads(Path(path).read_text())


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def immutable(path, value):
    value = canonical_json(value)
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'immutable refinement contract mismatch: {path}')
    else:
        atomic_json(path, value)
    return value


@lru_cache(maxsize=2)
def source_prior(path):
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    return PhaseGaussianPrior(FrozenRegressor.load(path, backend='native', device='cpu'))


class ExtraNoise:
    def __init__(self, base, seed):
        self.base = base
        self.random = np.random.default_rng(np.random.SeedSequence([seed, 7021]))

    def apply(self, commands, phase):
        value = np.array(self.base.apply(commands, phase=phase), copy=True)
        sigma = np.array([.0002] * 3 + [.002] * 3 + [0.] * 6 + [.03])
        return value + self.random.normal(size=value.shape) * sigma * ((np.asarray(phase) >= 2)[:, None, None])


def thin(meta, path):
    names = ('status', 'success', 'seed', 'mode', 'observed_mode', 'force_peak', 'lateral_force_peak',
             'final_position', 'final_force', 'orientation_error', 'seconds', 'final_force_rmse',
             'response_signature', 'tensor_sha256', 'wall_seconds')
    return {**{n: meta[n] for n in names}, 'path': str(path)}


def episode(job):
    """Atomic cached collection; a charged incomplete attempt is not replayed."""
    from scripts.cross_embodiment.cpu_mechanism_validation import rollout
    path = Path(job['path']); identity = checksum(job)
    context = dict(request_sha256=identity, role=job['role'], protocol=job.get('protocol','finite_refinement_v1'))
    if (path / 'metadata.json').exists():
        _, meta = load_bundle(path)
        if meta.get('artifact_context') != context or meta['code_fingerprint']['aggregate'] != job['code']:
            raise ValueError(f'cached episode provenance mismatch: {path}')
        return thin(meta, path)
    attempt = path / 'attempt.json'
    if attempt.exists():
        raise RuntimeError(f'incomplete charged {job["role"]} attempt; no automatic free replay: {path}')
    atomic_json(attempt, dict(request_sha256=identity, status='pending', started_unix=time.time(), role=job['role']))
    if job.get('mapping_path'):
        from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
        _, mapping_meta = load_bundle(job['mapping_path'])
        if mapping_meta['tensor_sha256'] != job['mapping_sha256']:
            raise ValueError('frozen mapping tensor changed')
        mapping = ReferenceTransport.load(job['mapping_path'], backend='native', device='cpu')
    else:
        mapping = None if job['parameters'] is None else make_mapping(job['parameters'])
    if job.get('extra_noise'):
        mapping = ExtraNoise(mapping, job['seed'])
    rollout(ContactConfig(**job['config']), ContactMechanismSpec(**job['mechanism']),
            seed=job['seed'], mode=job['mode'], prior=source_prior(job['prior']), transport=mapping,
            output=path, record_events=True, artifact_context=context)
    _, meta = load_bundle(path)
    if meta['code_fingerprint']['aggregate'] != job['code']:
        raise RuntimeError('collection sources changed during an experiment')
    atomic_json(attempt, dict(request_sha256=identity, status='complete', tensor_sha256=meta['tensor_sha256'], role=job['role']))
    return thin(meta, path)


def collect(jobs, workers, label, root):
    unique = {}
    for job in jobs:
        if job['path'] in unique and unique[job['path']] != job:
            raise ValueError('a physical episode path has incompatible requests')
        unique[job['path']] = job
    jobs = list(unique.values())
    immutable(root / '_manifests' / (label + '.json'), jobs)
    contract_path=root/'workflow_contract.json'
    params=read(contract_path)['config']['method_params'] if contract_path.exists() else {}
    if label=='search' and params.get('model_collection_backend')=='jax':
        qualification=read(params['gpu_qualification'])
        if not qualification.get('model_search_qualified'):
            raise ValueError('GPU model search requires completed semantic/batch qualification')
        # Source, selected-map validation and target evaluation stay on the
        # CPU reference. This is an explicit approximate model-search route.
        stage=run_stage('gpu_model_search',[sys.executable,'-m','scripts.cross_embodiment.gpu_finite_collection',
            '--manifest',str(root/'_manifests'/f'{label}.json'),'--root',str(root),
            '--batch-size',str(params.get('gpu_batch_size',64)),'--device',params.get('gpu_device','gpu:0')],
            root/'_logs',timeout=params['stage_timeout_seconds'])
        if stage['status']!='passed':raise RuntimeError('GPU model search failed; incomplete attempts are retained')
        return {job['path']:thin(load_bundle(job['path'])[1],job['path']) for job in jobs}
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(episode, job): job['path'] for job in jobs}
        for future in as_completed(futures):
            row = future.result(); rows[row['path']] = row
            if len(rows) % 24 == 0 or len(rows) == len(jobs):
                progress = dict(stage=label, complete=len(rows), planned=len(jobs))
                atomic_json(root / '_progress.json', progress)
                print(progress, flush=True)
    return rows


def calibration(job):
    from scripts.cross_embodiment.calibrate_contact_twin import collect_probes
    from genedynamics.learning.cross_embodiment.structured_response import fit_contact_twin, validate_twin
    path = Path(job['path']); contract = immutable(path / 'calibration-contract.json', job)
    if job.get('reuse'):
        reference=Path(job['reuse'])
        if digest(reference)!=job['reuse_sha256']:raise ValueError('counted calibration reference changed')
        prior_job=read(reference.parent/'calibration-contract.json')
        if any(prior_job[k]!=job[k] for k in ('config','truth','seed')):
            raise ValueError('calibration condition cannot be reused')
        original=read(reference);ledger=read(reference.parent/'probes/ledger.json')
        if original['budget']!=ledger or any(a['outcome']=='pending' for a in ledger['attempts']):
            raise ValueError('referenced calibration ledger is incomplete')
        result=dict(original,new_probes=0,reused_from=dict(path=str(reference),sha256=digest(reference),
                    ledger_path=str(reference.parent/'probes/ledger.json'),ledger_sha256=digest(reference.parent/'probes/ledger.json')))
        if (path/'calibration.json').exists():
            if read(path/'calibration.json')!=result:raise ValueError('reused calibration provenance mismatch')
        else:atomic_json(path/'calibration.json',result)
        return result
    if (path / 'calibration.json').exists():
        result = read(path / 'calibration.json')
        ledger = read(path / 'probes/ledger.json')
        if ledger != result['budget'] or len(ledger['attempts']) != 5:
            raise ValueError('calibration ledger provenance mismatch')
        return result
    config = ContactConfig(**job['config']); truth = ContactMechanismSpec(**job['truth'])
    records, ledger = collect_probes(config, truth, path / 'probes', seed=job['seed'], tangential=True)
    estimate, fit = fit_contact_twin(config, records[:3])
    validation = validate_twin(config, estimate, records[3:])
    result = dict(estimated=asdict(estimate), fit=fit, validation=validation, budget=ledger,
                  contract_sha256=checksum(contract), code=job['code'])
    atomic_json(path / 'calibration.json', result)
    return result


def splits(seed, params):
    base = 9000000 + seed * 1000 + params.get('reset_seed_offset',0)
    counts = dict(train=params['training_reset_groups'], model_validation=params['model_validation_reset_groups'],
                  evaluation=params['evaluation_reset_groups'], laws=params['law_reset_groups'])
    return {name: list(range(base + offset, base + offset + counts[name]))
            for name, offset in (('train', 0), ('model_validation', 100), ('evaluation', 200), ('laws', 300))}


def cells(contract):
    config = contract['config']; root = Path(config['output_dir'])
    for suite in config['suites']:
        for seed in config['seeds']:
            yield dict(suite=suite['name'], seed=seed,
                       path=root / ('level_' + suite['name']) / f'seed_{seed}', env=suite['env_params'])


def physical_job(contract, path, config, mechanism, seed, mode, parameters, role, extra_noise=False):
    job = dict(path=str(path), prior=contract['prior']['path'], prior_sha256=contract['prior']['sha256'],
                code=contract['code']['aggregate'], config=asdict(config), mechanism=asdict(mechanism),
                seed=int(seed), mode=int(mode), parameters=parameters, role=role, extra_noise=extra_noise)
    if 'metadata' in contract['config']:job['protocol']=contract['config']['metadata'].get('protocol','finite_refinement_v1')
    if role in ('evaluation', 'occupancy_audit') and parameters is not None:
        mapping_path = Path(path).parents[2] / 'frozen_maps' / Path(path).parent.name.removesuffix('_noise')
        _, meta = load_bundle(mapping_path)
        job.update(mapping_path=str(mapping_path), mapping_sha256=meta['tensor_sha256'])
    return job


def freeze_maps(path, bank, indices):
    for index in sorted(set(indices)):
        target = path / 'frozen_maps' / f'c{index}'
        if not (target / 'metadata.json').exists():
            make_mapping(bank[index]).save(target)


def scored(row, source):
    from scripts.cross_embodiment.run_theory_feasibility import costs
    original = costs(row, source)
    original['response_legacy'] = original['response']
    original['response'] = response_cost(row['response_signature'], source)
    original.pop('joint')
    return dict(**row, costs=original)


def summary(rows, attempted):
    from scripts.paper.cross_ebdm.diagnose_pilot import trace_metrics
    present = [r for r in rows if not r.get('unsupported')]
    margins = []
    for row in present:
        data, meta = load_bundle(row['path'])
        margins.append(trace_metrics(data, meta))
    return dict(attempted=attempted, executed=len(present), unsupported=attempted - len(present),
                successes=sum(r['success'] for r in present),
                mode_matches=sum(r['observed_mode'] == r.get('source_mode', r['mode']) for r in present),
                successful_mode_counts=[sum(r['success'] and r['observed_mode'] == m for r in present) for m in (0, 1)],
                observed_mode_counts=[sum(r['observed_mode'] == m for r in present) for m in (0, 1)],
                mean_task_cost=sum(r['costs']['task'] for r in present) / attempted + (attempted - len(present)) / attempted,
                mean_response_cost=sum(r['costs']['response'] for r in present) / len(present) if present else None,
                force_peak_max=max((r['force_peak'] for r in present), default=None),
                lateral_peak_max=max((r['lateral_force_peak'] for r in present), default=None),
                minimum_success_force_margin_N=min((r['force_hold_margin_N'] for r in margins if r['success']), default=None),
                statuses={str(s): sum(r['status'] == s for r in present) for s in sorted({r['status'] for r in present})})


def paired_task_audit(results, baseline='task_only'):
    from genedynamics.learning.cross_embodiment.research_audit import ResearchAudit
    names=[baseline]+[n for n in results if n!=baseline]
    seeds=sorted({r['seed'] for r in results[baseline]['episodes']})
    table=[]
    for seed in seeds:
        table.append([float(np.mean([1. if r.get('unsupported') else r['costs']['task']
                                    for r in results[name]['episodes'] if r['seed']==seed])) for name in names])
    return dict(columns=names,**ResearchAudit().paired_improvement(table),
                protocol_role='conditional finite development audit; no cross-calibration population guarantee')


def feasibility(contract):
    root = Path(contract['config']['output_dir']); params = contract['config']['method_params']
    workers = params['workers']; bank = contract['candidates']; studies = list(cells(contract))
    calibration_jobs = []
    for cell in studies:
        config = mating_contact_config(cell['env']['robot'])
        truth = ContactMechanismSpec(cell['env']['gain'], cell['env']['delay'])
        job=dict(path=str(cell['path']), seed=cell['seed'], config=asdict(config),truth=asdict(truth),code=contract['code']['aggregate'])
        if params.get('calibration_root'):
            reference=Path(params['calibration_root'])/('level_'+cell['suite'])/f'seed_{cell["seed"]}'/'calibration.json'
            job.update(reuse=str(reference),reuse_sha256=digest(reference))
        calibration_jobs.append(job)
    with ProcessPoolExecutor(max_workers=min(3, workers)) as pool:
        estimates = list(pool.map(calibration, calibration_jobs))
    for cell, estimate in zip(studies, estimates):
        cell.update(calibration=estimate, split=splits(cell['seed'], params))
    source_jobs = []
    for seed in contract['config']['seeds']:
        for role in ('train', 'evaluation'):
            for s in splits(seed, params)[role]:
                for m in (0, 1):
                    path = root / '_source' / role / f's{s}_m{m}'
                    source_jobs.append(physical_job(contract, path, mating_contact_config(), ContactMechanismSpec(), s, m, None, 'source_' + role))
    sources = collect(source_jobs, workers, 'source', root)
    if not all(r['success'] and r['observed_mode'] == r['mode'] for r in sources.values()):
        raise RuntimeError('frozen source prior failed new reset support checks')
    search_jobs = []
    for cell in studies:
        if not cell['calibration']['validation']['supported']:
            continue
        config = mating_contact_config(cell['env']['robot'])
        for role, mechanism in (('calibrated', ContactMechanismSpec(**cell['calibration']['estimated'])), ('nominal', ContactMechanismSpec())):
            for index, parameters in enumerate(bank):
                for seed in cell['split']['train']:
                    for mode in (0, 1):
                        path = cell['path'] / (role + '_search') / f'c{index}' / f's{seed}_m{mode}'
                        search_jobs.append(physical_job(contract, path, config, mechanism, seed, mode, parameters, 'training_' + role))
    searches = collect(search_jobs, workers, 'search', root) if search_jobs else {}
    validation_jobs = []
    for cell in studies:
        selections = {}; histories = {}
        config = mating_contact_config(cell['env']['robot'])
        for role, mechanism in (('calibrated', ContactMechanismSpec(**cell['calibration']['estimated'])), ('nominal', ContactMechanismSpec())):
            history = []
            if cell['calibration']['validation']['supported']:
                for index, parameters in enumerate(bank):
                    rows = []
                    for seed in cell['split']['train']:
                        for mode in (0, 1):
                            path = cell['path'] / (role + '_search') / f'c{index}' / f's{seed}_m{mode}'
                            source = sources[str(root / '_source/train' / f's{seed}_m{mode}')]['response_signature']
                            rows.append(scored(searches[str(path)], source))
                    history.append(dict(parameters=parameters, episodes=rows))
                selection = select_finite(history, task_slack=params['task_slack'])
            else:
                selection = dict(status='unsupported_response', full=None, task_only=None)
            histories[role] = history; selections[role] = selection
            atomic_json(cell['path'] / (role + '_search/search.json'), dict(history=history, selection=selection,
                        model_episode_budget=len(bank) * len(cell['split']['train']) * 2,
                        shared_full_and_task_only=True))
            for index in sorted({s for s in (selection['full'], selection['task_only']) if s is not None}):
                for seed in cell['split']['model_validation']:
                    for mode in (0, 1):
                        path = cell['path'] / ('model_validation_' + role) / f'c{index}' / f's{seed}_m{mode}'
                        validation_jobs.append(physical_job(contract, path, config, mechanism, seed, mode, bank[index], 'model_validation_' + role))
        cell.update(histories=histories, selections=selections)
    validation = collect(validation_jobs, workers, 'model_validation', root) if validation_jobs else {}
    evaluation_jobs = []
    for cell in studies:
        config = mating_contact_config(cell['env']['robot']); truth = ContactMechanismSpec(cell['env']['gain'], cell['env']['delay'])
        estimated = ContactMechanismSpec(**cell['calibration']['estimated'])
        methods = dict(direct=dict(index=None, supported=True, role='calibrated'))
        for name, role, objective in (('full', 'calibrated', 'full'), ('task_only', 'calibrated', 'task_only'), ('nominal', 'nominal', 'full')):
            index = cell['selections'][role][objective]
            checked = [] if index is None else [validation[str(cell['path'] / ('model_validation_' + role) / f'c{index}' / f's{s}_m{m}')]
                          for s in cell['split']['model_validation'] for m in (0, 1)]
            supported = bool(checked) and all(r['success'] and r['observed_mode'] == r['mode'] for r in checked)
            methods[name] = dict(index=index, supported=supported, role=role)
        freeze_maps(cell['path'], bank, [m['index'] for m in methods.values() if m['supported'] and m['index'] is not None])
        freeze = dict(methods=methods, candidate_parameters=bank, split=cell['split'], selections=cell['selections'],
                      prior_sha256=contract['prior']['sha256'], code=contract['code']['aggregate'],
                      calibration_sha256=digest(cell['path'] / 'calibration.json'),
                      protocol=contract['protocol'], source_conditioning='P1.6 common map across two source modes')
        immutable(cell['path'] / 'frozen.json', freeze); cell['methods'] = methods
        for method in methods.values():
            if not method['supported']:
                continue
            index = method['index']; label = 'identity' if index is None else f'c{index}'
            for seed in cell['split']['evaluation']:
                for mode in (0, 1):
                    for role, mech in (('evaluation', truth), ('occupancy_audit', estimated)):
                        path = cell['path'] / role / label / f's{seed}_m{mode}'
                        evaluation_jobs.append(physical_job(contract, path, config, mech, seed, mode, None if index is None else bank[index], role))
    evaluations = collect(evaluation_jobs, workers, 'evaluation', root)
    reports = []
    for cell in studies:
        results = {}; paired = {}
        for name, method in cell['methods'].items():
            rows = []; predictions = []; index = method['index']; label = 'identity' if index is None else f'c{index}'
            for seed in cell['split']['evaluation']:
                for mode in (0, 1):
                    if not method['supported']:
                        rows.append(dict(seed=seed, mode=mode, unsupported=True, success=False)); continue
                    source = sources[str(root / '_source/evaluation' / f's{seed}_m{mode}')]['response_signature']
                    rows.append(scored(evaluations[str(cell['path'] / 'evaluation' / label / f's{seed}_m{mode}')], source))
                    predictions.append(scored(evaluations[str(cell['path'] / 'occupancy_audit' / label / f's{seed}_m{mode}')], source))
            results[name] = dict(summary=summary(rows, len(rows)), episodes=rows)
            paired[name] = dict(episodes=len(predictions), status_matches=sum(a['status'] == p['status'] for a, p in zip([r for r in rows if not r.get('unsupported')], predictions)),
                               mean_task_error=float(np.mean([abs(a['costs']['task'] - p['costs']['task']) for a, p in zip([r for r in rows if not r.get('unsupported')], predictions)])) if predictions else None)
        selection = cell['selections']['calibrated']
        stability = None if selection['full'] is None else selection_stability_interval(cell['histories']['calibrated'], selection['full'])
        report = dict(complete=True, suite=cell['suite'], seed=cell['seed'], calibration_supported=cell['calibration']['validation']['supported'],
                      methods=cell['methods'], results=results, occupancy=paired, scalarization_stability=stability,
                      same_selected_map=cell['methods']['full']['index'] == cell['methods']['task_only']['index'],
                      probes_charged=cell['calibration'].get('new_probes',len(cell['calibration']['budget']['attempts'])),
                      reused_counted_probes=5 if cell['calibration'].get('reused_from') else 0,
                      paired_task_audit=paired_task_audit(results),
                      formal_statistical_evidence=False, scope='finite common-map development feasibility; explicit rejections retained')
        atomic_json(cell['path'] / 'report.json', report); reports.append(report)
    return finish(contract, reports)


def distribution(contract):
    root = Path(contract['config']['output_dir']); params = contract['config']['method_params']; workers = params['workers']
    parent = Path(params['feasibility_root']).resolve(); bank = contract['candidates']; studies = list(cells(contract))
    source_jobs = []
    for calibration_seed in contract['config']['seeds']:
        for seed in splits(calibration_seed, params)['laws']:
            for mode in (0, 1):
                path = root / '_source/laws' / f's{seed}_m{mode}'
                source_jobs.append(physical_job(contract, path, mating_contact_config(), ContactMechanismSpec(), seed, mode, None, 'source_laws'))
    sources = collect(source_jobs, workers, 'law_sources', root)
    if not all(r['success'] and r['observed_mode'] == r['mode'] for r in sources.values()):
        raise RuntimeError('new law source support failed')
    validation_jobs = []
    for cell in studies:
        origin = parent / ('level_' + cell['suite']) / f'seed_{cell["seed"]}'
        calibration_data = read(origin / 'calibration.json')
        history = read(origin / 'calibrated_search/search.json')['history']; nominal = read(origin / 'nominal_search/search.json')['history']
        chosen = conditional_selection(history, task_slack=params['task_slack']) if history else dict(full=[None]*2, task_only=[None]*2, supported=[False]*2, per_mode=[])
        public = conditional_selection(nominal, task_slack=params['task_slack']) if nominal else dict(full=[None]*2, task_only=[None]*2, supported=[False]*2, per_mode=[])
        cell.update(origin=origin, calibration=calibration_data, history=history, chosen=chosen, public=public)
        config = mating_contact_config(cell['env']['robot'])
        for role, selection, mechanism in (('calibrated', chosen, ContactMechanismSpec(**calibration_data['estimated'])), ('nominal', public, ContactMechanismSpec())):
            for mode in (0, 1):
                for index in sorted({s for s in (selection['full'][mode], selection['task_only'][mode]) if s is not None}):
                    for seed in splits(cell['seed'], params)['model_validation']:
                        path = cell['path'] / ('model_validation_' + role) / f'c{index}' / f's{seed}_m{mode}'
                        validation_jobs.append(physical_job(contract, path, config, mechanism, seed, mode, bank[index], 'law_model_validation_' + role))
    validation = collect(validation_jobs, workers, 'law_model_validation', root) if validation_jobs else {}
    jobs = []
    for cell in studies:
        config = mating_contact_config(cell['env']['robot']); truth = ContactMechanismSpec(cell['env']['gain'], cell['env']['delay'])
        estimated = ContactMechanismSpec(**cell['calibration']['estimated'])
        methods = {}
        for name, selection, role, objective in (('full', cell['chosen'], 'calibrated', 'full'), ('task_only', cell['chosen'], 'calibrated', 'task_only'), ('nominal', cell['public'], 'nominal', 'full')):
            supported = []
            for mode, index in enumerate(selection[objective]):
                rows = [] if index is None else [validation[str(cell['path'] / ('model_validation_' + role) / f'c{index}' / f's{s}_m{mode}')]
                            for s in splits(cell['seed'], params)['model_validation']]
                supported.append(bool(rows) and all(r['success'] and r['observed_mode'] == mode for r in rows))
            methods[name] = dict(indices=selection[objective], supported=supported, weights=[.5,.5], shuffle=False, extra_noise=False)
        identity = cell['history'][0]['episodes'] if cell['history'] else []
        identity_support = [bool(identity) and all(r['success'] and r['observed_mode'] == mode for r in identity if r['mode'] == mode) for mode in (0, 1)]
        identity_cost = [float(np.mean([r['costs']['task'] for r in identity if r['mode'] == m])) if identity else 1. for m in (0, 1)]
        methods['reweight_only'] = dict(indices=[0,0], supported=identity_support,
                weights=supported_weights(identity_cost, identity_support), shuffle=False, extra_noise=False)
        task_costs = [1. if index is None else float(np.mean([r['costs']['task'] for r in cell['history'][index]['episodes'] if r['mode'] == mode]))
                      for mode,index in enumerate(methods['task_only']['indices'])]
        methods['task_only_reweighted'] = dict(methods['task_only'], weights=supported_weights(task_costs, methods['task_only']['supported']))
        if params.get('include_full_reweighted'):
            full_costs=[1. if index is None else float(np.mean([r['costs']['task'] for r in cell['history'][index]['episodes'] if r['mode']==mode]))
                        for mode,index in enumerate(methods['full']['indices'])]
            methods['full_reweighted']=dict(methods['full'],weights=supported_weights(full_costs,methods['full']['supported']))
        methods['shuffled_coupling'] = dict(methods['full'], shuffle=True)
        methods['extra_noise'] = dict(methods['full'], extra_noise=True)
        methods['no_source_mode'] = dict(methods['full'], weights=[1.,0.])
        # Antithetic uniforms make the source law exactly balanced within a
        # reset cluster. This is a conditional-law fixture, not iid frequency
        # estimation. Every method sees the same reset/uniform/noise streams.
        rng = np.random.default_rng(cell['seed'] + 93842)
        schedule = []
        for seed in splits(cell['seed'], params)['laws']:
            u = float(rng.uniform(0., .5))
            for value in (u, 1.-u):
                schedule.append(dict(seed=seed, uniform=value, source_mode=int(value>=.5)))
        freeze_maps(cell['path'], bank, [i for method in methods.values() for mode,i in enumerate(method['indices']) if method['supported'][mode] and i is not None])
        freeze = dict(methods=methods, candidate_parameters=bank, schedule=schedule, chosen=cell['chosen'], public=cell['public'],
                      calibration_reference=dict(path=str(cell['origin'] / 'calibration.json'), sha256=digest(cell['origin'] / 'calibration.json'), reused_counted_probes=5),
                      training_search_reference=dict(path=str(cell['origin'] / 'calibrated_search/search.json'), sha256=digest(cell['origin'] / 'calibrated_search/search.json')),
                      code=contract['code']['aggregate'], prior_sha256=contract['prior']['sha256'],
                      protocol=contract['protocol'], target_task_training_interactions=0,
                      sampling='balanced antithetic conditional-law fixtures, clustered by reset',
                      scope='per-mode finite policy recombination from the identical model episode pool; no additional search rollouts')
        immutable(cell['path'] / 'frozen.json', freeze); cell.update(methods=methods, schedule=schedule)
        for name, method in methods.items():
            for item in schedule:
                if not sum(method['weights']):
                    continue
                mode = int(item['uniform'] >= method['weights'][0]); mode = 1-mode if method['shuffle'] else mode
                if not method['supported'][mode]:
                    continue
                index = method['indices'][mode]
                for role, mechanism in (('evaluation', truth), ('occupancy_audit', estimated)):
                    label=f'c{index}' + ('_noise' if method['extra_noise'] else '')
                    path=cell['path']/role/label/f's{item["seed"]}_m{mode}'
                    jobs.append(physical_job(contract,path,config,mechanism,item['seed'],mode,bank[index],role,method['extra_noise']))
    collected = collect(jobs, workers, 'law_evaluation', root) if jobs else {}
    reports=[]
    for cell in studies:
        results={}
        for name,method in cell['methods'].items():
            rows=[]
            for item in cell['schedule']:
                if not sum(method['weights']):
                    rows.append(dict(**item,unsupported=True,success=False));continue
                mode=int(item['uniform']>=method['weights'][0]);mode=1-mode if method['shuffle'] else mode
                if not method['supported'][mode]:
                    rows.append(dict(**item,unsupported=True,success=False));continue
                index=method['indices'][mode]; label=f'c{index}'+('_noise' if method['extra_noise'] else '')
                path=cell['path']/'evaluation'/label/f's{item["seed"]}_m{mode}'
                source=sources[str(root/'_source/laws'/f's{item["seed"]}_m{item["source_mode"]}')]['response_signature']
                rows.append(dict(**scored(collected[str(path)],source),source_mode=item['source_mode'],conditioning_mode=mode,uniform=item['uniform']))
            results[name]=dict(summary=summary(rows,len(rows)),episodes=rows,weights=method['weights'])
        full=results['full']['summary'];task=results['task_only']['summary']
        same=cell['methods']['full']['indices']==cell['methods']['task_only']['indices']
        report=dict(complete=True,suite=cell['suite'],seed=cell['seed'],methods=cell['methods'],results=results,
                    same_selected_map=same,probes_charged=0,reused_counted_probes=5,additional_search_episodes=0,
                    task_difference=full['mean_task_cost']-task['mean_task_cost'],
                    response_difference=None if full['mean_response_cost'] is None or task['mean_response_cost'] is None else full['mean_response_cost']-task['mean_response_cost'],
                    formal_statistical_evidence=False,
                    paired_task_audit=paired_task_audit(results),
                    contribution_status='not_demonstrated_same_maps' if same else 'development_tradeoff_requires_independent_confirmation',
                    scope='balanced clustered sampled laws; shuffle/removing mode tests source coupling, not independent novelty of transport')
        atomic_json(cell['path']/'report.json',report);reports.append(report)
    return finish(contract,reports)


def finish(contract, reports):
    from scripts.cross_embodiment.audit_validation_closure import inspect
    root=Path(contract['config']['output_dir']);params=contract['config']['method_params']
    paths=sorted(root.glob('level_*/seed_*/evaluation/*/s*_m*/metadata.json'))
    audit=[inspect(p.parent) for p in paths]
    mismatch=[r for r in audit if r['reported']!=r['accepted'] or not r['mode_classification_matches']]
    atomic_json(root/'acceptance.json',dict(episodes=len(audit),passed=bool(audit) and not mismatch,mismatches=mismatch,records=audit))
    totals={}
    for name in reports[0]['results']:
        entries=[r['results'][name]['summary'] for r in reports]
        totals[name]={k:sum(e[k] for e in entries) for k in ('attempted','executed','unsupported','successes','mode_matches')}
        totals[name]['successful_mode_counts']=[sum(e['successful_mode_counts'][m] for e in entries) for m in (0,1)]
    attempts=list(root.glob('level_*/seed_*/evaluation/*/s*_m*/attempt.json'))
    occupancy=list(root.glob('level_*/seed_*/occupancy_audit/*/s*_m*/metadata.json'))
    report=dict(complete=True,stage=params['stage'],replicates=reports,totals=totals,
                actual_target_evaluation_episodes=len(attempts),occupancy_model_episodes=len(occupancy),
                new_probes=sum(r['probes_charged'] for r in reports),artifact_consistency_passed=bool(audit) and not mismatch,
                all_full_cells_qualified=all(r['results']['full']['summary']['successes']==r['results']['full']['summary']['attempted'] for r in reports),
                transport_incremental_value='not_demonstrated_same_maps' if all(r['same_selected_map'] for r in reports) else 'requires_pareto_and_independent_analysis',
                sampling_and_scope='development; independent calibration replicates, reset clusters and source modes kept distinct; no global or real-robot guarantee')
    atomic_json(root/'scientific_report.json',report)
    if mismatch:
        raise RuntimeError('independent trace acceptance mismatch')
    print({k:report[k] for k in ('stage','new_probes','actual_target_evaluation_episodes','all_full_cells_qualified','transport_incremental_value')},flush=True)
    return report


def refinement(config, *, resume=False):
    prepare_environment();root=config.output_dir;params=config.method_params
    if config.backend!='numpy' or config.device!='cpu':
        raise ValueError('P1.5--P1.7 physics explicitly requires qualified numpy/cpu')
    root.mkdir(parents=True,exist_ok=True)
    for key in ('feasibility_root','calibration_root'):
        if (params.get(key) and not Path(params[key]).is_absolute() and config.metadata.get('development_output_root') and
                Path(params[key]).parts[:2]!=('results','_development')):
            params[key]=str(Path(config.metadata['development_output_root'])/params[key])
    if params['stage']=='1.7':
        origin=Path(params['feasibility_root']).resolve()
        previous=read(origin/'scientific_report.json')
        if not previous.get('complete') or not previous.get('artifact_consistency_passed'):
            raise ValueError('distribution requires a complete, independently audited feasibility pool')
    policy=ArtifactPolicy(str(root),params['max_artifact_bytes'],params['min_free_bytes'])
    policy_path=root/'artifact_policy.json';atomic_json(policy_path,asdict(policy),enforce_policy=False);os.environ[POLICY_ENV]=str(policy_path)
    prior=Path(params['prior']).resolve();_,meta=load_bundle(prior)
    contract=dict(config=config.to_dict(),code=code_fingerprint(),candidates=refinement_candidates(params.get('candidate_family','stage_v1')),
                  prior=dict(path=str(prior),sha256=meta['tensor_sha256']),assets=asset_manifest(),
                  versions={n:importlib.metadata.version(n) for n in ('numpy','mujoco','scipy','jax','jaxlib')},
                  workflow_sha256=digest(Path(__file__)),protocol=config.metadata.get('protocol','finite_refinement_v1'),
                  support_sources={name:digest(Path(name)) for name in ('genedynamics/experiments/artifacts.py',
                    'genedynamics/experiments/process.py','genedynamics/experiments/runner.py','scripts/research/environment.py')},
                  selection='empirical feasibility gate, then response minimization within 0.02 task-cost band; full and task-only share candidates',
                  scope='authorized development experiments; no target task training, GPU physics or superiority presumed')
    if params['stage']=='1.7':
        parent_contract=read(origin/'workflow_contract.json')
        if parent_contract['code']['aggregate']!=contract['code']['aggregate'] or parent_contract['prior']!=contract['prior'] or parent_contract['candidates']!=contract['candidates']:
            raise ValueError('distribution pool code/prior/candidate provenance mismatch')
        if parent_contract['config']['seeds']!=config.seeds or parent_contract['config']['suites']!=config.suites:
            raise ValueError('distribution pool calibration/condition mismatch')
        contract['feasibility_workflow_sha256']=digest(origin/'workflow_contract.json')
    if params.get('calibration_root'):
        calibration_root=Path(params['calibration_root']).resolve();parent=read(calibration_root/'workflow_contract.json')
        stable=['genedynamics/learning/cross_embodiment/contact.py',
                'genedynamics/envs/domains/manipulation/cross_embodiment/backends/contact.py',
                'genedynamics/learning/cross_embodiment/structured_response.py',
                'genedynamics/learning/cross_embodiment/backends/numpy/structured_response.py',
                'scripts/cross_embodiment/calibrate_contact_twin.py']
        compatible=all(parent['code']['files'][p]==contract['code']['files'][p] for p in stable)
        if not compatible and params.get('model_collection_backend')=='jax':
            # P2 changes only the non-NumPy admission guard in the public
            # contact frontend. Check retained old source, not a hash waiver.
            import ast,zipfile
            name=stable[0];overlay=calibration_root/'_environment/source-overlay.zip'
            with zipfile.ZipFile(overlay) as archive:old=archive.read(name)
            import hashlib
            if hashlib.sha256(old).hexdigest()!=parent['code']['files'][name]:raise ValueError('old CPU source snapshot checksum mismatch')
            def cpu_frontend(source):
                tree=ast.parse(source)
                runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContactRunner')
                init=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
                init.body=[n for n in init.body if not (isinstance(n,ast.If) and ast.unparse(n.test).startswith("backend != 'numpy'"))]
                return ast.dump(tree,include_attributes=False)
            compatible=(cpu_frontend(old.decode())==cpu_frontend(Path(name).read_text()) and
                all(parent['code']['files'][p]==contract['code']['files'][p] for p in stable[1:]))
            contract['cpu_calibration_compatibility']=dict(old_overlay=str(overlay),old_overlay_sha256=digest(overlay),
                checked='AST equality except non-NumPy admission guard; all CPU physics/fitter files exactly equal',passed=compatible)
        if not compatible or parent['assets']!=contract['assets'] or parent['versions']!=contract['versions']:
            raise ValueError('calibration physics/assets/environment changed; cannot reuse counted probes')
        contract['counted_calibration_pool']=dict(path=str(calibration_root),workflow_sha256=digest(calibration_root/'workflow_contract.json'))
    if params.get('model_collection_backend')=='jax':
        qualification=Path(params['gpu_qualification'])
        if not read(qualification).get('model_search_qualified'):raise ValueError('GPU model search not qualified')
        qualified_contract=read(qualification.parent/'workflow_contract.json')
        numeric=[n for n in contract['code']['files'] if n.startswith('genedynamics/')]
        if any(qualified_contract['code']['files'].get(n)!=contract['code']['files'][n] for n in numeric):
            raise ValueError('GPU numerical providers changed after qualification')
        contract['gpu_qualification']=dict(path=str(qualification),sha256=digest(qualification),
            source_contract_sha256=digest(qualification.parent/'workflow_contract.json'),
            model_search_only=True,independent_model_validation='numpy/cpu',target_evaluation='numpy/cpu')
        contract['scope']='GPU approximate finite model search; independent CPU validation; frozen CPU target confirmation; no presumed superiority'
    path=root/'workflow_contract.json'
    if path.exists() and not resume:
        raise FileExistsError('refinement exists; use --resume with identical code/config/environment')
    immutable(path,contract)
    report_path=root/'scientific_report.json'
    if resume and report_path.exists() and read(report_path).get('complete'):
        return read(report_path)
    stage=run_stage('P'+str(params['stage']).replace('.','_'),[sys.executable,'-m',__spec__.name,'--worker-contract',str(path)],
                    root/'_logs',timeout=params['stage_timeout_seconds'])
    if stage['status']!='passed':raise RuntimeError('refinement stage failed; inspect bounded stage log')
    return read(report_path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,nargs='?');p.add_argument('--worker-contract',type=Path)
    p.add_argument('--development-root',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();prepare_environment()
    if args.worker_contract:
        contract=read(args.worker_contract)
        if contract['code']['aggregate']!=code_fingerprint()['aggregate']:raise ValueError('worker collection sources changed')
        stage=contract['config']['method_params']['stage']
        if stage=='1.6':feasibility(contract)
        elif stage=='1.7':distribution(contract)
        else:raise ValueError('unsupported refinement phase')
        return
    from genedynamics.experiments.framework.config import ExperimentConfig
    config=ExperimentConfig.from_yaml(args.config)
    if args.development_root:config.use_development_output_root(args.development_root)
    if config.validate():p.error('; '.join(config.validate()))
    if args.dry_run:print(json.dumps(config.to_dict(),indent=2));return
    refinement(config,resume=args.resume)


if __name__=='__main__':
    main()
