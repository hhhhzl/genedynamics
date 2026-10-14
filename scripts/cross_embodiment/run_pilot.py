"""P0/P1 workflow using existing task/category/method + level/seed paths.

Consumes ExperimentConfig and its development-root mapping. It schedules the
existing source, prior, calibration/search and qualification entrypoints, never
silently reuses an old study or a charged-but-missing probe.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from genedynamics.experiments.artifacts import ArtifactPolicy, POLICY_ENV, atomic_json, ResourceExhausted
from genedynamics.experiments.process import run_stage
from scripts.research.environment import ROOT, environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_manifest():
    import subprocess
    base = ROOT / 'third_party/mujoco_menagerie'
    paths = ['franka_emika_panda/panda.xml', 'ufactory_xarm7/xarm7_nohand.xml',
             'universal_robots_ur5e/ur5e.xml']
    # Model construction consumes meshes while deriving conservative proxies.
    files = {str(p.relative_to(base)): digest(p) for directory in {Path(p).parts[0] for p in paths}
             for p in sorted((base / directory).rglob('*')) if p.is_file()}
    if any(not (base / p).exists() for p in paths):
        raise FileNotFoundError('initialize the pinned Menagerie assets before starting the pilot')
    return {'revision': subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip(),
            'patch_sha256': digest(ROOT / 'third_party/patches/mujoco_menagerie-go2-mjx.patch'), 'files': files}


def pilot(config, *, resume=False, skip_qualification=False):
    prepare_environment()
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    out = config.output_dir
    params = config.method_params
    if config.device != 'cpu' or config.backend != 'numpy':
        raise ValueError('P1 structured physics requires numpy/cpu; GPU qualification is a separate stage')
    out.mkdir(parents=True, exist_ok=True)
    policy = ArtifactPolicy(str(out), params['max_artifact_bytes'], params['min_free_bytes'])
    policy_path = out / 'artifact_policy.json'
    atomic_json(policy_path, asdict(policy), enforce_policy=False)
    os.environ[POLICY_ENV] = str(policy_path)
    policy.check(16 * 1024**2)
    code = code_fingerprint()
    extra_paths = [ROOT / 'genedynamics/experiments/artifacts.py', ROOT / 'genedynamics/experiments/process.py',
                   ROOT / 'scripts/research/environment.py']
    contract = {'config': config.to_dict(), 'code': code,
                'workflow_sources': {str(p.relative_to(ROOT)): digest(p) for p in extra_paths},
                'assets': asset_manifest(), 'versions': {n: importlib.metadata.version(n)
                    for n in ('jax','jaxlib','mujoco','mujoco-mjx','torch','numpy','scipy','osqp')},
                'qualification_requested': bool(params.get('qualification', True) and not skip_qualification),
                'scope': 'fresh development pilot; no historical artifacts; no GPU physics or formal superiority claim'}
    contract_path = out / 'workflow_contract.json'
    if contract_path.exists():
        if not resume:
            raise FileExistsError('workflow already exists; use --resume or a fresh --development-root')
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('workflow code/config/assets/environment changed; choose a fresh development root')
    else:
        atomic_json(contract_path, contract)
    report = {'complete': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'contract_sha256': digest(contract_path), 'qualification_requested': contract['qualification_requested'],
              'stages': [], 'studies': []}
    timeout = params.get('stage_timeout_seconds', 7200)

    def stage(name, module, arguments, complete=None):
        checkpoint = out / '_logs' / (name + '.stage.json')
        if resume and checkpoint.exists() and json.loads(checkpoint.read_text()).get('status') == 'passed' and (complete is None or complete.exists()):
            row = json.loads(checkpoint.read_text())
            row = dict(row, resumed=True)
        else:
            row = run_stage(name, [sys.executable, '-m', module, *map(str, arguments)], out / '_logs', timeout=timeout)
        report['stages'].append(row)
        atomic_json(out / 'workflow_report.json', report)
        if row['status'] != 'passed':
            raise RuntimeError(f'{name} failed; inspect {row["log"]}')

    try:
        (out / '_qualification/temporary').mkdir(parents=True, exist_ok=True)
        stage('runtime_regression', 'pytest', ['-q', 'test/unit/test_research_runtime.py',
              'test/unit/test_experiment_config.py', '--junitxml=' + str(out / '_qualification/runtime.xml'),
              '--basetemp='+str(out / '_qualification/temporary/runtime')],
              out / '_qualification/runtime.xml')
        if params.get('qualification', True) and not skip_qualification:
            stage('qualification', 'scripts.cross_embodiment.qualify_cpu',
                  ['--output', out / '_qualification', '--timeout', timeout], out / '_qualification/report.json')
        runtime = Path(os.environ['CROSS_EMBODIMENT_NATIVE_LIBRARY']).parent
        if not (runtime / 'libcross_embodiment.so').exists():
            stage('native', 'scripts.cross_embodiment.build_native', ['--output', runtime / 'libcross_embodiment.so'])
        teacher, prior = out / '_source/teacher', out / '_source/prior'
        stage('source_teacher', 'scripts.cross_embodiment.cpu_mechanism_validation',
              ['--research', '--seeds', *params['source_seeds'], '--output', teacher], teacher / 'report.json')
        rows = json.loads((teacher / 'report.json').read_text())
        if not rows or not all(r['success'] and r['observed_mode'] == r['mode'] for r in rows):
            raise RuntimeError('nominal source physical support failed; target adaptation was not started')
        stage('source_prior', 'scripts.cross_embodiment.train_phase_prior',
              ['--source', teacher, '--output', prior, '--backend', 'native', '--device', 'cpu'], prior / 'metadata.json')
        # A fresh test reset range per suite and calibration replicate; all
        # methods within a study share the same reset and mode schedule.
        for suite_index, suite in enumerate(config.suites):
            for seed in config.seeds:
                target = out / ('level_' + suite['name']) / f'seed_{seed}'
                env = suite['env_params']
                test_start = 100000 + suite_index * 100000 + seed * 10
                train_start = 10000 + suite_index * 100000 + seed * 10
                stage(f'{suite["name"]}_{seed}', 'scripts.cross_embodiment.run_theory_feasibility',
                      ['--prior', prior, '--output', target, '--backend', 'native', '--device', 'cpu',
                       '--robot', env['robot'], '--gain', env['gain'], '--delay', env['delay'],
                       '--calibration-seed', seed, '--test-seed-start', test_start,
                       '--train-seed-start', train_start,
                       '--reset-seeds', params['evaluation_reset_groups']], target / 'report.json')
                study = json.loads((target / 'report.json').read_text())
                report['studies'].append({'suite': suite['name'], 'seed': seed, 'path': str(target),
                                         'status': study.get('status', 'complete')})
        from scripts.cross_embodiment.audit_validation_closure import inspect
        checks = [inspect(p.parent) for p in out.glob('level_*/seed_*/evaluation/*/s*_m*/metadata.json')]
        mismatches = [r for r in checks if r['reported'] != r['accepted'] or not r['mode_classification_matches']]
        atomic_json(out / 'acceptance.json', {'artifact_consistency_passed': bool(checks) and not mismatches,
                    'episodes': len(checks), 'mismatches': mismatches, 'records': checks,
                    'scope': 'trace/report consistency; scientific gates are reported separately'})
        summary = scientific_summary(out, config)
        atomic_json(out / 'scientific_report.json', summary)
        report.update(complete=True, status='complete', artifact_consistency_passed=bool(checks) and not mismatches,
                      scientific_gates=summary['gates'])
        if mismatches:
            raise RuntimeError('independent physical acceptance mismatch')
    except BaseException as error:
        report.update(complete=False, status='resource_exhausted' if isinstance(error, ResourceExhausted) else 'interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(out / 'workflow_report.json', report, enforce_policy=False)
    print(json.dumps({'output': str(out), 'status': report['status'], 'gates': report['scientific_gates']}, indent=2))
    return report


def scientific_summary(out, config):
    rows = []
    for suite in config.suites:
        for seed in config.seeds:
            path = out / ('level_' + suite['name']) / f'seed_{seed}' / 'report.json'
            study = json.loads(path.read_text())
            evaluations = study.get('evaluations', {})
            metrics = {name: {**entry['summary'],
                       'mean_task_cost': sum(e['costs']['task'] for e in entry['episodes']) / len(entry['episodes']),
                       'mean_response_cost': sum(e['costs']['response'] for e in entry['episodes']) / len(entry['episodes'])}
                       for name, entry in evaluations.items()}
            rows.append({'suite': suite['name'], 'seed': seed, 'calibration_supported': study['calibration']['validation']['supported'],
                         'probes_charged': len(study['calibration']['budget']['attempts']),
                         'same_selected_parameters': study.get('same_model_controller_equals_transport'), 'methods': metrics})
    supported = all(r['calibration_supported'] for r in rows)
    identical = all(r['same_selected_parameters'] is True for r in rows if r['methods'])
    return {'complete': True, 'replicates': rows,
            'gates': {'source_support': True, 'calibration_support': supported,
                      'transport_incremental_value': 'not_demonstrated_same_maps' if identical and supported else 'requires_analysis',
                      'formal_statistical_evidence': 'not_run', 'gpu_physics': 'not_run'},
            'scope': 'finite stage-affine CPU pilot; independent new calibrations; equal-budget task-only baseline; ties and failures retained'}


def prepare_environment():
    runtime = Path(os.environ.get('GENEDYNAMICS_RUNTIME', ROOT / '.runtime/cross_ebdm'))
    os.environ.update(environment(runtime))
    os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    os.environ.setdefault('XLA_FLAGS','--xla_disable_hlo_passes=constant_folding')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--development-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-qualification', action='store_true', help='explicit partial scope, recorded in workflow report')
    args = parser.parse_args()
    prepare_environment()
    from genedynamics.experiments.framework.config import ExperimentConfig
    config = ExperimentConfig.from_yaml(args.config)
    if args.development_root:
        config.use_development_output_root(args.development_root)
    if config.metadata.get('workflow') != 'cross_embodiment_pilot':
        parser.error('config must declare workflow: cross_embodiment_pilot')
    if config.validate():
        parser.error('; '.join(config.validate()))
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return
    pilot(config, resume=args.resume, skip_qualification=args.skip_qualification)


if __name__ == '__main__':
    main()
