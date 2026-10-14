"""Audit completed immutable P1 studies without rerunning any target episode.

Collection and audit provenance are separate: postprocessing may use newer
code, but every physical trace must retain the original collection fingerprint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from genedynamics.experiments.artifacts import POLICY_ENV, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ[POLICY_ENV] = str(root / 'artifact_policy.json')
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    from scripts.cross_embodiment.audit_validation_closure import inspect
    from scripts.cross_embodiment.run_pilot import scientific_summary
    from genedynamics.experiments.framework.config import ExperimentConfig

    contract_path = root / 'workflow_contract.json'
    contract = json.loads(contract_path.read_text())
    config = ExperimentConfig(**contract['config'])
    original_code = contract['code']['aggregate']
    _, prior = load_bundle(root / '_source/prior')
    records, studies = [], []
    for suite in config.suites:
        for seed in config.seeds:
            path = root / ('level_' + suite['name']) / f'seed_{seed}'
            study = json.loads((path / 'report.json').read_text())
            declared = json.loads((path / 'contract.json').read_text())
            frozen = json.loads((path / 'frozen_evaluation.json').read_text())
            assert study['complete'] and study['contract'] == declared
            assert declared['code_fingerprint']['aggregate'] == original_code
            assert declared['source_sha256'] == prior['tensor_sha256']
            assert frozen['contract_sha256'] == hashlib.sha256((path / 'contract.json').read_bytes()).hexdigest()
            assert frozen['methods'] == study['methods']
            ledger = json.loads((path / 'probes/ledger.json').read_text())
            assert ledger == study['calibration']['budget'] and len(ledger['attempts']) == 5
            assert all(a['outcome'] != 'pending' for a in ledger['attempts'])
            count = 0
            for method, evaluation in study['evaluations'].items():
                assert evaluation['summary']['attempted'] == len(declared['test_seeds']) * 2
                for row in evaluation['episodes']:
                    artifact = path / 'evaluation' / method / f"s{row['seed']}_m{row['mode']}"
                    _, metadata = load_bundle(artifact)
                    assert metadata['code_fingerprint']['aggregate'] == original_code
                    assert metadata['success'] == row['success'] and metadata['status'] == row['status']
                    records.append(inspect(artifact))
                    count += 1
            assert count == len(study['methods']) * len(declared['test_seeds']) * 2
            studies.append({'suite': suite['name'], 'seed': seed, 'path': str(path), 'status': 'complete'})
    mismatches = [r for r in records if r['reported'] != r['accepted'] or not r['mode_classification_matches']]
    acceptance = {'artifact_consistency_passed': bool(records) and not mismatches, 'episodes': len(records),
                  'mismatches': mismatches, 'records': records, 'collection_code_fingerprint': original_code,
                  'audit_code_fingerprint': code_fingerprint()['aggregate'],
                  'scope': 'read-only trace/report consistency; no new probes, simulation or optimization'}
    atomic_json(root / 'acceptance.json', acceptance)
    if mismatches:
        raise RuntimeError(f'independent acceptance found {len(mismatches)} mismatches')
    science = scientific_summary(root, config)
    atomic_json(root / 'scientific_report.json', science)
    before = json.loads((root / 'workflow_report.json').read_text())
    history = root / '_logs/workflow-before-final-audit.json'
    if not history.exists():
        atomic_json(history, before)
    report = dict(before, complete=True, status='complete_audited', studies=studies,
                  artifact_consistency_passed=True, scientific_gates=science['gates'],
                  recovery='completed immutable studies audited after JSON tuple/list resume mismatch; no target replay',
                  collection_code_fingerprint=original_code, audit_code_fingerprint=acceptance['audit_code_fingerprint'])
    if 'error' in before:
        report['prior_orchestration_error'] = before['error']
        report.pop('error', None)
        report.pop('error_type', None)
    atomic_json(root / 'workflow_report.json', report)
    print(json.dumps({'complete': True, 'episodes': len(records), 'mismatches': len(mismatches),
                      'target_replayed': False, 'scope': report['recovery']}, indent=2))


if __name__ == '__main__':
    main()
