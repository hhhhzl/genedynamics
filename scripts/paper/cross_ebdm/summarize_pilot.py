"""Turn a completed P1 workflow into a traceable report, including no-go gates."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from genedynamics.experiments.artifacts import POLICY_ENV, artifact_write, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ[POLICY_ENV] = str(root / 'artifact_policy.json')
    workflow = json.loads((root / 'workflow_report.json').read_text())
    if not workflow.get('complete'):
        raise ValueError('workflow must finish before reporting the pilot')
    science = json.loads((root / 'scientific_report.json').read_text())
    acceptance = json.loads((root / 'acceptance.json').read_text())
    qualification = json.loads((root / '_qualification/report.json').read_text())
    rows = science['replicates']
    grouped = defaultdict(list)
    feasible = []
    for row in rows:
        calibrated = row['methods'].get('calibrated_transport')
        feasible.append(bool(calibrated and calibrated['successes'] >= .95 * calibrated['attempted']
                             and calibrated['mode_matches'] == calibrated['attempted']))
        for method, metrics in row['methods'].items():
            grouped[row['suite'], method].append(metrics)
    summary = []
    for (suite, method), metrics in sorted(grouped.items()):
        summary.append({'suite': suite, 'method': method, 'independent_calibrations': len(metrics),
                        'successes': sum(m['successes'] for m in metrics),
                        'attempted': sum(m['attempted'] for m in metrics),
                        'mode_matches': sum(m['mode_matches'] for m in metrics),
                        'force_peak_max': max(m['force_peak_max'] for m in metrics),
                        'mean_seconds': sum(m['mean_seconds'] for m in metrics) / len(metrics),
                        'mean_task_cost': sum(m['mean_task_cost'] for m in metrics) / len(metrics),
                        'mean_response_cost': sum(m['mean_response_cost'] for m in metrics) / len(metrics)})
    decision = {'complete': True, 'scope': science['scope'],
                'cpu_regression': qualification['test_counts'],
                'artifact_consistency_passed': acceptance['artifact_consistency_passed'],
                'distinct_physical_evaluation_episodes': acceptance['episodes'],
                'target_probes_charged': sum(r['probes_charged'] for r in rows),
                'all_calibrations_supported': all(r['calibration_supported'] for r in rows),
                'all_calibrated_task_cells_qualified': all(feasible),
                'same_transport_and_task_only_maps': all(r['same_selected_parameters'] is True for r in rows),
                'per_replicate': rows, 'summary': summary,
                'postprocessor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ('workflow_report.json','scientific_report.json','acceptance.json')},
                'next_gate': 'verify source-conditioned transport value before formal scale; task feasibility and response calibration are separate gates'}
    out = root / '_reports'
    atomic_json(out / 'pilot-summary.json', decision)
    text = ['# P0/P1 pilot', '',
            f"CPU regression: {qualification['test_counts']}. Physical evaluation traces: {acceptance['episodes']}. "
            f"New charged probes: {decision['target_probes_charged']}.", '',
            f"Artifact consistency: {decision['artifact_consistency_passed']}. "
            f"Calibration response support: {decision['all_calibrations_supported']}. "
            f"All calibrated task cells qualified: {decision['all_calibrated_task_cells_qualified']}. "
            f"Transport/task-only selected identical maps: {decision['same_transport_and_task_only_maps']}.", '',
            '| Condition | Method | Success | Mode matches | Peak force (N) | Mean time (s) |',
            '| --- | --- | --- | --- | --- | --- |']
    for row in summary:
        text.append(f"| {row['suite']} | {row['method']} | {row['successes']}/{row['attempted']} | "
                    f"{row['mode_matches']}/{row['attempted']} | {row['force_peak_max']:.3f} | {row['mean_seconds']:.3f} |")
    text.extend(['', 'Each condition has three independent new calibrations and optimizations. '
                 'Methods share reset/mode schedules and a finite candidate pool. '
                 'Small development results are not formal statistical evidence. '
                 'A response-calibration pass does not establish target task feasibility. '
                 'Identical maps do not demonstrate an incremental transport benefit.', ''])
    payload = '\n'.join(text).encode()
    with artifact_write(out / 'pilot-summary.md', len(payload) + 4096):
        (out / 'pilot-summary.md').write_bytes(payload)
    print(json.dumps({key: decision[key] for key in ('distinct_physical_evaluation_episodes',
          'target_probes_charged','all_calibrations_supported','all_calibrated_task_cells_qualified',
          'same_transport_and_task_only_maps')}, indent=2))


if __name__ == '__main__':
    main()
