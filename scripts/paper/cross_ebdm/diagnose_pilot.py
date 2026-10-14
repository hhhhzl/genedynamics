"""Read-only P1.5 diagnosis of paired target/model traces; no new interaction."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from genedynamics.experiments.artifacts import atomic_json, artifact_write
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle


def trace_metrics(data, meta):
    valid = data['valid'][0].astype(bool)
    c = meta['config']; dt = c['control_dt']; n = round(.3 / dt)
    p = data['tcp_position'][0, valid]; f = data['normal_force'][0, valid]
    cmd = data['command'][0, valid, 12]
    lost = data['contact_lost'][0, valid].astype(bool)
    force_margin = float(2.5 - np.abs(f[-n:] - c['desired_force']).max())
    goal = np.array([.44, -.2, .596 - c['insertion_depth']])
    result = dict(status=meta['status'], success=meta['success'], seconds=meta['seconds'],
                  command_force_final=float(cmd[-1]), actual_force_final=float(f[-1]),
                  force_hold_margin_N=force_margin,
                  position_hold_margin_m=float(c['position_tolerance'] - np.linalg.norm(p[-n:] - goal, axis=-1).max()),
                  lateral_margin_N=float(c['lateral_force_limit'] - data['lateral_force_peak'][0, valid].max()),
                  total_force_margin_N=float(c['force_limit'] - data['force_peak'][0, valid].max()),
                  saturation_max=float(data['saturation'][0, valid].max()),
                  integral_final=float(data['force_integral'][0, valid][-1]),
                  lost_ticks=int(lost.sum()), mode_match=meta['observed_mode'] == meta['mode'])
    starts = np.flatnonzero(lost & ~np.r_[False, lost[:-1]])
    windows = []
    for index in starts[-3:]:
        indices = range(max(0, index - 5), min(len(f), index + 17))
        windows.append([dict(t=round((j + 1) * dt, 6), force=float(f[j]), command=float(cmd[j]),
                             z=float(p[j, 2]), lost=bool(lost[j]),
                             integral=float(data['force_integral'][0, valid][j]),
                             contact=bool(data['allowed_contact'][0, valid][j])) for j in indices])
    result['loss_windows'] = windows
    return result


def diagnose(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('diagnostic output must not mutate the original pilot tree')
    records = []
    for path in sorted(root.glob('level_*/seed_*/evaluation/*/s*_m*/metadata.json')):
        actual, meta = load_bundle(path.parent)
        relative = path.parent.relative_to(root)
        predicted_path = root.joinpath(*[('occupancy_audit' if p == 'evaluation' else p) for p in relative.parts])
        predicted, prediction = load_bundle(predicted_path)
        valid = actual['valid'][0].astype(bool); pv = predicted['valid'][0].astype(bool)
        paired = dict(status_match=meta['status'] == prediction['status'],
                      time_match=meta['seconds'] == prediction['seconds'],
                      final_force_error_N=abs(meta['final_force'] - prediction['final_force']),
                      force_peak_error_N=abs(meta['force_peak'] - prediction['force_peak']),
                      final_position_error_m=float(np.linalg.norm(np.array(meta['final_position']) - prediction['final_position'])),
                      same_valid_shape=bool(np.array_equal(valid, pv)))
        records.append(dict(path=str(relative), tensor_sha256=meta['tensor_sha256'],
                            prediction_sha256=prediction['tensor_sha256'],
                            **trace_metrics(actual, meta), paired=paired))
    if not records:
        raise ValueError('no completed pilot evaluations')
    groups = {}
    for level in sorted({r['path'].split('/')[0] for r in records}):
        rows = [r for r in records if r['path'].startswith(level + '/') and '/calibrated_transport/' in r['path']]
        groups[level] = dict(episodes=len(rows), statuses=dict(Counter(str(r['status']) for r in rows)),
                             saturation_max=max(r['saturation_max'] for r in rows),
                             support_loss_force_commands=[r['command_force_final'] for r in rows if r['status'] == 8],
                             timeout_forces=[dict(command=r['command_force_final'], actual=r['actual_force_final'], integral=r['integral_final'])
                                             for r in rows if r['status'] == 5],
                             success_force_margins=[r['force_hold_margin_N'] for r in rows if r['success']])
    report = dict(complete=True, stage='P1.5', original_root=str(root), episodes=len(records),
                  status_matches=sum(r['paired']['status_match'] for r in records),
                  time_matches=sum(r['paired']['time_match'] for r in records),
                  max_final_force_error_N=max(r['paired']['final_force_error_N'] for r in records),
                  groups=groups, records=records, new_target_interactions=0,
                  interpretation=['The calibrated twin reproduces these controlled failures; larger model training is not the first remedy.',
                                  'Shared controller or physics defects remain possible and require independent checks.',
                                  'The 5 N task target differs from the transported force command.',
                                  'Successful mode labels without task feasibility do not establish useful mode coverage.',
                                  'Tiny paired error on a finite class is not a uniform kernel or statistical certificate.'])
    atomic_json(output / 'diagnostic-report.json', report)
    lines = ['# P1.5 paired trajectory diagnosis', '', f'Original pilot: `{root}`.', '',
             f'{len(records)} existing pairs; {report["status_matches"]} status matches; {report["time_matches"]} time matches. New target interactions: 0.', '',
             '| Condition | Calibrated statuses | Maximum torque ratio |', '| --- | --- | --- |']
    lines += [f'| {name} | {g["statuses"]} | {g["saturation_max"]:.4f} |' for name, g in groups.items()]
    lines += ['', *report['interpretation'], '', 'Full force, position, lateral and event windows are in diagnostic-report.json.']
    payload = ('\n'.join(lines) + '\n').encode()
    with artifact_write(output / 'diagnostic-report.md', len(payload) + 4096):
        (output / 'diagnostic-report.md').write_bytes(payload)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = diagnose(args.root, args.output)
    print(json.dumps({k: report[k] for k in ('episodes', 'status_matches', 'time_matches', 'groups')}, indent=2))


if __name__ == '__main__':
    main()
