"""Finite, feasibility-gated transport selection at a host research boundary.

Both objectives consume exactly the same calibrated candidate episodes.
Acceptance on finite training resets is an empirical filter, not a guarantee.
"""
import math


def refinement_candidates(family='stage_v1'):
    """24 fixed candidates; the P1 bank plus intermediate load and impedance."""
    bank = [dict(style='stage', scale=s, stiffness=k, bias=list(b))
            for s in (1., 2.5) for k in (0., 1.)
            for b in ((0., 0.), (-.004, .002), (-.008, .004))]
    bank += [dict(style='stage', scale=s, stiffness=k, bias=[0., 0.])
             for s in (1.25, 1.5, 1.75, 2.) for k in (0., 1.)]
    bank += [dict(style='impedance', scale=s, stiffness=k, bias=[0., 0.])
             for s in (1., 1.5) for k in (-.5, .5)]
    if family=='stage_v1':return bank
    if family!='recovery_v2':raise ValueError('unknown declared finite transport family')
    retained=[bank[i] for i in (0,10,11,16,17,19)]
    recovery_profiles=[[-.5,-.5,0.,0.,0.,0.], [.25,.25,0.,.25,.25,.25], [0.,0.,0.,.5,.5,.5]]
    hybrid=[dict(style='stage',scale=s,stiffness=k,bias=[0.,0.],recovery_stiffness=profile)
            for profile in recovery_profiles for s in (1.5,1.75,2.) for k in (0.,1.)]
    return retained+hybrid


def make_mapping(parameters):
    from .reference_transport import ReferenceTransport
    k = (parameters['stiffness'],) * 2 + (0.,) + (parameters['stiffness'],) * 3
    if parameters.get('style') == 'impedance':
        return ReferenceTransport.from_impedance(parameters['scale'], k, backend='native')
    return ReferenceTransport.from_stage_parameters(parameters['scale'], parameters['bias'],
                alignment_stiffness=k, insertion_stiffness=k,
                recovery_stiffness=parameters.get('recovery_stiffness',(0.,)*6),backend='native')


def response_cost(signature, source):
    """Bounded 24-feature physical response discrepancy, fixed before tests.

Original features: presence, duration/15s, mean xy/20mm, mean F/10N,
peak lateral/limit, for each of four phases. Convert xy to 2mm units
and normal force to 2.5N units; clip each squared discrepancy to [0,1].
"""
    import numpy as np
    a, b = np.asarray(signature, float), np.asarray(source, float)
    if a.shape != (24,) or b.shape != (24,) or not np.isfinite([a, b]).all():
        raise ValueError('finite paired 24-feature responses required')
    scales = np.tile([1., 1., .1, .1, .25, 1.], 4)
    return float(np.minimum(((a - b) / scales) ** 2, 1.).mean())


def select_finite(history, *, modes=(0, 1), task_slack=.02):
    """Task-only optimum and minimum response cost within its task-cost band.

Reject an entirely infeasible class. Full and task-only share the filter,
mode information, capacity, probes, and simulator episodes. No test data here.
"""
    if not history or not modes or task_slack < 0 or not math.isfinite(task_slack):
        raise ValueError('nonempty declared class, modes and nonnegative slack required')
    scores = []
    for index, entry in enumerate(history):
        rows = [r for r in entry['episodes'] if r['mode'] in modes]
        covered = all(any(r['mode'] == m for r in rows) for m in modes)
        feasible = covered and all(r['success'] and r['observed_mode'] == r['mode'] for r in rows)
        scores.append(dict(index=index, feasible=feasible,
                           task=sum(r['costs']['task'] for r in rows) / len(rows) if rows else 1.,
                           response=sum(r['costs']['response'] for r in rows) / len(rows) if rows else 1.))
    feasible = [s for s in scores if s['feasible']]
    if not feasible:
        return dict(status='no_feasible_candidate', task_only=None, full=None,
                    task_slack=task_slack, scores=scores, admissible=[])
    best = min(feasible, key=lambda s: (s['task'], s['index']))
    band = [s for s in feasible if s['task'] <= best['task'] + task_slack + 1e-12]
    full = min(band, key=lambda s: (s['response'], s['task'], s['index']))
    return dict(status='supported_finite_training', task_only=best['index'], full=full['index'],
                task_slack=task_slack, scores=scores, admissible=[s['index'] for s in band],
                empirical_task_difference=full['task'] - best['task'],
                empirical_response_difference=full['response'] - best['response'])


def conditional_selection(history, *, task_slack=.02):
    per_mode = [select_finite(history, modes=(m,), task_slack=task_slack) for m in (0, 1)]
    return dict(per_mode=per_mode, full=[s['full'] for s in per_mode],
                task_only=[s['task_only'] for s in per_mode],
                supported=[s['status'] == 'supported_finite_training' for s in per_mode])


def supported_weights(costs, supported, *, source_weights=(.5, .5), temperature=.2):
    if len(costs) != 2 or len(supported) != 2 or len(source_weights) != 2 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('two-mode finite law required')
    if any(not math.isfinite(c) for c in costs) or any(w < 0 for w in source_weights) or not math.isclose(sum(source_weights), 1.):
        raise ValueError('finite costs and normalized source law required')
    eligible = [m for m in (0, 1) if supported[m] and source_weights[m] > 0]
    if not eligible:
        return [0., 0.]
    baseline = min(costs[m] for m in eligible)
    weights = [source_weights[m] * math.exp(-(costs[m] - baseline) / temperature) if m in eligible else 0. for m in (0, 1)]
    return [w / sum(weights) for w in weights]


def selection_stability_interval(history, index):
    """Exact lambda interval where a finite candidate minimizes C+lambda R.

Includes only empirically feasible policies. Empty intervals are explicit;
this algebra diagnoses ties and does not claim population optimality.
"""
    feasible = select_finite(history)['scores']
    chosen = feasible[index]
    if not chosen['feasible']:
        return dict(nonempty=False, reason='candidate_infeasible')
    low, high = 0., math.inf
    for other in feasible:
        if not other['feasible']:
            continue
        d_c = chosen['task'] - other['task']; d_r = chosen['response'] - other['response']
        if abs(d_r) < 1e-15:
            if d_c > 1e-12:
                return dict(nonempty=False, reason='dominated_at_equal_response')
        elif d_r > 0:
            high = min(high, -d_c / d_r)
        else:
            low = max(low, -d_c / d_r)
    return dict(nonempty=low <= high + 1e-12, lambda_min=max(0., low),
                lambda_max=None if math.isinf(high) else high,
                scope='exact finite empirical feasible-class interval, not a statistical certificate')
