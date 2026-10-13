"""Independently recompute strict acceptance from valid saved physical traces."""
import argparse,json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle


def audit(root):
    records=[]
    paths=set(root.glob('*/s*_m*/metadata.json')) | set(root.glob('s*_m*/metadata.json')) | set(root.glob('*/sample_*/metadata.json'))
    for metadata in sorted(paths):
        arrays,meta=load_bundle(metadata.parent);config=meta['config'];valid=arrays['valid'][0].astype(bool)
        p=arrays['tcp_position'][0,valid];r=arrays['tcp_rotation'][0,valid];f=arrays['normal_force'][0,valid]
        goal=np.array([.44,-.2,.596-config['insertion_depth']]);desired=np.diag([1.,-1.,-1.])
        angles=np.arccos(np.clip((np.trace(desired@r.swapaxes(-1,-2),axis1=-2,axis2=-1)-1.)/2.,-1.,1.))
        n=round(.3/config['control_dt'])
        conditions={
            'hold_samples':len(p)>=n,
            'position_hold':bool(np.all(np.linalg.norm(p[-n:]-goal,axis=-1)<config['position_tolerance'])),
            'depth_hold':bool(np.all(.596-p[-n:,2]>.014-1e-9)),
            'orientation_hold':bool(np.all(angles[-n:]<config['orientation_tolerance'])),
            'force_hold':bool(np.all(np.abs(f[-n:]-config['desired_force'])<2.5)),
            'total_force_peak':bool(np.max(arrays['force_peak'])<=config['force_limit']),
            'lateral_force_peak':bool(np.max(arrays['lateral_force_peak'])<=config['lateral_force_limit']),
            'collision_free':not bool(arrays['forbidden_collision'].any()),
            'timeout':meta['seconds']<=config['timeout_seconds']+1e-9,
        }
        physical_acceptance=all(conditions.values())
        records.append({'path':str(metadata.parent),'reported_success':meta['success'],
            'independent_acceptance':physical_acceptance,'conditions':conditions,
            'orientation_max_hold_deg':float(np.rad2deg(angles[-n:].max())),
            'minimum_depth_hold_mm':float(1000*(.596-p[-n:,2]).min())})
    mismatches=[r for r in records if r['reported_success']!=r['independent_acceptance']]
    report={'episodes':len(records),'reported_successes':sum(r['reported_success'] for r in records),
        'mismatches':mismatches,'passed':bool(records) and not mismatches,
        'scope':'independent hold/geometry/force/collision/time audit; actuator and joint failures remain simulator terminal checks',
        'records':records}
    (root/'acceptance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='records'},flush=True)
    return report['passed']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',type=Path,nargs='+');a=p.parse_args()
    results=[audit(root) for root in a.roots]
    if not all(results):raise SystemExit(1)


if __name__=='__main__':main()
