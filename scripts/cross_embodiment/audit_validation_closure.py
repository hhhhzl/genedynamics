"""Recompute final acceptance from every distinct new physical evaluation trace."""
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.run_theory_feasibility import publish


def inspect(path):
    d,m=load_bundle(path);c=m["config"];v=d["valid"][0].astype(bool);n=round(.3/c["control_dt"])
    if "normal_force" not in d:
        good=(d["position_error"][0,v]<=c["position_tolerance"])&(d["orientation_error"][0,v]<=c["orientation_tolerance"])
        conditions=dict(hold=len(good)>=n and bool(good[-n:].all()),clearance=bool(d["clearance_lower_bound"][0,v].min()>=-c["collision_tolerance"]),
            speed=bool(d["max_joint_speed"][0,v].max()<=c["joint_speed_limit"]))
        p=d["tcp_position"][0,v];crossing=(p[:,0]>=.465)&(p[:,0]<=.535)
        upper=bool((crossing&(p[:,2]>=.64)).any());side=bool((crossing&(p[:,1]<=-.15)).any())
        observed_mode=2 if upper and side else 0 if upper else 1 if side else -1
    else:
        insertion=c["environment"]!="guided_contact_alignment"
        goal=[.44,-.2,.596-c["insertion_depth"]] if insertion else [.46,-.18,.596 if c["guide_profile"]=="spatial_dual_channel" else .586]
        p=d["tcp_position"][0,v];r=d["tcp_rotation"][0,v];f=d["normal_force"][0,v]
        angles=np.arccos(np.clip((np.trace(np.diag([1.,-1.,-1.])@r.swapaxes(-1,-2),axis1=-2,axis2=-1)-1.)/2.,-1.,1.))
        conditions=dict(hold_samples=len(p)>=n,position_hold=bool((np.linalg.norm(p[-n:]-goal,axis=-1)<c["position_tolerance"]).all()),
            orientation_hold=bool((angles[-n:]<c["orientation_tolerance"]).all()),force_hold=bool((np.abs(f[-n:]-c["desired_force"])<2.5).all()),
            total_force=bool(d["force_peak"][0,v].max()<=c["force_limit"]),lateral_force=bool(d["lateral_force_peak"][0,v].max()<=c["lateral_force_limit"]),collision_free=not bool(d["forbidden_collision"][0,v].any()))
        if "support_count" in d:conditions.update(support=bool(d["support_count"][0,v].min()>=2),body_tilt=bool(d["base_tilt"][0,v].max()<=.5),qp_feasible=bool(d["wbc_residual"][0,v].max()<=.01))
        observed_mode=-1
        if len(p)>=2:
            if insertion:
                touch=np.flatnonzero(f>.3)
                if len(touch):
                    first=touch[0];withdraw=np.flatnonzero(p[first:,2]>p[first,2]+.012)
                    if len(withdraw):observed_mode=int(p[:first+withdraw[0]+1,1].min()<p[0,1]-.004)
            else:
                x=np.flatnonzero(p[:,0]-p[0,0]>.015);y=np.flatnonzero(p[:,1]-p[0,1]>.008)
                if len(x) and len(y):observed_mode=0 if x[0]<y[0] else 1
    conditions["deadline"]=m["seconds"]<=c["timeout_seconds"]+1e-9
    accepted=all(conditions.values())
    return dict(path=str(path),reported=bool(m["success"]),accepted=accepted,conditions=conditions,sha256=m["tensor_sha256"],
        observed_mode=observed_mode,reported_observed_mode=m["observed_mode"],mode_classification_matches=observed_mode==m["observed_mode"])


def main():
    root=Path("tmp/cross_embodiment/mechanism_closure_v2");paths=set();completed={};counts={}
    for study in ("finite_response_law","regularized_transport_repeats","spatial_guide_chain_v4","obstacle_mechanism_chain_v2","mobile_coupled_chain","body_mechanism_matrix","body_mechanism_matrix_v2",
                  "obstacle_timestep_refinement","guide_timestep_refinement","mobile_timestep_refinement"):
        report=root/study/"report.json"
        summary=json.loads(report.read_text()) if report.exists() else {}
        completed[study]=summary.get("complete",False)
        expected=summary.get("distinct_physical_target_evaluation_episodes",summary.get("distinct_physical_evaluation_episodes",summary.get("distinct_target_evaluation_episodes")))
        if "cells" in summary:expected=sum(r["distinct_physical_target_evaluation_episodes"] for r in summary["cells"].values())
        found=set();role="physical" if study=="finite_response_law" else "evaluation"
        for meta in (root/study).rglob("metadata.json"):
            if role in meta.relative_to(root/study).parts:found.add(meta.parent)
        paths.update(found);counts[study]=dict(expected=expected,found=len(found),matches=expected==len(found))
    noisy=set()
    for meta in (root/"noisy_response_occupancy").rglob("metadata.json"):
        if "physical" in meta.relative_to(root/"noisy_response_occupancy").parts:noisy.add(meta.parent)
    paths.update(noisy);noise_summary=json.loads((root/"noisy_response_occupancy/report.json").read_text())
    completed["noisy_response_occupancy"]=noise_summary.get("complete",False)
    expected=noise_summary["distinct_target_evaluation_episodes"];counts["noisy_response_occupancy"]=dict(expected=expected,found=len(noisy),matches=expected==len(noisy))
    rows=[inspect(p) for p in sorted(paths)];mismatch=[r for r in rows if r["reported"]!=r["accepted"] or not r["mode_classification_matches"]]
    passed=bool(rows) and not mismatch and all(completed.values()) and all(r["matches"] for r in counts.values())
    publish(root/"acceptance.json",dict(episodes=len(rows),passed=passed,completed_studies=completed,episode_counts=counts,mismatches=mismatch,records=rows,
        scope="distinct actual physical evaluation traces, including retained failed matrix v1; independent terminal geometry/hold/force/body and geometric mode classification; joint limits checked by simulator status, not reconstructed from sparse encoders"))
    print(dict(episodes=len(rows),mismatches=len(mismatch),passed=passed,completed_studies=completed),flush=True)


if __name__=="__main__":main()
