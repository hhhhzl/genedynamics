"""New paired timestep checks for frozen obstacle, guide and mobile maps."""
import argparse
import hashlib
import json
from pathlib import Path
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.validate_obstacle_mechanism import rollout_job
from scripts.cross_embodiment.validate_mobile_mechanism import episode
from scripts.cross_embodiment.collect_conditional_response import collect_one
from scripts.cross_embodiment.run_theory_feasibility import publish


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("tmp/cross_embodiment/mechanism_closure_v2"));a=p.parse_args()
    definitions=[("obstacle",a.root/"obstacle_mechanism_chain_v2",73000),
                 ("guide",a.root/"spatial_guide_chain_v4",74000),
                 ("mobile",a.root/"mobile_coupled_chain",72000)]
    reports={}
    for role,source,seed in definitions:
        c=json.loads((source/"contract.json").read_text());f=json.loads((source/"frozen.json").read_text())
        cfg=c["target_config"] if role=="guide" else c["config"]
        out=a.root/("mobile_timestep_refinement" if role=="mobile" else role+"_timestep_refinement")
        if role=="obstacle":prior="tmp/cross_embodiment/learned_cpu_v2/source_prior";param=f["stiffness"]["calibrated"]
        elif role=="guide":prior=str(a.root/"spatial_guide_chain_v3/source_prior");param=f["parameters"]["calibrated"]
        else:prior="tmp/cross_embodiment/mechanism_revision/strict_phase_prior";param=f["parameters"]["coupled"]
        expected_prior=f.get("source_sha256",f.get("source_prior_sha256"))
        if expected_prior!=load_bundle(prior)[1]["tensor_sha256"]:raise ValueError("frozen prior checksum mismatch")
        contract=dict(config=cfg,mechanism=c["diagnostic_truth"],source_prior_sha256=load_bundle(prior)[1]["tensor_sha256"],parameters=param,
            seed=seed,modes=[0,1],physics_dt=[.002,.001],wbc_dt=.004 if role=="mobile" else None,
            frozen_sha256=hashlib.sha256((source/"frozen.json").read_bytes()).hexdigest(),
            scope="paired development timestep check; frozen map, no fitting, 4 distinct actual evaluations")
        # Keep the original mobile protocol and its already completed episodes.
        if role=="mobile":
            old=json.loads((out/"contract.json").read_text())
            if any(old[k]!=contract[k] for k in ("config","mechanism","source_prior_sha256","parameters","seed","modes","physics_dt","wbc_dt")):
                raise ValueError("original mobile timestep contract mismatch")
        else:
            if (out/"contract.json").exists() and json.loads((out/"contract.json").read_text())!=contract:raise ValueError("timestep contract mismatch")
            publish(out/"contract.json",contract)
        rows=[]
        for dt in contract["physics_dt"]:
            for mode in (0,1):
                path=str(out/f"evaluation/dt{dt}_m{mode}");config=dict(cfg,physics_dt=dt)
                if role=="obstacle":row=rollout_job((path,config,c["diagnostic_truth"],prior,param,seed,mode))
                elif role=="guide":row=collect_one((path,config,c["diagnostic_truth"],prior,f["methods"]["calibrated"],param,seed,mode,None))
                else:row=episode((path,config,c["diagnostic_truth"],param,seed,mode))
                rows.append(dict(dt=dt,**row));print(dict(environment=role,dt=dt,success=row["success"],mode=mode,observed_mode=row["observed_mode"]),flush=True)
        report=dict(complete=True,passed=all(r.get("joint_success",r["success"]) and r["observed_mode"]==r["mode"] for r in rows),episodes=rows,
            distinct_physical_target_evaluation_episodes=4,contract=contract)
        if role!="mobile":publish(out/"report.json",report)
        reports[role]=report
    publish(a.root/"environment_timesteps.json",dict(complete=True,passed=all(r["passed"] for r in reports.values()),environments=reports,
        scope="12 physical development episodes; qualitative success/mode agreement at paired 2/1 ms steps, not uniform trajectory or contact-event convergence"))


if __name__=="__main__":main()
