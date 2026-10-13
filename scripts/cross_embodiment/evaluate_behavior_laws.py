"""Actual sampled behavior laws, coupling shuffle, and extra-noise ablation.

Restricted support means the predeclared zero-stiffness correction class, not
global robot infeasibility. Feasibility is inferred from training-twin traces;
evaluation failures remain failures, never used to alter the supported law.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.research_audit import ResearchAudit
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.run_theory_feasibility import publish,transport


class ExtraNoise:
    def __init__(self,base,seed):
        self.base=base
        self.random=np.random.default_rng(np.random.SeedSequence([seed,7021]))

    def apply(self,commands,phase):
        value=np.array(self.base.apply(commands,phase),copy=True)
        noise=self.random.normal(size=value.shape)*np.array([.0002]*3+[.002]*3+[0.]*6+[.03])
        return value+noise*((np.asarray(phase)>=2)[:,None,None])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior",type=Path,default=Path("tmp/cross_embodiment/mechanism_revision/strict_phase_prior"))
    p.add_argument("--samples",type=int,default=12);p.add_argument("--seed",type=int,default=32000)
    a=p.parse_args()
    if a.samples<1:raise ValueError("positive sample count required")
    study=json.loads((a.study/"contract.json").read_text())
    calibration=json.loads((a.study/"calibration.json").read_text())
    search=json.loads((a.study/"calibrated_search/search.json").read_text())
    history=search["history"]
    best=search["winners"]["joint"]
    limited=min((i for i,h in enumerate(history) if h["parameters"]["stiffness"]==0.),key=lambda i:history[i]["means"]["joint"])
    laws={}
    for name,index in (("both_feasible",best),("restricted_stiffness",limited)):
        rows=history[index]["episodes"]
        model_costs=[float(np.mean([r["costs"]["joint"] for r in rows if r["mode"]==m])) for m in (0,1)]
        support=[all(r["success"] for r in rows if r["mode"]==m) for m in (0,1)]
        laws[name]=dict(ResearchAudit().mode_law(model_costs,[.5,.5],support,temperature=.2),model_costs=model_costs,support=support,candidate=index)
    random=np.random.default_rng(a.seed)
    uniforms=random.uniform(size=a.samples)
    seeds=random.integers(100000,200000,size=a.samples).tolist()
    methods={"deterministic":(best,[.5,.5],False,False),"extra_noise":(best,[.5,.5],True,False),
        "shuffled_coupling":(best,[.5,.5],False,True),
        "model_cost_reweighted":(best,laws["both_feasible"]["probabilities"],False,False),
        "restricted_unfiltered":(limited,[.5,.5],False,False),
        "restricted_supported":(limited,laws["restricted_stiffness"]["probabilities"],False,False)}
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior))
    contract=dict(source_sha256=prior.regressor.metadata["tensor_sha256"],sample_seeds=seeds,mode_uniforms=uniforms.tolist(),methods=methods,
        laws=laws,code_fingerprint=code_fingerprint(),
        scope="development sampled laws with shared uniforms/source noise; support is model-derived within a restricted correction class; no global infeasibility claim")
    path=a.output/"contract.json"
    # JSON normalizes tuples to lists.
    contract=json.loads(json.dumps(contract))
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError("behavior law resume contract mismatch")
    publish(path,contract)
    config=ContactConfig(**study["config"]);truth=ContactMechanismSpec(**study["diagnostic_truth"])
    results={}
    for name,(index,weights,extra,shuffle) in methods.items():
        rows=[]
        if sum(weights)==0:
            results[name]=dict(attempted=a.samples,successes=0,unsupported=a.samples,scope="all support rejected; no robot rollout")
            continue
        for i,(seed,u) in enumerate(zip(seeds,uniforms)):
            source_mode=int(u>=weights[0]);deployed_mode=1-source_mode if shuffle else source_mode
            path=a.output/name/f"sample_{i}"
            if (path/"metadata.json").exists():row=load_bundle(path)[1]
            else:
                mapping=transport(history[index]["parameters"])
                if extra:mapping=ExtraNoise(mapping,seed)
                row=rollout(config,truth,seed=seed,mode=deployed_mode,prior=prior,transport=mapping,output=path)
            rows.append(dict(**row,source_mode=source_mode,coupling_match=bool(row["observed_mode"]==source_mode)))
        requested=np.bincount([r["source_mode"] for r in rows],minlength=2)
        realized=[sum(r["observed_mode"]==m for r in rows) for m in (-1,0,1,2)]
        successful=[sum(r["success"] and r["observed_mode"]==m for r in rows) for m in (0,1)]
        results[name]=dict(attempted=len(rows),successes=sum(r["success"] for r in rows),source_counts=requested.tolist(),
            realized_counts=dict(zip(map(str,(-1,0,1,2)),realized)),successful_mode_counts=successful,
            coupling_matches=sum(r["coupling_match"] for r in rows),episodes=rows)
        publish(a.output/"report.json",dict(complete=False,contract=contract,results=results))
        print(dict(method=name,**{k:v for k,v in results[name].items() if k!="episodes"}),flush=True)
    publish(a.output/"report.json",dict(complete=True,contract=contract,results=results,
        scope="actual trajectory-classified sampled laws; small development sample, not a mode probability confidence certificate"))


if __name__=="__main__":main()
