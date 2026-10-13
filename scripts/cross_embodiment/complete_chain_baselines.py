"""Add matched nominal-response baseline to an immutable completed chain report."""
import argparse,json
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment import MechanismSpec
from genedynamics.learning.cross_embodiment.contact import ContactConfig,research_contact_config
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.validate_mechanism_chain import tune_transport,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--chain",type=Path,required=True);p.add_argument("--prior",type=Path,required=True)
    a=p.parse_args();report=json.loads((a.chain/"report.json").read_text());output=a.chain/"nominal_response_baseline"
    output.mkdir(parents=True,exist_ok=True)
    if (output/"report.json").exists():raise FileExistsError("baseline already evaluated; inspect saved report")
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior))
    if prior.regressor.metadata["tensor_sha256"]!=report["source_prior_sha256"]:raise ValueError("source prior mismatch")
    config=ContactConfig(**report["config"]);truth=MechanismSpec(**report["diagnostic_truth"])
    source_force=np.mean([rollout(research_contact_config(),MechanismSpec(),seed=130,mode=m,prior=prior)["final_force"] for m in (0,1)])
    tune_transport(config,MechanismSpec(),prior,output,source_force)
    transport=ReferenceTransport.load(output/"transport");rows=[]
    for seed in report["test_seeds"]:
        for mode in (0,1):
            r=rollout(config,truth,seed=seed,mode=mode,prior=prior,transport=transport,output=output/f"s{seed}_m{mode}")
            rows.append(r);print(r,flush=True)
    result={"summary":summarize(rows),"episodes":rows,"calibration":"zero context; same native body model and twin optimization budget",
            "source_prior_sha256":report["source_prior_sha256"]}
    (output/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(result["summary"],flush=True)


if __name__=="__main__":main()
