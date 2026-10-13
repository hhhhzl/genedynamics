"""Distil source-only Gaussian phase references, verify independent physical paths.

The linear one-hot model is deliberately small. It separates source-law and
contact-feedback qualification from target response/transport research.
"""
import argparse
from pathlib import Path
import numpy as np
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
from genedynamics.learning.cross_embodiment.models import LearnedModel,ModelSpec
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.learning_data import Normalization


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--phase-support",type=int,nargs="+",default=[0,1,2,3]);p.add_argument("--backend",choices=("jax","native"),default="jax")
    a=p.parse_args(); episodes=[]; targets=[[] for _ in range(8)]; fingerprints=[]
    for path in sorted(a.source.glob("s*_m*")):
        data,meta=load_bundle(path)
        mechanism=meta["diagnostic_mechanism"]
        nominal={"servo_gain":1.,"delay_steps":0,"payload_mass":.5,"contact_friction":.4,"contact_time_constant":.02}
        if meta["config"]["robot"]!="panda" or any(k not in nominal or v!=nominal[k] for k,v in mechanism.items()) or not {"servo_gain","delay_steps","payload_mass"}<=mechanism.keys():
            raise ValueError("nominal Panda source data required")
        if not meta["success"] or meta["observed_mode"]!=meta["mode"]: raise ValueError("source success and physical mode evidence required")
        episodes.append(meta); fingerprints.append(meta["tensor_sha256"])
        for phase,reference in zip(data["phase"].ravel(),data["reference"][:,0]): targets[int(phase)*2+meta["mode"]].append(reference)
    if not a.phase_support or not set(a.phase_support)<={0,1,2,3}:raise ValueError("invalid declared phase support")
    if len(episodes)<4 or any(not targets[phase*2+mode] for phase in a.phase_support for mode in (0,1)):
        raise ValueError("two independent seeds and all declared phase/mode bins required")
    if any(targets[phase*2+mode] for phase in set(range(4))-set(a.phase_support) for mode in (0,1)):raise ValueError("undeclared phase present in source data")
    table=np.stack([np.mean(values,axis=0) if values else np.zeros(5) for values in targets]).astype(np.float32)
    model=LearnedModel(ModelSpec(8,5,hidden=(),ensemble=1),backend=a.backend)
    norm=lambda n:Normalization(np.zeros(n,np.float32),np.ones(n,np.float32))
    parameters=model.import_parameters({"layer.0.weight":table[None],"layer.0.bias":np.zeros((1,5),np.float32)})
    prior=FrozenRegressor(model,parameters,norm(8),norm(5),{
        "learned_role":"phase_source_prior","source_role":"nominal Panda event-conditioned reference distillation",
        "reference_std":[.00005,.00005,.00005,.0002,.005],"mode_probabilities":[.5,.5],
        "source_tensor_sha256":fingerprints,"phase_counts":[len(v) for v in targets],
        "phase_support":a.phase_support,
        "scope":"Gaussian reference prior; phase feedback/geometry decoder shared across all baselines; no flow claim"})
    prior.save(a.output,fit_report={"source_episodes":len(episodes),"all_physical_success":True,
        "distinct_initial_paths":len(set(fingerprints)),"reference_fit":"conditional sample mean; closed form maximum likelihood"})
    print({"source_episodes":len(episodes),"output":str(a.output)},flush=True)


if __name__=="__main__":main()
