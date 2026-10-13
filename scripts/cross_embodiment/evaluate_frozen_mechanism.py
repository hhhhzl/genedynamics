"""Fresh reset-cluster evaluation of immutable source/transport artifacts.

No calibration or optimization occurs here. Both recovery modes share a reset
seed, so uncertainty uses independent reset clusters rather than pretending
that the two modes are independent experiments.
"""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from genedynamics.learning.cross_embodiment import MechanismSpec
from genedynamics.learning.cross_embodiment.contact import ContactConfig
from genedynamics.learning.cross_embodiment.reference_transport import ReferenceTransport
from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
from genedynamics.learning.cross_embodiment.data import code_fingerprint
from scripts.cross_embodiment.cpu_mechanism_validation import rollout
from scripts.cross_embodiment.validate_mechanism_chain import summarize


def cluster_summary(rows):
    result=summarize(rows);seeds=sorted(set(r['seed'] for r in rows))
    groups=np.array([np.mean([r['success'] for r in rows if r['seed']==s]) for s in seeds])
    random=np.random.default_rng(42)
    means=random.choice(groups,(10000,len(groups)),replace=True).mean(axis=1)
    result.update(reset_clusters=len(seeds),success_rate=float(groups.mean()),
        reset_bootstrap_95ci=np.quantile(means,[.025,.975]).tolist(),
        orientation_max_deg=float(np.rad2deg(max(r['orientation_error'] for r in rows))),
        lateral_force_peak_max=max(r['lateral_force_peak'] for r in rows),
        strict_mode_successes=sum(r['success'] and r['mode']==r['observed_mode'] for r in rows))
    # Bootstrap degenerates at all-success/all-failure. Give exact binomial CI
    # for the conservative cluster event "both modes succeed", as well.
    from scipy.stats import beta
    k=sum(all(r['success'] for r in rows if r['seed']==s) for s in seeds);n=len(seeds)
    result['both_modes_success_clusters']=k
    result['both_modes_exact_95ci']=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chain',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed-start',type=int,required=True);p.add_argument('--reset-seeds',type=int,default=20)
    p.add_argument('--nominal-chain',type=Path,help='reuse the identical public body/prior/config nominal checkpoint')
    a=p.parse_args();main_report=json.loads((a.chain/'report.json').read_text());nominal_chain=a.nominal_chain or a.chain
    nominal_report=json.loads((nominal_chain/'report.json').read_text())
    if nominal_report['config']!=main_report['config'] or nominal_report['source_prior_sha256']!=main_report['source_prior_sha256']:
        raise ValueError('reused nominal checkpoint config/prior mismatch')
    nominal=nominal_chain/'nominal_response_baseline/transport'
    if not (nominal/'metadata.json').exists():raise ValueError('complete matched nominal-response baseline first')
    prior=PhaseGaussianPrior(FrozenRegressor.load(a.prior));full=ReferenceTransport.load(a.chain/'transport');zero=ReferenceTransport.load(nominal)
    if prior.regressor.metadata['tensor_sha256']!=main_report['source_prior_sha256']:raise ValueError('source mismatch')
    config=ContactConfig(**main_report['config']);truth=MechanismSpec(**main_report['diagnostic_truth'])
    context=MechanismSpec(**main_report['estimated']);seeds=list(range(a.seed_start,a.seed_start+a.reset_seeds))
    if set(seeds)&set(main_report['test_seeds']):raise ValueError('fresh reset seeds required')
    methods={'direct':None,'calibrated_transport':full,'force_only_transport':ReferenceTransport.from_parameters(main_report['transport_scale']),
        'system_id_inverse':ReferenceTransport.from_parameters(float(np.clip(1./context.servo_gain,.5,2.5))), 'nominal_response_transport':zero}
    contract={'source_sha256':prior.regressor.metadata['tensor_sha256'],'full_sha256':full.regressor.metadata['tensor_sha256'],
        'nominal_sha256':zero.regressor.metadata['tensor_sha256'],'config':asdict(config),'diagnostic_truth':asdict(truth),
        'test_seeds':seeds,'methods':list(methods),'adaptation':'immutable five-probe checkpoint; no new probes or fit',
        'code_fingerprint':code_fingerprint(),'statistics':'reset-cluster bootstrap and exact both-modes cluster CI'}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'frozen_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    evaluations={}
    for name,transport in methods.items():
        rows=[]
        for seed in seeds:
            for mode in (0,1):
                row=rollout(config,truth,seed=seed,mode=mode,prior=prior,transport=transport,output=a.output/name/f's{seed}_m{mode}')
                rows.append(row)
            print({'method':name,'reset_seed':seed,'successes':sum(r['success'] for r in rows[-2:])},flush=True)
        evaluations[name]={'summary':cluster_summary(rows),'episodes':rows}
        (a.output/'report.json').write_text(json.dumps({'contract':contract,'evaluations':evaluations,'complete':False},indent=2)+'\n')
    comparisons={}
    from scipy.stats import binomtest
    def successes(name,seed):return sum(r['success'] for r in evaluations[name]['episodes'] if r['seed']==seed)
    for name in methods:
        if name=='calibrated_transport':continue
        wins=sum(successes('calibrated_transport',s)>successes(name,s) for s in seeds)
        losses=sum(successes('calibrated_transport',s)<successes(name,s) for s in seeds)
        comparisons[name]={'reset_cluster_wins':wins,'losses':losses,'ties':len(seeds)-wins-losses,
            'paired_two_sided_sign_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.}
    final={'contract':contract,'evaluations':evaluations,'comparisons':comparisons,'complete':True,
        'scope':'conditional on one calibration per mechanism; two modes per reset are dependent; selected simulation mechanism family, not universal or real-robot evidence'}
    (a.output/'report.json').write_text(json.dumps(final,indent=2)+'\n')
    print({k:v['summary'] for k,v in evaluations.items()},flush=True)


if __name__=='__main__':main()
