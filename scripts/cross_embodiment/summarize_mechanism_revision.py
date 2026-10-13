"""Produce a reproducible table/figure from completed, audited frozen studies."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    studies=[]
    for name in ('derated','moderate','stress'):
        folder=a.root/('frozen_'+name);r=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'acceptance_audit.json').read_text())
        if not r['complete'] or not audit['passed']:raise ValueError('completed studies and independent audit required')
        studies.append((name,r))
    methods=list(studies[0][1]['evaluations']);all_rows=[];comparisons=[]
    for name,report in studies:
        for method,data in report['evaluations'].items():
            summary=data['summary'];all_rows.append({'study':name,'method':method,**summary})
        full=report['evaluations']['calibrated_transport']['episodes']
        for method in methods:
            if method=='calibrated_transport':continue
            other=report['evaluations'][method]['episodes'];seeds=report['contract']['test_seeds']
            # Predeclared task/response evaluation cost, no train-only oracle.
            config=report['contract']['config'];goal=np.array([.44,-.2,.596-config['insertion_depth']])
            def cost(row):
                return (np.linalg.norm(np.asarray(row['final_position'])-goal)/config['position_tolerance'])**2+(row['final_force']-5.)**2+(row['orientation_error']/config['orientation_tolerance'])**2+10.*(not row['success'])+2.*(row['mode']!=row['observed_mode'])
            differences=np.array([np.mean([cost(r) for r in other if r['seed']==s])-np.mean([cost(r) for r in full if r['seed']==s]) for s in seeds])
            random=np.random.default_rng(42);boot=random.choice(differences,(10000,len(seeds)),replace=True).mean(axis=1)
            comparisons.append({'study':name,'baseline':method,'mean_baseline_minus_calibrated_cost':float(differences.mean()),
                'reset_cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'paired_success':report['comparisons'][method]})
    (a.root/'frozen_summary.json').write_text(json.dumps({'rows':all_rows,'comparisons':comparisons,
        'scope':'three selected simulation contexts, one calibration each; correct model structure, no sensor noise; exploratory comparisons'},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#8b949e','#168a6f','#d08b21','#a6b5c8','#7856aa'];labels=['Direct','Calibrated','Force only','Inverse gain','Nominal response']
    fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained')
    for i,(name,report) in enumerate(studies):
        context=report['contract']['diagnostic_truth'];title=f"gain {context['servo_gain']:.2f}, delay {context['delay_steps']*20} ms"
        x=np.arange(len(methods));rates=[];low=[];high=[];forces=[];flow=[];fhigh=[]
        for method in methods:
            data=report['evaluations'][method];s=data['summary'];rate=s['both_modes_success_clusters']/s['reset_clusters'];lo,hi=s['both_modes_exact_95ci']
            rates.append(rate);low.append(rate-lo);high.append(hi-rate)
            rows=data['episodes'];groups=np.array([np.mean([r['final_force_rmse']**2 for r in rows if r['seed']==seed]) for seed in report['contract']['test_seeds']])
            val=float(np.sqrt(groups.mean()));random=np.random.default_rng(42);boot=np.sqrt(random.choice(groups,(10000,len(groups)),replace=True).mean(axis=1));lo,hi=np.quantile(boot,[.025,.975])
            forces.append(val);flow.append(max(val-lo,0));fhigh.append(max(hi-val,0))
        axes[i,0].bar(x,rates,color=colors,yerr=[low,high],capsize=3,edgecolor='white');axes[i,0].set_ylim(0,1.08);axes[i,0].set_ylabel('Both task attempts succeed / reset');axes[i,0].set_title(title)
        axes[i,1].bar(x,forces,color=colors,yerr=[flow,fhigh],capsize=3,edgecolor='white');axes[i,1].set_ylabel('Final axial-load RMSE (N)');axes[i,1].set_title('5 N contact-load reference, valid final 0.3 s')
        for ax in axes[i]:
            ax.set_xticks(x,labels,rotation=20,ha='right');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Panda → xArm7: frozen structured-twin mechanism study\n20 independent reset clusters/context; five charged calibration probes',fontsize=13)
    fig.savefig(a.root/'frozen_mechanism_results.png',dpi=180);fig.savefig(a.root/'frozen_mechanism_results.svg')
    for row in all_rows:print(row['study'],row['method'],f"{row['successes']}/{row['attempted']}",f"mode {row['mode_matches']}/{row['attempted']}",f"force RMSE {row['final_force_rmse']:.3f} N",flush=True)


if __name__=='__main__':main()
