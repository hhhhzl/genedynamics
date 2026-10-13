"""Read-only research summary and scientific figure from completed CPU reports."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.cross_embodiment.run_theory_feasibility import publish


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("tmp/cross_embodiment/theory_feasibility"))
    a=p.parse_args();root=a.root
    names=("derated_contact","fresh_paired_audit","behavior_laws","ur5e_calibrated",
           "noisy_contact_occupancy","outside_family_occupancy","learned_episode_response")
    reports={name:json.loads((root/name/"report.json").read_text()) for name in names}
    if not all(report["complete"] for report in reports.values()):raise ValueError("completed reports required")
    primary=reports["derated_contact"];fresh=reports["fresh_paired_audit"];laws=reports["behavior_laws"]
    summary=dict(primary={name:r["summary"] for name,r in primary["evaluations"].items()},
        fresh=fresh["summaries"],paired_certificate=fresh["observed_task_cost_certificate"],
        same_model_controller_equals_transport=primary["same_model_controller_equals_transport"],
        model_cost_certificate=primary["finite_cost_certificate"],
        sampled_laws={name:{k:v for k,v in row.items() if k!="episodes"} for name,row in laws["results"].items()},
        ur5e={name:r["summary"] for name,r in reports["ur5e_calibrated"]["evaluations"].items()},
        noisy_occupancy=reports["noisy_contact_occupancy"]["task_occupancy_metrics"],
        outside_family_occupancy=reports["outside_family_occupancy"]["task_occupancy_metrics"],
        learned_validation=reports["learned_episode_response"]["validation"],
        learned_supported=reports["learned_episode_response"]["validation_supported"],
        scope="conditional structured CPU feasibility; no anchoring/noise/neural/mobile/universal superiority claim")
    publish(root/"summary.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    names=list(summary["primary"]);labels=["Direct","Calibrated","Task-only\nsearch","Nominal\nresponse"]
    counts=[summary["primary"][name]["successes"] for name in names]
    axes[0].bar(np.arange(4),np.asarray(counts)/10,color=["#b8c0cc","#247f79","#68a59e","#dfa25c"])
    axes[0].set_xticks(np.arange(4),labels);axes[0].set_ylim(0,1.18);axes[0].set_ylabel("Task success fraction")
    axes[0].set_title("xArm7: independent 5 resets × 2 modes")
    for i,count in enumerate(counts):axes[0].text(i,count/10+.04,f"{count}/10",ha="center")
    axes[0].text(.5,-.22,"Calibrated and task-only select identical maps",transform=axes[0].transAxes,ha="center",fontsize=9)
    names=["deterministic","extra_noise","shuffled_coupling"]
    success=[laws["results"][n]["successes"]/12 for n in names]
    match=[laws["results"][n]["coupling_matches"]/12 for n in names]
    x=np.arange(3)
    axes[1].bar(x-.18,success,.36,label="Task success",color="#247f79")
    axes[1].bar(x+.18,match,.36,label="Source-mode correspondence",color="#dfa25c")
    axes[1].set_xticks(x,["Deterministic","Extra noise","Shuffled\ncoupling"]);axes[1].set_ylim(0,1.18)
    axes[1].set_title("12 actual source-law samples per method");axes[1].legend(loc="lower left",fontsize=8)
    axes[1].text(.5,-.22,"All three realize the same 6:6 mode marginal",transform=axes[1].transAxes,ha="center",fontsize=9)
    rows=fresh["episodes"];seeds=fresh["contract"]["reset_seeds"]
    values={name:[np.mean([r["task_cost"] for r in table if r["seed"]==seed]) for seed in seeds] for name,table in rows.items()}
    axes[2].scatter(values["direct"],values["calibrated_transport"],s=25,color="#247f79",alpha=.7)
    axes[2].plot([0,1],[0,1],"--",color="#a3adba",lw=1)
    axes[2].set(xlim=(-.02,1.03),ylim=(-.02,1.03),xlabel="Direct task cost",ylabel="Calibrated task cost",title="25 fresh paired reset clusters")
    certificate=fresh["observed_task_cost_certificate"]
    axes[2].text(.05,.88,f"Mean reduction: {certificate['mean_improvements'][0]:.3f}\n95% one-sided lower bound: {certificate['improvement_lower'][0]:.3f}",transform=axes[2].transAxes,fontsize=9)
    fig.suptitle("Matched-keyway CPU feasibility — conditional on one calibration",fontsize=13)
    fig.savefig(root/"theory_feasibility.png",dpi=180)
    fig.savefig(root/"theory_feasibility.svg")
    plt.close(fig)
    print(dict(summary=str(root/"summary.json"),figure=str(root/"theory_feasibility.png")))


if __name__=="__main__":main()
