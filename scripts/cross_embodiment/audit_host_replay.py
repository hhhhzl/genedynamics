"""Independent Linux-to-macOS replay before resuming frozen CPU evaluation."""
from dataclasses import asdict
from pathlib import Path
import platform
import numpy as np
import scipy
import mujoco
from genedynamics.learning.cross_embodiment.contact import mating_contact_config,ContactMechanismSpec
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.collect_conditional_response import collect_one,candidate_bank
from scripts.cross_embodiment.run_theory_feasibility import publish


def main():
    root=Path("tmp/cross_embodiment/mechanism_closure_v2")
    comparisons=[]
    for mode in (0,1):
        old=root/f"regularized_transport_repeats/r0/source/s42000_m{mode}"
        new=root/f"host_replay/source/s42000_m{mode}"
        collect_one((str(new),asdict(mating_contact_config()),asdict(ContactMechanismSpec()),
            "tmp/cross_embodiment/mechanism_revision/strict_phase_prior",0,candidate_bank()[0],42000,mode,None))
        a,ma=load_bundle(old);b,mb=load_bundle(new)
        same_shapes=all(a[k].shape==b[k].shape for k in a)
        errors={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a} if same_shapes else {}
        passed=(same_shapes and ma["status"]==mb["status"] and ma["observed_mode"]==mb["observed_mode"] and
            errors["tcp_position"]<1e-5 and errors["tcp_rotation"]<1e-4 and errors["normal_force"]<.01)
        comparisons.append(dict(mode=mode,passed=passed,same_shapes=same_shapes,max_errors=errors,
            old_sha256=ma["tensor_sha256"],new_sha256=mb["tensor_sha256"]))
    report=dict(passed=all(c["passed"] for c in comparisons),comparisons=comparisons,
        runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__),
        scope="two complete nominal Panda source traces only; does not certify all cross-platform contact bifurcations")
    publish(root/"host_replay/report.json",report);print(report,flush=True)


if __name__=="__main__":main()
