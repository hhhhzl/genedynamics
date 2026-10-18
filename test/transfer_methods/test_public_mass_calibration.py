"""Public fitting must recover useful dynamics without target carry or labels."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.runtime import MechanismSpec
from genedynamics.solvers.single.cross_ebdm.learning.physics_calibration import PROBE_VERSION, PublicProbe, PublicPhysicsTwin, probe_loss
from genedynamics.solvers.single.cross_ebdm.learning.mass_calibration import PublicMassFamily, fit_public_mass_twin


def record(kind, index):
    task = instantiate_task("T3", robot="xarm7", mechanism_spec=MechanismSpec(.72, 2, .9))
    initial = "clear" if kind == "free" else "jammed"
    seed = 26900 + index
    rows = [task.reset((seed,), seed=seed, initial=initial)]; actions = []
    for step in range(20):
        action = np.zeros((1, 13), np.float32)
        if kind == "free":
            action[0, :2] = [.025 if step < 10 else -.025, .012]
            action[0, 5] = .12
        else:
            action[0, 12] = 3.
        result = task.step(action); rows.append(result.observation); actions.append(action[0])
    return task.spec, PublicProbe(dict(version=PROBE_VERSION, index=index, purpose="fit", seed=seed,
        sample_id=seed, initial=initial, kind=kind), {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, actions)


def test_bounded_public_joint_fit_improves_responses_on_noncontract_fixture():
    spec, free = record("free", 0); _, loaded = record("loaded", 1)
    family = PublicMassFamily(delays=(0, 2), gain_grid_points=3, mass_grid_points=3,
        free_refine_steps=8, joint_refine_delays=2, joint_max_evaluations=90)
    fitted, receipt = fit_public_mass_twin(spec, [free, loaded], family=family)
    nominal = PublicPhysicsTwin(spec, MechanismSpec())
    error = np.mean([probe_loss(p.observations, nominal.predict(p)[0], free=p.metadata["kind"] == "free") for p in (free, loaded)])
    assert fitted.delay_steps == 2
    assert family.gain_min <= fitted.servo_gain <= family.gain_max
    assert family.payload_min <= fitted.payload_mass <= family.payload_max
    assert receipt["fit_loss"] < .1 * error
    assert receipt["validation_used_for_selection"] is False
    assert receipt["fit_probe_sha256"] == [free.digest, loaded.digest]
    resumed, resumed_receipt = fit_public_mass_twin(spec, [free, loaded], family=family, cached_fit=receipt)
    assert resumed == fitted
    assert resumed_receipt["new_hypothesis_calls"] == 0
    assert resumed_receipt["cached_hypothesis_calls"] == receipt["unique_hypothesis_calls"]


def test_validation_and_missing_excitation_cannot_select_parameters():
    spec, free = record("free", 2)
    with pytest.raises(ValueError, match="fit probes"):
        fit_public_mass_twin(spec, [replace(free, metadata=dict(free.metadata, purpose="validation"))])
    with pytest.raises(ValueError, match="loaded"):
        fit_public_mass_twin(spec, [free])
    with pytest.raises(ValueError):
        PublicMassFamily(payload_min=-1)
    with pytest.raises(ValueError):
        PublicMassFamily(delays=(True,))
