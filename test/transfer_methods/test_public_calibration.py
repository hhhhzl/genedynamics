"""Real physics checks for the public probe/training information boundary."""
import json
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.runtime import MechanismSpec
from genedynamics.solvers.single.cross_ebdm.learning.physics_calibration import (
    PROBE_VERSION, PublicProbe, PublicPhysicsTwin, PublicQueryLedger, CountedPublicTask,
    fit_public_twin, probe_loss,
)


def real_probe(mechanism=MechanismSpec(.65,3,.5), steps=30):
    task = instantiate_task("T3",robot="xarm7",mechanism_spec=mechanism)
    public = task.reset((22900,),seed=22900,initial="clear")
    rows, requests = [public], []
    for step in range(steps):
        command = np.zeros((1,13),np.float32)
        command[0,:2] = [.025 if step<15 else -.025,.012]
        command[0,5] = .12
        requests.append(command[0])
        rows.append(task.step(command).observation)
    metadata = dict(version=PROBE_VERSION,index=0,purpose="fit",seed=22900,sample_id=22900,initial="clear",kind="free")
    return task, PublicProbe(metadata,{k:np.concatenate([r[k] for r in rows]) for k in rows[0]},requests)


def test_oracle_twin_reconstructs_cold_reset_and_hidden_target_mutations_do_not_enter():
    task, record = real_probe()
    twin = PublicPhysicsTwin(task.spec,MechanismSpec(.65,3,.5))
    predicted, accounting = twin.predict(record)
    for key in record.observations:
        np.testing.assert_array_equal(predicted[key],record.observations[key])
    assert accounting["control_steps"] == 30
    assert probe_loss(record.observations,predicted,free=True) == 0
    # These are evaluator-only objects. The sealed training data and twin
    # prediction remain independent of every subsequent target carry change.
    digest = record.digest
    task.state.command_buffer[:] = 1
    task.state.force_integral[:] = 9
    task.state.reference_position[:] += .03
    after,_ = twin.predict(record)
    assert record.digest == digest
    for key in predicted: np.testing.assert_array_equal(predicted[key],after[key])
    wrong,_ = PublicPhysicsTwin(task.spec,MechanismSpec(1.,0,.5)).predict(record)
    assert probe_loss(record.observations,wrong,free=True) > .001


def test_probe_seal_rejects_private_metadata_arrays_and_modification(tmp_path):
    _,record = real_probe(steps=3)
    metadata = dict(record.metadata, command_buffer=[0.]*13)
    with pytest.raises(ValueError,match="metadata"):
        PublicProbe(metadata,record.observations,record.requests)
    with pytest.raises(ValueError,match="extra/missing"):
        PublicProbe(dict(record.metadata),dict(record.observations,physics=np.zeros((4,7))),record.requests)
    with pytest.raises(ValueError): record.requests[0,0] = 1
    path = tmp_path / "probe"; record.save(path)
    assert PublicProbe.load(path).digest == record.digest
    with np.load(path / "public.npz") as data: arrays = {k:data[k] for k in data.files}
    np.savez_compressed(path / "public.npz",**arrays,command_buffer=np.zeros(13))
    with pytest.raises(ValueError,match="private/unknown"): PublicProbe.load(path)
    arrays["observation.tcp_position"] = arrays["observation.tcp_position"].copy()
    arrays["observation.tcp_position"][1,0] += .01
    np.savez_compressed(path / "public.npz",**arrays)
    with pytest.raises(ValueError,match="sealed"): PublicProbe.load(path)


def test_validation_never_enters_parameter_selection_and_legacy_records_rejected():
    task,record = real_probe(steps=3)
    validation = PublicProbe(dict(record.metadata,purpose="validation"),record.observations,record.requests)
    with pytest.raises(ValueError,match="fit probes"):
        fit_public_twin(task.spec,[validation])
    with pytest.raises(TypeError,match="legacy"):
        PublicPhysicsTwin(task.spec,MechanismSpec()).predict(deepcopy(task.records[0]))


def test_query_guard_persists_before_failed_physics_and_cannot_resume_for_free(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = PublicQueryLedger(path,limits={"real_probe":1})
    task = CountedPublicTask(task_spec("T3",robot="xarm7"),ledger=ledger,role="real_probe")
    task.reset((22901,),seed=22901)
    def fail(*args,**kwargs): raise RuntimeError("in-flight physics interruption")
    task.runner.collect_chunk = fail
    with pytest.raises(RuntimeError,match="interruption"): task.step(np.zeros((1,13)))
    row = json.loads(path.read_text())["roles"]["real_probe"]
    assert row["command_requests"] == 1
    assert row["unconfirmed_physics_steps_upper_bound"] == 10
    with pytest.raises(RuntimeError,match="ceiling"): task.step(np.zeros((1,13)))
    with pytest.raises(FileExistsError): PublicQueryLedger(path,limits={"real_probe":1})


def test_model_early_failure_is_penalized_instead_of_dropping_missing_prefix():
    _,record = real_probe(steps=3)
    predicted = {k:v[:2] for k,v in record.observations.items()}
    assert probe_loss(record.observations,predicted,free=True) > 100000
