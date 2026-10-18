import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from genedynamics.envs.domains.manipulation.transfer_workcell.runtime import TransferWorkcell
from genedynamics.tasks.transfer import ObstacleTransferConfig, ExecutionSpec, MechanismSpec, RandomRequest, SkillSpec
from genedynamics.solvers.single.cross_ebdm.learning.artifact_bundle import load_bundle, save_bundle
from genedynamics.solvers.single.cross_ebdm.learning.probe_budget import ProbeBudget
from genedynamics.core.backends.capabilities import Capabilities
from genedynamics.envs.domains.manipulation.transfer_workcell.registry import backend_registry


def test_public_import_without_numerical_runtimes():
    result = subprocess.run([sys.executable, "-S", "-c", """
import sys
from genedynamics.tasks.transfer import ObstacleTransferConfig
assert not ({'jax', 'torch', 'mujoco', 'numpy'} & set(sys.modules))
"""], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("factory", [lambda: ObstacleTransferConfig(control_dt=.003), lambda: ObstacleTransferConfig(max_delay_steps=.5),
                                      lambda: SkillSpec(horizon=.5), lambda: MechanismSpec(delay_steps=-1),
                                      lambda: RandomRequest(0, (1, 1)), lambda: ExecutionSpec(dtype="float64")])
def test_invalid_contracts(factory):
    with pytest.raises(ValueError):
        factory()


def test_frontend_delegates_custom_backend_without_tensor_types():
    class Fake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def capabilities(self):
            return Capabilities("fixture", "cpu", "fixture", frozenset({"collect", "evaluate"}))
        def reset_batch(self, request, **kwargs):
            return request.sample_ids
        def collect_chunk(self, state, commands):
            return state, commands
    backend_registry.register("test_fixture", Fake)
    runner = TransferWorkcell(execution=ExecutionSpec(backend="test_fixture"))
    assert runner.collect_chunk(runner.reset_batch(RandomRequest(0, (7,))), "opaque") == ((7,), "opaque")
    with pytest.raises(NotImplementedError, match="train"):
        runner.train_step(None, None, None)


def test_probe_failures_consume_budget_before_execution():
    budget = ProbeBudget(2)
    i = budget.reserve("fit")
    budget.complete(i, "failed")
    budget.reserve("validation")
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.reserve("fit")
    with pytest.raises(ValueError):
        budget.complete(-1, "success")
    with pytest.raises(ValueError, match="already"):
        budget.complete(i, "success")
    assert [a["outcome"] for a in budget.attempts] == ["failed", "pending"]


def test_portable_artifact_roundtrip_and_corruption(tmp_path):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_bundle(tmp_path, {"weight": array}, {"role": "fixture"})
    tensors, metadata = load_bundle(tmp_path)
    np.testing.assert_array_equal(tensors["weight"], array)
    assert metadata["linear_weight_layout"] == "in_out"
    import json
    tensor_file=json.loads((tmp_path/"metadata.json").read_text())["tensor_file"]
    with open(tmp_path / tensor_file, "ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_artifact_rejects_nonportable_tensors(tmp_path):
    with pytest.raises(ValueError):
        save_bundle(tmp_path, {"bad": np.array([object()])}, {})
