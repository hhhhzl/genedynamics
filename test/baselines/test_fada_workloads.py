"""Real staggered Panda episodes exercise workload partial reset and charging."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from genedynamics.experiments.baselines import instantiate_task
from scripts.baselines.fada_workloads import execute_workload


class ZeroPolicy:
    """Explicit deterministic control fixture, with real MuJoCo transitions."""
    def act(self, observation):
        return torch.zeros((len(observation["history_obs"]), 13))

    def reset(self):
        pass


def test_real_staggered_termination_resets_only_finished_slots_and_history():
    task = instantiate_task("T3", robot="panda", seed=28600, history_len=4,
                            config_overrides={"timeout_seconds": .06})
    identities = (28600, 28601)
    task.reset(identities, seed=28600)
    task.step(np.zeros((2, 13), dtype=np.float32))
    task.reset(identities, seed=28601, mask=[False, True])
    task.set_modes([0, 1])
    profile, actions, statuses = execute_workload(task, ZeroPolicy(), sample_ids=identities,
        seed=28602, device="cpu", iterations=4)
    assert [row["mask"] for row in profile["partial_resets"]] == [[True, False], [False, True]]
    assert profile["history_valid_counts_before_step"][2:] == [[1, 3], [2, 1]]
    assert profile["physical_accounting_delta_after_initial_reset"]["control_steps"] == 8
    assert profile["physical_accounting_delta_after_initial_reset"]["physics_steps"] == 80
    assert profile["physical_accounting_delta_after_initial_reset"]["reset_worlds"] == 2
    assert actions.shape == (4, 2, 13) and statuses.shape == (4, 2)
    assert task.evaluate()["status_mismatches"] == 0
    assert len(task.records) == 5
