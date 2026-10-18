import torch

from genedynamics.experiments.plugins.methods.transfer.dial import TorchDIALPlanner


def test_dial_sampling_moves_future_plan_toward_quadratic_optimum():
    planner=TorchDIALPlanner(seed=7)
    before=planner.dense(planner.nodes).clone()
    action=planner.replan(lambda values:(values-.4).square().mean((1,2)))
    after=planner.dense(planner.nodes)
    assert torch.allclose(action,torch.zeros_like(action),rtol=0,atol=2e-8)
    assert (after[:-1]-.4).abs().mean() < (before[:-1]-.4).abs().mean()
    assert planner.candidate_rollouts == 10*2049


def test_dial_state_replays_next_replan_exactly():
    planner=TorchDIALPlanner(seed=11)
    cost=lambda values:(values+.2).square().mean((1,2))
    planner.replan(cost);state=planner.state();first=planner.replan(cost);after=planner.state()
    planner.restore(state);second=planner.replan(cost)
    assert torch.equal(first,second)
    assert torch.equal(after["nodes"],planner.state()["nodes"])
    assert after["candidate_rollouts"] == planner.state()["candidate_rollouts"]
