"""Stage 5 — DiffuseBot (gradient co-design) baseline registration.

CPU: importing the baselines package registers DiffuseBotBaseline; the gradient
run() differentiates through the MPM rollout and is verified on GPU.
"""

from __future__ import annotations


def test_diffusebot_baseline_registered():
    import genedynamics.experiments.framework.baselines as B    (registers)
    from genedynamics.experiments.framework.baseline_registry import (
        has_baseline,
        get_baseline,
        list_baselines,
    )
    assert "diffusebot" in list_baselines()
    assert has_baseline("diffusebot")
    assert get_baseline("diffusebot").name == "diffusebot"


def test_diffusebot_run_requires_jax_mpm_evaluator():
    """Without a JAX-MPM evaluator (no `_scene`/`_mpm_cfg`), run() must fail
    loudly rather than silently no-op."""
    import pytest
    from genedynamics.experiments.framework.baselines import DiffuseBotBaseline
    from genedynamics.experiments.framework.baseline import BaselineConfig

    bl = DiffuseBotBaseline()
    cfg = BaselineConfig(task_id="crawling_ground", extra={})
    with pytest.raises(RuntimeError):
        bl.run(cfg, evaluator=object(), task_spec=None, x_dim=27, phi_dim=80)
