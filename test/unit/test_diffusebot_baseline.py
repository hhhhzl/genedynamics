"""Stage 5 — DiffuseBot (gradient co-design) optimizer registration.

CPU: importing solvers/single/codesign_solvers registers the DiffuseBot co-design
optimizer into the co-design solver registry; the gradient run() differentiates
through the MPM rollout and is verified on GPU. (Post-refactor: there is no
`experiments/.../baselines/` layer — optimizers are registered solvers.)
"""

from __future__ import annotations


def test_diffusebot_codesign_registered():
    import genedynamics.solvers.single.codesign_solvers  # noqa: F401  (registers)
    from genedynamics.experiments.framework.codesign_runner import (
        list_codesign_solvers,
        get_codesign_solver,
    )
    assert "diffusebot" in list_codesign_solvers()
    assert callable(get_codesign_solver("diffusebot"))


def test_diffusebot_run_requires_jax_mpm_evaluator():
    """Without a JAX-MPM evaluator (no `_scene`/`_mpm_cfg`), run() must fail
    loudly rather than silently no-op."""
    import pytest
    from genedynamics.solvers.single.codesign_optimizers.diffusebot import DiffuseBotBaseline
    from genedynamics.experiments.framework.baseline import BaselineConfig

    bl = DiffuseBotBaseline()
    cfg = BaselineConfig(task_id="crawling_ground", extra={})
    with pytest.raises(RuntimeError):
        bl.run(cfg, evaluator=object(), task_spec=None, x_dim=27, phi_dim=80)
