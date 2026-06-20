"""Generic co-design experiment runner — drives any REGISTERED solver, no baselines.

The experiment framework is task-agnostic: it builds the co-design problem
(a (dynamics, energy) env wrapping the JAX-MPM evaluator) and runs a co-design
optimizer looked up by name from the co-design solver registry. There is no
`BaselineProtocol` / `baselines/` layer — each optimizer is a registered solver.

Two kinds of registered solver coexist uniformly behind one `run(...)` call:
  * GENERAL solvers (CEM, CMA-ES) consume the co-design problem purely through
    the (dynamics, energy) contract — `build_codesign_problem` + the general
    `Solver(dynamics, energy).solve(x0, horizon=1)`. They are NOT co-design
    specific; co-design is just one (state, action) problem they solve.
  * SPECIALIZED co-design solvers (MRMFMBD, SHAC, DiffuseBot) read the richer
    problem (scene/cfg/evaluator/regimes/fidelity) off the same env.

A registered co-design solver is a callable
    run(evaluator, task_spec, config, *, x_dim, phi_dim) -> CoDesignResult
where `config` carries method_params (the YAML `method_params`) + seed + scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Co-design solver registry (replaces experiments/.../baselines + baseline_registry)
# --------------------------------------------------------------------------- #
_CODESIGN_SOLVERS: Dict[str, Callable[..., "CoDesignResult"]] = {}


def register_codesign_solver(name: str, run_fn: Callable[..., "CoDesignResult"]) -> None:
    _CODESIGN_SOLVERS[str(name)] = run_fn


def get_codesign_solver(name: str) -> Callable[..., "CoDesignResult"]:
    if name not in _CODESIGN_SOLVERS:
        _ensure_registered()
    if name not in _CODESIGN_SOLVERS:
        raise KeyError(f"co-design solver {name!r} not registered. "
                       f"Available: {sorted(_CODESIGN_SOLVERS)}")
    return _CODESIGN_SOLVERS[name]


def list_codesign_solvers() -> List[str]:
    _ensure_registered()
    return sorted(_CODESIGN_SOLVERS)


_REGISTERED = False


def _general_solver_run(solver_cls, default_kw: Dict[str, Any]):
    """Adapter: run ANY general Solver(dynamics, energy) on the co-design env.

    Used for the general algorithms (CEM, CMA-ES) — they are NOT co-design
    specific; co-design is just a horizon-1 (state, action) problem they solve.
    The design theta is recovered from the solver's best action via the env's
    reparameterization. No solver code is changed.
    """
    def run_fn(evaluator, task_spec, config, *, x_dim, phi_dim) -> CoDesignResult:
        import jax.numpy as jnp
        from genedynamics.core.backends.runtime import RuntimeBackendManager

        mp = config["method_params"]
        dynamics, energy, x0, x_opt_dim = codesign_problem_from_evaluator(
            evaluator, x_dim=x_dim, phi_dim=phi_dim, method_params=mp)

        RuntimeBackendManager.set_backend("jax")
        backend = RuntimeBackendManager.get_backend()
        kw = dict(default_kw)
        kw["num_samples"] = int(mp.get("num_samples", mp.get("M", kw.get("num_samples", 64))))
        kw["num_iterations"] = int(mp.get("num_iterations", mp.get("K", kw.get("num_iterations", 6))))
        solver = solver_cls(dynamics, energy, backend, horizon=1, dt=1.0, seed=int(config["seed"]), **kw)

        traj = solver.solve(x0, horizon=1)
        best_a = np.asarray(traj.actions[0], dtype=np.float32)
        theta = np.asarray(dynamics.action_to_theta(jnp.asarray(best_a)))
        reward = float(dynamics.jax_transition(jnp.zeros(1), jnp.asarray(best_a))[0])
        return CoDesignResult(
            theta=theta, x=theta[:x_opt_dim], phi=theta[x_opt_dim:],
            return_=reward, success=bool(reward > 0.0),
            num_evaluations=kw["num_samples"] * kw["num_iterations"],
            metadata={"solver": solver_cls.__name__, "kind": "general"},
        )
    return run_fn


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    # General sampling solvers consume the co-design env via (dynamics, energy).
    from genedynamics.solvers.single.cem.cem import CEMSolver
    register_codesign_solver("cem", _general_solver_run(
        CEMSolver, dict(num_samples=32, num_iterations=6, elite_frac=0.25,
                        init_std=0.8, min_std=0.1, action_limit=3.0)))
    from genedynamics.solvers.single.cmaes.cmaes import CMAESSolver
    register_codesign_solver("cmaes", _general_solver_run(
        CMAESSolver, dict(num_samples=32, num_iterations=6, elite_frac=0.5,
                          sigma0=0.8, min_std=0.05, action_limit=3.0)))
    # Specialized co-design solvers (mrmfmbd / shac / diffusebot) register
    # themselves on import; pull them in here as they are migrated.
    try:
        from genedynamics.solvers.single import codesign_solvers  # noqa: F401
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Uniform result (was BaselineResult)
# --------------------------------------------------------------------------- #
@dataclass
class CoDesignResult:
    theta: np.ndarray
    x: np.ndarray
    phi: np.ndarray
    return_: float
    success: bool
    num_evaluations: int = 0
    wall_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # When a wrapped optimizer already produces a full JSON-safe result dict
    # (e.g. MRMFMBD's bridge_history / theta_history / final_eval_returns), pass
    # it through verbatim so nothing is lost downstream (renderer / analysis).
    full_dict: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.full_dict is not None:
            return self.full_dict
        return {
            "theta": np.asarray(self.theta).tolist(),
            "x": np.asarray(self.x).tolist(),
            "phi": np.asarray(self.phi).tolist(),
            "return_": float(self.return_),
            "success": bool(self.success),
            "num_evaluations": int(self.num_evaluations),
            "wall_time": float(self.wall_time),
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Helpers shared by general-solver co-design adapters
# --------------------------------------------------------------------------- #
def codesign_problem_from_evaluator(evaluator, *, x_dim, phi_dim, method_params):
    """Build the (dynamics, energy, x0) co-design problem from a JAX-MPM evaluator.

    Reads the scene/cfg the evaluator already constructed and the morphology
    bounds/symmetry from method_params (YAML). Returns (dynamics, energy, x0,
    x_opt_dim) so callers can map the solver's best action back to theta.
    """
    from genedynamics.envs.external.jax_mpm.codesign_env import build_codesign_problem

    mp = dict(method_params or {})
    voxel_dims = mp.get("voxel_dims")
    z_sym = str(mp.get("morphology_symmetry", "")).lower() == "z"

    # A2 decoder: when method_params request the latent path, the GENERAL solver
    # (CEM/CMA-ES) optimizes the SAME 32-d latent w as MRMFMBD/DiffuseBot (fair
    # Table-1: same morphology search space; only the OPTIMIZER differs).
    morph_decoder = None
    morph_latent_dim = int(mp.get("morph_latent_dim", 0))
    dpath = str(mp.get("morph_decoder_path", "") or "")
    x_lo, x_hi, x_mean = float(mp.get("x_lo", 0.2)), float(mp.get("x_hi", 1.0)), float(mp.get("x_mean", 0.6))
    if morph_latent_dim > 0 and dpath:
        import json as _json, os as _os
        from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
        from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
        _dc = _json.load(open(_os.path.join(dpath, "decoder_config.json")))
        _mc = MorphDecoderConfig(latent_dim=int(_dc["latent_dim"]), hidden_dim=int(_dc["hidden_dim"]),
            n_voxels=int(_dc["n_voxels"]), x_lo=float(_dc["x_lo"]), x_hi=float(_dc["x_hi"]),
            beta_kl=float(_dc.get("beta_kl", 1e-3)), decode_actuator=bool(_dc.get("decode_actuator", False)),
            decode_stiffness=bool(_dc.get("decode_stiffness", False)), n_actuators=int(_dc.get("n_actuators", 0)))
        morph_decoder = MorphDecoder.load(_os.path.join(dpath, "decoder_params.npz"), _mc)
        x_opt_dim = morph_latent_dim
        z_sym = False
        x_lo = float(mp.get("morph_latent_lo", -2.0)); x_hi = float(mp.get("morph_latent_hi", 2.0))
        x_mean = float(mp.get("morph_latent_mean", 0.0))
        voxel_dims = tuple(voxel_dims) if voxel_dims is not None else None
    elif voxel_dims is not None and z_sym:
        vx, vy, vz = (int(v) for v in voxel_dims)
        x_opt_dim = vx * vy * (vz // 2)
    else:
        x_opt_dim = int(x_dim)
        voxel_dims = tuple(voxel_dims) if voxel_dims is not None else None

    friction = float(evaluator._mode_friction[0]) if getattr(evaluator, "_mode_friction", None) else 0.5
    num_env_steps = int(mp.get("num_env_steps", getattr(evaluator._mpm_cfg, "env_horizon", 200)))

    # prior-seeded init: load the latent w0 (encoded 3D-prior body, e.g. TripoSG).
    morph_init = None
    _minit = str(mp.get("morph_init_path", "") or "")
    if _minit:
        import numpy as _np
        morph_init = _np.load(_minit).astype(_np.float32).ravel()

    dynamics, energy, x0 = build_codesign_problem(
        evaluator._scene, evaluator._mpm_cfg,
        x_opt_dim=x_opt_dim, phi_dim=int(phi_dim),
        x_lo=x_lo, x_hi=x_hi, x_mean=x_mean,
        phi_lo=float(mp.get("phi_lo", -0.5)), phi_hi=float(mp.get("phi_hi", 0.5)),
        phi_mean=float(mp.get("phi_mean", 0.0)),
        friction=friction, num_env_steps=num_env_steps,
        z_sym=z_sym, voxel_dims=voxel_dims, morph_decoder=morph_decoder, morph_init=morph_init,
    )
    return dynamics, energy, x0, x_opt_dim


# --------------------------------------------------------------------------- #
# Generic runner (replaces BaselineExperimentPlatform.run_single dispatch)
# --------------------------------------------------------------------------- #
def run_codesign(
    solver_name: str,
    *,
    task_domain: str,
    task_id: str,
    evaluator_params: Dict[str, Any],
    method_params: Dict[str, Any],
    seed: int = 0,
    scheduler: Optional[Any] = None,
    project_root: str = ".",
) -> CoDesignResult:
    """Build evaluator + co-design problem, run the registered solver, return result."""
    import time
    from genedynamics.experiments.framework.task_domain_provider import get_task_domain_provider

    provider = get_task_domain_provider(task_domain)
    task_spec = provider.get_task_spec(task_id)
    ev_params = dict(evaluator_params or {})
    evaluator = provider.create_evaluator(
        str(project_root),
        max_workers=ev_params.get("max_workers", 0),
        cache_size=ev_params.get("cache_size", 64),
        **{k: v for k, v in ev_params.items() if k not in ("max_workers", "cache_size")},
    )

    x_dim = int(getattr(task_spec, "x_dim", 3))
    phi_dim = int(getattr(task_spec, "phi_dim", 4))
    voxel_dims = ev_params.get("voxel_dims")
    mp = dict(method_params or {})
    if voxel_dims is not None:
        x_dim = int(voxel_dims[0]) * int(voxel_dims[1]) * int(voxel_dims[2])
        mp.setdefault("voxel_dims", list(voxel_dims))

    config = {"seed": int(seed), "scheduler": scheduler, "method_params": mp}
    run_fn = get_codesign_solver(solver_name)
    t0 = time.perf_counter()
    result = run_fn(evaluator, task_spec, config, x_dim=x_dim, phi_dim=phi_dim)
    if not result.wall_time:
        result.wall_time = time.perf_counter() - t0
    return result
