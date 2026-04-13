#!/usr/bin/env python3
"""Verify the full co-design pipeline (up to SoftZoo rollout boundary)."""

import os
import sys
import yaml

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/workspace")

print("=== E2E Pipeline Verification ===")

# 1. YAML config
with open("configs/co_design/phase_a_smoke.yaml") as f:
    cfg = yaml.safe_load(f)
print("[PASS] YAML config loaded")

# 2. Scheduler
from genedynamics.experiments.common.constraints import create_scheduler_from_config
scheduler = create_scheduler_from_config(cfg["scheduler_config"], "jax")
from genedynamics.core.constraints.core.types import ScheduleState
dp = scheduler.diffusion_schedulers[0].diffusion_params(ScheduleState(k=0, K=1))
assert dp["M_k"] == 2 and dp["Ndiffuse"] == 2
mk = dp["M_k"]
nd = dp["Ndiffuse"]
tk = dp["T_k"]
print(f"[PASS] CompositeScheduler: M_k={mk}, Ndiffuse={nd}, T_k={tk}")

# 3. Platform + BaselineConfig
from genedynamics.experiments.framework.baseline_platform import (
    BaselineExperimentConfig,
    BaselineExperimentPlatform,
)
from genedynamics.experiments.framework.baseline import BaselineConfig

exp_cfg = CoDesignExperimentConfig(
    baseline_name=cfg["baseline_name"],
    task_id=cfg["task_id"],
    seeds=cfg.get("seeds", [0]),
    scheduler_config=cfg.get("scheduler_config"),
    method_params=cfg.get("method_params", {}),
    evaluator_runtime=cfg.get("evaluator_runtime", {}),
)
platform = CoDesignExperimentPlatform(
    exp_cfg, project_root="/workspace", use_checkpointing=False, use_logging=False
)
sched = platform._create_scheduler()
mp = platform._resolve_method_params()
bl_config = BaselineConfig(task_id=cfg["task_id"], seed=0, extra=mp, scheduler=sched)
dp2 = bl_config.get_diffusion_params()
assert dp2["Ndiffuse"] == 2
print(f"[PASS] Platform + BaselineConfig: K={dp2['Ndiffuse']}, M={dp2['M_k']}")

# 4. Baseline registry
from genedynamics.experiments.framework.baseline_registry import list_baselines
import genedynamics.experiments.framework.baselines  # noqa: F401
bl = list_baselines()
assert "mrmfmbd" in bl and "cmaes" in bl
print(f"[PASS] Baselines: {bl}")

# 5. Param extraction
from genedynamics.core.inference.annealed_bridge import BridgeScheduleConfig, LinearBridgeSchedule

extra = bl_config.extra
diff_params = bl_config.get_diffusion_params()
K = int(diff_params.get("Ndiffuse", extra.get("K", 50)))
M = int(diff_params.get("M_k", extra.get("M", 8)))
T = float(diff_params.get("T_k", extra.get("reward_temperature", 0.1)))
beta0 = float(diff_params.get("beta0", extra.get("beta0", 1e-6)))
betaK = float(diff_params.get("betaT", extra.get("betaT", 1.0)))
print(f"[PASS] MRMFMBD params: K={K}, M={M}, T={T}, beta0={beta0}, betaK={betaK}")

bsc = BridgeScheduleConfig(K=K, beta0=beta0, betaK=betaK)
schedule = LinearBridgeSchedule(bsc)
b0 = schedule.beta(0)
bk = schedule.beta(K - 1)
print(f"[PASS] BridgeSchedule: beta(0)={b0:.6f}, beta({K-1})={bk:.6f}")

# 6. S1 + S3
from genedynamics.solvers.single.mrmfmbd import (
    ModeMarginalizerS1,
    default_mode_system_config,
    create_fidelity_ladder,
)

mm = ModeMarginalizerS1(default_mode_system_config(1), backend="jax")
fl = create_fidelity_ladder(K=K, num_levels=1, ladder_type="geometric", step_ratio=1.5)
print("[PASS] S1 ModeMarginalizerS1 + S3 FidelityLadder")

# 7. Evaluator
from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig

evaluator = SoftZooRolloutEvaluator(
    config=SoftZooEvaluatorConfig(max_workers=0, cache_size=64),
    project_root="/workspace",
    runtime_config=cfg.get("evaluator_runtime", {}),
)
print("[PASS] SoftZooRolloutEvaluator instantiated")

# 8. SoftZoo import
from genedynamics.envs.external.softzoo.bootstrap import ensure_softzoo_on_path

code_root = ensure_softzoo_on_path()
print(f"[PASS] SoftZoo on path: {code_root}")

# 9. Posterior backend
from genedynamics.solvers.single.mrmfmbd import (
    ThetaParametrization,
    ThetaPrior,
    ThetaPriorConfig,
    MRMFMBDPosteriorBackendJax,
    PosteriorBridgeConfig,
)

tp = ThetaParametrization(x_dim=3, phi_dim=4, x_bounds=(0.01, 2.0), phi_bounds=(5.0, 150.0))
prior = ThetaPrior(tp, ThetaPriorConfig(x_std=0.5, phi_std=30.0))
backend = MRMFMBDPosteriorBackendJax(
    evaluator=evaluator,
    theta_param=tp,
    theta_prior=prior,
    bridge_schedule=schedule,
    mode_marginalizer=mm,
    fidelity_ladder=fl,
    config=PosteriorBridgeConfig(K=K, M=M, reward_temperature=T, fine_fidelity_level=0),
    task_id="crawling_ground",
    num_modes=1,
    seed=0,
    show_tqdm=False,
)
print(f"[PASS] MRMFMBDPosteriorBackendJax (theta_dim={tp.theta_dim})")

print()
print("=== All framework checks PASSED ===")
print(
    "NOTE: Actual SoftZoo rollout requires native x86_64 Linux "
    "(Taichi LLVM incompatible with Rosetta emulation)."
)
