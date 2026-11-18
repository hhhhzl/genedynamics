"""Task-owned binding from a trained AdaptSim policy to T3 public feedback."""

from __future__ import annotations

from typing import Any

import numpy as np

from .obstacle_scaled_path import PublicScaledObstaclePathPolicy
from .guide_contact import PublicResponseNormalizedGuideCompletionPolicy
from genedynamics.tasks.transfer.reposition_dock import (
    mobile_mating_training_config_compatible,
)


GUIDE_POLICY_FIELDS = (
    "authority_growth_per_control", "nominal_lead_m",
    "maximum_contact_speed_mps",
)
GUIDE_POLICY_LOWER = np.asarray([.01, .006, .02], dtype=np.float64)
GUIDE_POLICY_UPPER = np.asarray([.04, .01, .04], dtype=np.float64)


class PublicAdaptSimGuidePolicy(PublicResponseNormalizedGuideCompletionPolicy):
    """Own only public T2 gains; physical execution remains task-shared."""

    def __init__(self, spec: Any, policy: Any, *, mode: int):
        vector = np.asarray(policy, dtype=np.float64)
        if (spec.alias != "T2" or spec.config.robot != "xarm7"
                or mode not in (0, 1) or vector.shape != (3,)
                or not np.isfinite(vector).all()
                or np.any(vector < GUIDE_POLICY_LOWER)
                or np.any(vector > GUIDE_POLICY_UPPER)):
            raise ValueError("AdaptSim T2 public guide policy outside declared support")
        super().__init__(spec, mode=mode)
        self.AUTHORITY_GROWTH_PER_CONTROL = float(vector[0])
        self.NOMINAL_LEAD_M = float(vector[1])
        self.MAXIMUM_CONTACT_SPEED_MPS = float(vector[2])


def public_guide_actor(task: Any, policy: Any, *, mode: int):
    """Decode a bounded T2 AdaptSim task policy into public feedback only."""
    return PublicAdaptSimGuidePolicy(task.spec, policy, mode=mode)


def public_guide_checkpoint_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *,
    checkpoint_sha256: str, report_sha256: str, mode: int,
):
    """Admit only the audited T2 population policy under its exact task contract."""
    from pathlib import Path
    from genedynamics.tasks.transfer.guide_policy_search import (
        TRAINING_VERSION, _task_config,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    if task.spec.alias != "T2" or task.spec.config.robot != "xarm7":
        raise ValueError("AdaptSim guide policy is trained only for T2/xarm7")
    if (checkpoint.get("version") != TRAINING_VERSION
            or recipe.get("version") != TRAINING_VERSION
            or recipe.get("task") != "T2" or recipe.get("robot") != "xarm7"
            or report.get("version") != TRAINING_VERSION
            or report.get("status") != "complete_development"
            or recipe.get("policy_fields") != list(GUIDE_POLICY_FIELDS)
            or recipe.get("formal_performance_evidence") is not False
            or report.get("formal_performance_evidence") is not False
            or report.get("checkpoint", {}).get("sha256") != checkpoint_sha256
            or audit.get("schema_version") != "adaptsim-t2-population-independent-ledger-audit-v1"
            or audit.get("status") != "passed"
            or audit.get("report_sha256") != report_sha256
            or audit.get("checkpoint_sha256") != checkpoint_sha256
            or audit.get("physical_episodes_accounted") != 882):
        raise ValueError("AdaptSim T2 training, checkpoint, or audit identities differ")
    root = Path(__file__).resolve().parents[4]
    overrides, shared = _task_config(recipe, root)
    if (shared != task.shared_execution
            or task_spec("T2", robot="xarm7", config_overrides=overrides).config
            != task.spec.config):
        raise ValueError("AdaptSim T2 task or shared execution differs from training")
    policy = checkpoint.get("final_policy")
    if policy is None or policy != report.get("deployment", {}).get("final_policy"):
        raise ValueError("AdaptSim T2 final policy differs from training report")
    return public_guide_actor(task, policy, mode=mode)


def public_obstacle_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *, checkpoint_sha256: str,
    report_sha256: str, mode: int,
) -> PublicScaledObstaclePathPolicy:
    """Bind the audited AdaptSim T1 final policy without simulator truth."""
    from genedynamics.tasks.transfer.obstacle_policy_search import META_DEVELOPMENT_VERSION

    version = META_DEVELOPMENT_VERSION
    if task.spec.alias != "T1" or task.spec.config.robot != "xarm7":
        raise ValueError("AdaptSim obstacle policy is trained only for T1/xarm7")
    independent = recipe.get("training_cohort") == "independent_main_table_training_v1"
    audit_status = "passed_development_only" if independent else "passed"
    if (checkpoint.get("version") != version
            or recipe.get("version") != version
            or recipe.get("task") != "T1" or recipe.get("robot") != "xarm7"
            or report.get("version") != version
            or report.get("status") != "complete_meta_development"
            or report.get("formal_performance_evidence") is not False
            or report.get("checkpoint", {}).get("sha256") != checkpoint_sha256
            or audit.get("status") != audit_status
            or audit.get("report_sha256") != report_sha256
            or audit.get("formal_performance_evidence") is not False):
        raise ValueError("AdaptSim T1 checkpoint and audited training identities differ")
    if independent and (
        audit.get("schema_version") != "adaptsim-t1-independent-training-ledger-audit-v1"
        or audit.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("AdaptSim T1 independent training ledger differs")
    policy = np.asarray(checkpoint.get("final_policy"), dtype=np.float64)
    bounds = recipe["task_policy_search"]
    if (policy.shape != (6,) or not np.isfinite(policy).all()
            or np.any(policy < np.asarray(bounds["lower"]))
            or np.any(policy > np.asarray(bounds["upper"]))):
        raise ValueError("AdaptSim T1 policy is outside the declared training support")
    return PublicScaledObstaclePathPolicy(task.spec, policy, mode=mode)


def public_mating_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], *, checkpoint_sha256: str, mode: int, seed: int,
):
    from genedynamics.tasks.transfer.mating_policy_search import (
        MatingPolicySearchTask, POLICY_FIELDS, POPULATION_DEVELOPMENT_VERSION,
    )
    from .mating_feedback import PublicLivenessMatingFeedbackPolicy

    version = POPULATION_DEVELOPMENT_VERSION
    if task.spec.alias != "T3" or task.spec.config.robot != "xarm7":
        raise ValueError("AdaptSim mating policy is trained only for T3/xarm7")
    if mode not in (0, 1):
        raise ValueError("AdaptSim mating policy requires a public mode")
    if (checkpoint.get("version") != version
            or recipe.get("version") != version
            or report.get("version") != version
            or report.get("status") != "complete_development"):
        raise ValueError("AdaptSim training identity or completion differs")
    if recipe.get("task") != "T3" or recipe.get("robot") != "xarm7":
        raise ValueError("AdaptSim recipe task or robot differs")
    if recipe.get("policy_fields") != list(POLICY_FIELDS):
        raise ValueError("AdaptSim policy chart differs from the recipe")
    if recipe.get("shared_execution") != task.shared_execution:
        raise ValueError("AdaptSim recipe shared execution differs from the task")
    if report.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("AdaptSim report checkpoint identity differs")
    if report.get("formal_performance_evidence") is not False:
        raise ValueError("AdaptSim training artifact is mislabeled as formal evidence")
    policy = checkpoint.get("final_policy")
    if (policy is None or policy != report.get("deployment", {}).get("final_policy")):
        raise ValueError("AdaptSim final policy differs from the training report")
    search = MatingPolicySearchTask(
        robot="xarm7", feedback_base=recipe["feedback_base"],
        shared_execution=recipe["shared_execution"],
    )
    return PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )


def public_mobile_mating_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *,
    checkpoint_sha256: str, report_sha256: str, mode: int, seed: int,
):
    """Bind the independently reconciled T4 AdaptSim policy to public feedback."""
    from genedynamics.tasks.transfer.mobile_mating_policy_search import (
        MOBILE_CONFIG, POLICY_FIELDS, TRAINING_VERSION, MobileMatingPolicySearchTask,
    )
    from genedynamics.tasks.transfer.specs import task_spec
    from .mating_feedback import PublicLivenessMatingFeedbackPolicy

    if task.spec.alias != "T4" or task.spec.config.robot != "go2_z1" or mode not in (0, 1):
        raise ValueError("AdaptSim mobile mating actor requires T4/go2_z1 and a public mode")
    if (checkpoint.get("version") != TRAINING_VERSION
            or recipe.get("version") != TRAINING_VERSION
            or recipe.get("task") != "T4" or recipe.get("robot") != "go2_z1"
            or recipe.get("policy_fields") != list(POLICY_FIELDS)
            or recipe.get("shared_execution") != task.shared_execution
            or (
                task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG).config
                != task.spec.config
                and not mobile_mating_training_config_compatible(task)
            )
            or report.get("version") != TRAINING_VERSION
            or report.get("task") != "T4"
            or report.get("status") != "complete_development"
            or report.get("checkpoint", {}).get("sha256") != checkpoint_sha256
            or report.get("formal_performance_evidence") is not False
            or audit.get("schema_version") != "adaptsim-t4-population-independent-ledger-audit-v1"
            or audit.get("status") != "passed"
            or audit.get("report_sha256") != report_sha256
            or audit.get("checkpoint_sha256") != checkpoint_sha256
            or audit.get("physical_episodes_accounted") != 882):
        raise ValueError("AdaptSim T4 training, checkpoint, audit or task identity differs")
    policy = checkpoint.get("final_policy")
    if policy is None or policy != report.get("deployment", {}).get("final_policy"):
        raise ValueError("AdaptSim T4 final policy differs from training report")
    search = MobileMatingPolicySearchTask(
        feedback_base=recipe["feedback_base"],
        shared_execution=recipe["shared_execution"],
    )
    return PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )
