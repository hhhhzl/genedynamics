"""Task-owned actors for ASID-identified downstream public policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from .obstacle_scaled_path import PublicScaledObstaclePathPolicy


def public_obstacle_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], *, report_pointer: dict[str, str], mode: int,
) -> PublicScaledObstaclePathPolicy:
    from genedynamics.tasks.transfer.obstacle_asid import GRID_VERSION

    if (checkpoint.get("schema_version") != "asid-obstacle-public-policy-checkpoint-v1"
            or checkpoint.get("task") != "T1"
            or checkpoint.get("robot") != "xarm7"
            or checkpoint.get("development_only") is not True
            or checkpoint.get("source_report") != report_pointer
            or recipe.get("version") != GRID_VERSION
            or recipe.get("task") != "T1"
            or recipe.get("robot") != "xarm7"
            or report.get("version") != GRID_VERSION
            or report.get("status") != "complete_development_chain"
            or report.get("formal_performance_evidence") is not False):
        raise ValueError("ASID training and checkpoint identities differ")
    policy = checkpoint.get("policy")
    if policy != report.get("task_policy", {}).get("best_policy"):
        raise ValueError("ASID final public policy differs from training report")
    limits = recipe["task_policy"]
    vector = np.asarray(policy, dtype=np.float64)
    if (vector.shape != (6,) or not np.isfinite(vector).all()
            or np.any(vector < np.asarray(limits["lower"]))
            or np.any(vector > np.asarray(limits["upper"]))):
        raise ValueError("ASID final policy is outside the declared training support")
    return PublicScaledObstaclePathPolicy(task.spec, vector, mode=mode)


def public_guide_checkpoint_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *,
    report_pointer: dict[str, str], mode: int,
):
    """Bind a completed, independently audited T2 ASID public guide policy."""
    from .adaptsim import public_guide_actor
    from genedynamics.tasks.transfer.guide_asid import TRAINING_VERSION

    if (checkpoint.get("schema_version") != "asid-guide-public-policy-checkpoint-v1"
            or checkpoint.get("task") != "T2"
            or checkpoint.get("robot") != "xarm7"
            or checkpoint.get("development_only") is not True
            or checkpoint.get("source_report") != report_pointer
            or checkpoint.get("identified_mechanism") != report.get("identification", {}).get("best_parameters")
            or recipe.get("version") != TRAINING_VERSION
            or recipe.get("task") != "T2"
            or recipe.get("robot") != "xarm7"
            or report.get("version") != TRAINING_VERSION
            or report.get("status") != "complete_development_chain"
            or report.get("formal_performance_evidence") is not False
            or audit.get("schema_version") != "asid-t2-full-chain-independent-audit-v1"
            or audit.get("status") != "passed_development_only"
            or audit.get("report_sha256") != report_pointer["sha256"]
            or audit.get("formal_performance_evidence") is not False):
        raise ValueError("ASID T2 training, audit, and checkpoint identities differ")
    policy = checkpoint.get("policy")
    if policy != report.get("task_policy", {}).get("best_policy"):
        raise ValueError("ASID T2 final public policy differs from training report")
    vector = np.asarray(policy, dtype=np.float64)
    support = recipe["task_policy"]
    if (vector.shape != (3,) or not np.isfinite(vector).all()
            or np.any(vector < np.asarray(support["lower"]))
            or np.any(vector > np.asarray(support["upper"]))):
        raise ValueError("ASID T2 final public guide is outside declared training support")
    return public_guide_actor(task, vector, mode=mode)


def public_mating_checkpoint_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *,
    report_pointer: dict[str, str], mode: int, seed: int,
):
    """Bind the audited T3 ASID policy to the shared public mating actor."""
    from genedynamics.tasks.transfer.controllers.mating_feedback import (
        PublicLivenessMatingFeedbackPolicy,
    )
    from genedynamics.tasks.transfer.mating_asid import TRAINING_VERSION
    from genedynamics.tasks.transfer.mating_policy_search import MatingPolicySearchTask

    if (task.spec.alias != "T3" or task.spec.config.robot != "xarm7"
            or mode not in (0, 1)
            or checkpoint.get("schema_version") != "asid-mating-public-policy-checkpoint-v1"
            or checkpoint.get("task") != "T3"
            or checkpoint.get("robot") != "xarm7"
            or checkpoint.get("development_only") is not True
            or checkpoint.get("source_report") != report_pointer
            or checkpoint.get("identified_mechanism") != report.get("identification", {}).get("best_parameters")
            or recipe.get("version") != TRAINING_VERSION
            or recipe.get("task") != "T3"
            or recipe.get("robot") != "xarm7"
            or recipe.get("shared_execution") != task.shared_execution
            or report.get("version") != TRAINING_VERSION
            or report.get("status") != "complete_development_chain"
            or report.get("formal_performance_evidence") is not False
            or audit.get("schema_version") != "asid-t3-full-chain-independent-audit-v1"
            or audit.get("status") != "passed_development_only"
            or audit.get("report_sha256") != report_pointer["sha256"]
            or audit.get("formal_performance_evidence") is not False):
        raise ValueError("ASID T3 training, audit, and checkpoint identities differ")
    policy = checkpoint.get("policy")
    if policy != report.get("task_policy", {}).get("best_policy"):
        raise ValueError("ASID T3 final public policy differs from training report")
    search = MatingPolicySearchTask(
        robot="xarm7", feedback_base=recipe["feedback_base"],
        shared_execution=recipe["shared_execution"],
    )
    return PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )


def public_mobile_mating_checkpoint_actor(
    task: Any, checkpoint: dict[str, Any], recipe: dict[str, Any],
    report: dict[str, Any], audit: dict[str, Any], *,
    report_pointer: dict[str, str], mode: int, seed: int,
):
    """Bind one audited T4 ASID policy to the shared mobile public actor."""
    from .mating_feedback import PublicLivenessMatingFeedbackPolicy
    from genedynamics.tasks.transfer.mobile_asid_training import VERSION
    from genedynamics.tasks.transfer.mobile_mating_policy_search import (
        MOBILE_CONFIG, MobileMatingPolicySearchTask,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    if (task.spec.alias != "T4" or task.spec.config.robot != "go2_z1"
            or mode not in (0, 1)
            or task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG).config
            != task.spec.config
            or checkpoint.get("schema_version") != "asid-mobile-public-policy-checkpoint-v2"
            or checkpoint.get("task") != "T4"
            or checkpoint.get("robot") != "go2_z1"
            or checkpoint.get("development_only") is not True
            or checkpoint.get("source_report") != report_pointer
            or checkpoint.get("identified_mechanism") != report.get("identification", {}).get("best_parameters")
            or recipe.get("version") != VERSION
            or recipe.get("task") != "T4"
            or recipe.get("robot") != "go2_z1"
            or recipe.get("shared_execution") != task.shared_execution
            or report.get("version") != VERSION
            or report.get("status") != "complete_development_chain"
            or report.get("formal_performance_evidence") is not False
            or audit.get("schema_version") != "asid-t4-full-chain-independent-audit-v2"
            or audit.get("status") != "passed_development_only"
            or audit.get("report_sha256") != report_pointer["sha256"]
            or audit.get("formal_performance_evidence") is not False):
        raise ValueError("ASID T4 training, audit, task or checkpoint identity differs")
    policy = checkpoint.get("policy")
    if policy != report.get("task_policy", {}).get("best_policy"):
        raise ValueError("ASID T4 public policy differs from the training report")
    search = MobileMatingPolicySearchTask(
        feedback_base=recipe["feedback_base"],
        shared_execution=recipe["shared_execution"],
        reset_seed_pool=tuple(recipe["reset_seed_pool"]),
    )
    return PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )
