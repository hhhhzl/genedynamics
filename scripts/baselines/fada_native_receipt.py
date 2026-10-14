"""Aggregate native FADA execution evidence without promoting it to paper scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-run", type=Path, required=True)
    args = parser.parse_args()
    output, corpus = args.output.resolve(), args.corpus_run.resolve()
    if "latex" in output.parts or not corpus.is_relative_to(output):
        parser.error("Native receipts must be outside LaTeX and include their corpus run")

    def read(path):
        return json.loads(path.read_text())

    def evidence(path):
        if not path.is_file():
            raise FileNotFoundError(f"Required qualification evidence missing: {path}")
        return {"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size}

    summary = read(corpus / "summary.json")
    config = read(corpus / "config_snapshot.json")
    training = read(corpus / "training_log.json")
    source = read(output / "student_corpus_source_stability_metrics.json")
    pre = read(output / "student_corpus_pre_sft_metrics.json")
    post = read(output / "student_corpus_post_sft_metrics.json")
    collection = read(output / "student_corpus_collection_metrics.json")
    adaptation = read(output / "fixed_target_calibration/adaptation_metrics.json")
    resume = read(output / "corpus_resume/dagger_resume_metrics.json")
    warmup = summary["warmup_stats"]
    for evaluation in (source, pre, post, collection):
        if evaluation["history_reset_mask_dtype"] != "torch.bool":
            raise RuntimeError("Superseded integer-indexing evaluation cannot enter canonical receipt")
    if pre["matched_velocity_schedule_sha256"] != post["matched_velocity_schedule_sha256"]:
        raise RuntimeError("Pre/post target command schedules differ")
    target_hashes = pre["fixed_target_dynamics"]["coefficient_hashes"]
    if any(value["fixed_target_dynamics"]["coefficient_hashes"] != target_hashes for value in (post, collection)):
        raise RuntimeError("Calibration/pre/post target coefficient draws differ")
    if collection["aggregate_action_seconds"] != 120.0 or collection["num_envs"] != 1:
        raise RuntimeError("Canonical native calibration must be one fixed-target120-second recording")
    if adaptation["dataset_sha256"] != collection["dataset_sha256"]:
        raise RuntimeError("Adaptation did not consume the canonical target recording")
    if post["student_sha256"] != adaptation["checkpoint_sha256"]:
        raise RuntimeError("Post-SFT checkpoint differs from budgeted adapted snapshot")
    if not resume["planner_idm_optimizer_rng_normalization_next_update_exact"]:
        raise RuntimeError("Native learner next-update resume failed")
    checkpoint = corpus / "model_final.pt"
    if any(value["student_sha256"] != sha256(checkpoint) for value in (source, pre, collection)):
        raise RuntimeError("Canonical source/pre/collection policies differ")

    fixed_files = ["oracle_metrics.json", "oracle_logs/model_native_resume.pt", "oracle_rng.pt",
        "assets/author_g1_model_24999.pt", "assets/g1.manifest.json", "assets/g1.inventory.json",
        "assets/oracle_g1_mjwarp_envs4_seed42.pt", "oracle_derivations/mjwarp_envs4_seed42.json",
        "native_config.yaml", "native_requirements.lock.txt", "native_doctor.log",
        "student_corpus_source_stability_metrics.json", "student_corpus_pre_sft_metrics.json",
        "student_corpus_collection_metrics.json", "student_corpus_post_sft_metrics.json",
        "target_corpus_collection.h5", "fixed_target_calibration/adaptation_metrics.json",
        "fixed_target_calibration/student_idm_adapted.pt", "fixed_target_calibration/idm_training_resume.pt",
        "corpus_resume/dagger_resume_metrics.json", "corpus_resume/dagger_resume_verified.pt",
        "corpus_resume/dagger_resume_rng.pt", "g1_native_corpus16k.command.json",
        "g1_native_corpus16k.resources.json", "g1_native_corpus16k.log", "superseded_native_receipts.json",
        "fada_cpu_parity.xml", "fada_cuda_parity.xml"]
    artifacts = [evidence(output / name) for name in fixed_files]
    artifacts.extend(evidence(corpus / name) for name in
                     ("summary.json", "config_snapshot.json", "training_log.json", "model_final.pt"))
    artifacts.extend(evidence(Path(config[key])) for key in
                     ("offline_cache_path", "offline_suboptimal_cache_path", "warmup_ckpt_path"))
    artifacts.append(evidence(corpus / "checkpoints/online_buffer_latest.npz"))
    weak = warmup["suboptimal_offline_collect"]
    source_ready = source["failure_resets"] == 0
    target_ready = pre["failure_resets"] == post["failure_resets"] == 0
    report = {
        "schema": "genedynamics.fada.native_qualification.v1",
        "source_commit": "290b3a9c5c0ebe2e9851b61350617544be05e9bc",
        "simulation_adapted": True,
        "original_paper_quantitative_reproduction": {"qualified": False,
            "reason": "Official native G1 source and author oracle run on mjwarp; released training simulator IsaacSim and original paper scores are not reproduced."},
        "student_stability": {"qualified": source_ready and target_ready,
            "source_failure_resets": source["failure_resets"],
            "source_first_failure_seconds": source["first_failure_seconds"],
            "target_pre_failure_resets": pre["failure_resets"],
            "target_post_failure_resets": post["failure_resets"],
            "independent_reset_seed": source["reset_seed"], "evaluation_seconds_per_env": 10.0},
        "execution_gates": {
            "verified_public_author_oracle": True,
            "oracle_closed_loop_ppo_update_native_save_load_optimizer_normalizer_rng": True,
            "source_dagger_with_weak_data_strict_shadow_and_two_optimizers": summary["status"] == "completed",
            "source_next_update_resume_train_replay_optimizer_normalizer_rng": True,
            "target_public_records_single_fixed_robot120_seconds": True,
            "target_idm_lora_update_resume_optimizer_normalizer_rng": adaptation["optimizer_rng_normalization_next_update_exact"],
            "target_fixed_coefficient_hashes_and_matched_pre_post_commands": True,
            "actual_history_source_execution_parity": source["source_execution_parity"],
            "four_cross_ebdm_tasks": "qualified separately; native robot evidence does not qualify task adaptations"},
        "source_training": {
            "optimal_records": summary["optimal_offline_buffer_size"],
            "weak_records": summary["suboptimal_offline_buffer_size"],
            "online_train_records": summary["online_buffer_size"],
            "weak_pool_members": len(weak["selected_checkpoints"]),
            "weak_members_actually_sampled": sum(count > 0 for count in weak["checkpoint_steps"].values()),
            "training_updates": training[-1]["train_step_count"], "batch_size": config["batch_size"],
            "source_default_updates": 32000, "source_default_batch_size": 4096,
            "source_default_num_envs": 1024,
            "source_default_requested_optimal_records": 2560000,
            "source_default_requested_weak_records": 2560000,
            "source_default_data_formula": "5 warmup rollouts ×1024 environments ×500 control steps per source; weak ratio1.0",
            "data_budget_depends_on_parallel_env_count_unless_offline_steps_explicit": True,
            "sampled_update_budget_fraction_of_source_defaults": training[-1]["train_step_count"] * config["batch_size"] / (32000 * 4096),
            "all_source_randomizers_preserved": True,
            "warmup_validation": {key: warmup[key] for key in
                ("idm_validation_action_loss", "idm_validation_action_loss_unweighted", "planner_validation_future_obs_loss")},
            "last_dagger_validation": {key: training[-1][key] for key in
                ("idm_validation_action_loss", "idm_validation_action_loss_unweighted", "planner_validation_future_obs_loss")},
            "planner_obs_loss_coefficient": 0.0,
            "qualification": "Bounded training chain, far below source default sampled-update budget; no convergence claim."},
        "target_information_budget": {"canonical_calibration_action_seconds": 120.0,
            "superseded_calibration_action_seconds_retained": 120.0,
            "total_calibration_action_seconds_spent": 240.0,
            "includes_no_target_rewards_or_privileged_state_in_fitter": True,
            "adaptation_steps": adaptation["adaptation_steps"], "verification_updates": adaptation["verification_updates"],
            "batch_size": adaptation["batch_size"], "actual_batch_sizes": adaptation["actual_batch_sizes"],
            "drop_last": False, "selection": "final budgeted snapshot before resume verification updates"},
        "matched_target_results": {"pre_mean_reward": pre["mean_reward"], "post_mean_reward": post["mean_reward"],
            "command_schedule_sha256": pre["matched_velocity_schedule_sha256"],
            "reward_used_only_for_evaluation": True, "target_startup_draw_seed": 42,
            "reset_seed": 1042, "coefficient_hashes": target_hashes},
        "diagnosis": [
            "The original source and port produce exactly equal full action horizons, public observation scaling and commands on sampled actual histories.",
            "Native integer done indexing was fixed; affected earlier runner evaluations and B-log qualification are superseded and their data costs retained.",
            "Source held-out action losses and native closed-loop stability differ; teacher-forced reconstruction alone does not establish a usable source student.",
            "The bounded source run uses 1/256 of default sampled optimizer budget and samples only part of the reward-aware weak-checkpoint pool.",
            "The original source objective has zero explicit planner observation loss; future-observation MSE is a diagnostic, not a missing ported loss."],
        "remaining_gates": ["Stable source student under released data/update scale and independent native resets",
            "Stable matched target pre/post evaluation", "Original IsaacSim native quantitative reproduction",
            "Full Cross-EBDM task qualification under the task-specific protocol"],
        "resource_profile": read(output / "g1_native_corpus16k.resources.json"),
        "adaptation_resource_profile": {key: adaptation[key] for key in
            ("wall_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")},
        "scaling_cost": {
            "measured_bounded_source_seconds": read(output / "g1_native_corpus16k.resources.json")["wall_seconds"],
            "measured_single_target120s_collection_wall_seconds": collection["wall_seconds"],
            "measured_released100_step_target_adaptation_wall_seconds": adaptation["wall_seconds"],
            "release_default_collection_vs_requested16k_factor_per_source": 2560000 / 16384,
            "release_default_sampled_optimizer_budget_factor": (32000 * 4096) / (training[-1]["train_step_count"] * config["batch_size"]),
            "estimate": "At unchanged16-env collection throughput, full default data is roughly156× collection work. Learning work is256× sampled optimizer budget; wall time is not linearly extrapolated across batch sizes. Source cost includes collection, so multiplying its whole wall time would overstate measured collection-only cost.",
            "scalable_cli": "python -m scripts.baselines.fada_native dagger --profile release --num-envs 16 --offline-steps 2560000 --dagger-run-name g1_release_scale --max-seconds <explicit-budget> --max-gpu-mib <explicit-budget> --output <config-mirrored-leaf>"},
        "artifacts": artifacts,
    }
    destination = output / "native_qualification.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(destination), "student_stability": report["student_stability"],
                      "simulation_adapted": True, "paper_quantitative_reproduction": False}, indent=2))


if __name__ == "__main__":
    main()
