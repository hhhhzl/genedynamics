"""Evaluation-only checks for the official native robot qualification route.

Simulator parameters remain evaluator metadata and never enter the public target
record or the student. Source collection/training keeps every source randomizer.
"""

from __future__ import annotations

import hashlib

import torch


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reset_public_histories(done, history_obs, history_act, valid, previous_action):
    """Clear only completed environments, accepting native integer reset buffers."""
    mask = done.to(dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != history_obs.shape[0]:
        raise ValueError("Native done mask must have one entry per environment")
    for tensor in (history_obs, history_act, valid, previous_action):
        tensor[mask] = 0
    return mask


class FixedTargetDynamics:
    """Hold one sampled target's coefficients across independent episode resets.

    Source startup mass/COM/friction remain sampled normally. The target retains
    that draw and its first-reset PD gains and action delay; reset posture, push
    scheduling and stochastic torque disturbance remain active. No source preset
    is edited. Holding coefficients is an explicit evaluation protocol choice.
    """

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("A fixed target qualification uses exactly one environment")
        self.env = env
        actuator = env.randomization_manager.get_state("actuator_randomizer_state")
        self.tensors = {"kp_scale": actuator.kp_scale, "kd_scale": actuator.kd_scale,
                        "rfi_lim_scale": actuator.rfi_lim_scale,
                        "action_delay_idx": env.action_delay_idx,
                        "default_dof_pos": env.default_dof_pos}
        self.saved = {key: value.detach().clone() for key, value in self.tensors.items()}
        self.start = self.fingerprint()
        original_reset = env.randomization_manager.reset

        def reset_and_restore(env_ids):
            original_reset(env_ids)
            for key, value in self.tensors.items():
                value.copy_(self.saved[key])

        env.randomization_manager.reset = reset_and_restore

    def fingerprint(self):
        import warp as wp

        model = self.env.simulator.backend.mjw_model
        result = {key: tensor_digest(value) for key, value in self.tensors.items()}
        for name in ("body_mass", "body_inertia", "body_ipos", "geom_friction"):
            result[name] = tensor_digest(wp.to_torch(getattr(model, name)))
        return result

    def report(self):
        end = self.fingerprint()
        if self.start != end:
            raise RuntimeError("The fixed target dynamics changed during native qualification")
        return {"startup_draw_retained": True, "coefficient_hashes": end,
                "coefficient_hashes_unchanged": True,
                "still_randomized": ["reset posture", "push timing/velocity", "torque disturbance"],
                "evaluator_only": True}


class SourceExecutionParity:
    """Compare real native checkpoint inference on actual closed-loop histories."""

    def __init__(self, payload, device):
        from holosoma.fada.planner_idm.config import FADAConfig
        from holosoma.fada.planner_idm.model import build_student_policy

        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self.model = build_student_policy(FADAConfig(**payload["cfg"]),
                obs_dim=payload["obs_dim"], act_dim=payload["act_dim"], cmd_dim=payload["cmd_dim"]).to(device).eval()
        self.model.load_split_state_dict(planner_state_dict=payload["planner_state_dict"],
                                         idm_state_dict=payload["idm_state_dict"])
        self.payload = payload
        self.count = 0
        self.maximum_action_error = 0.0
        self.maximum_observation_error = 0.0
        self.maximum_command_error = 0.0

    def check(self, env, policy, obs, act, command, mask):
        from holosoma.fada.planner_idm.eval import _extract_compact_obs, _extract_current_command

        self.maximum_observation_error = max(self.maximum_observation_error, float((
            _extract_compact_obs(env, compact_obs_term_scale=self.payload["compact_obs"]["term_scale"]) - obs[:, -1]).abs().max()))
        self.maximum_command_error = max(self.maximum_command_error, float((
            _extract_current_command(env, command_profile=self.payload["cfg"]["command_profile"]) - command).abs().max()))
        normalized_obs, normalized_act, normalized_command = obs, act, command
        backend = policy.backend
        if backend.io_normalization:
            normalized_obs = (obs - backend.stats["obs"]["mean"]) / backend.stats["obs"]["std"]
            normalized_act = (act - backend.stats["act"]["mean"]) / backend.stats["act"]["std"]
            normalized_command = (command - backend.stats["cmd"]["mean"]) / backend.stats["cmd"]["std"]
        with torch.inference_mode():
            expected = self.model(normalized_obs, normalized_act, normalized_command, history_valid_mask=mask)
            if backend.io_normalization:
                expected = expected * backend.stats["act"]["std"] + backend.stats["act"]["mean"]
            actual = policy.predict(obs, act, command, history_valid_mask=mask)
        error = float((expected - actual).abs().max())
        self.maximum_action_error = max(self.maximum_action_error, error)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.count += 1

    def report(self):
        return {"actual_closed_loop_histories_checked": self.count,
                "maximum_action_abs_error": self.maximum_action_error,
                "maximum_compact_obs_abs_error": self.maximum_observation_error,
                "maximum_command_abs_error": self.maximum_command_error,
                "same_checkpoint_same_mha_mode_exact": True,
                "action_history": "previous clipped student action, right aligned, reset to zero at done",
                "normalization_enabled": bool(self.payload.get("normalization", {}).get("io_enabled", False)),
                "does_not_establish": "student convergence or simulator equivalence to IsaacSim"}
