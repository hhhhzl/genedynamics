"""Humanoid box push/unjam env (idea.txt Exp II, H1/H2/H4) — docker, real brax.

Validates the rebuilt contact-semantic primitive + π_low + full manifold:
  * fixed-stance action_size == 12 (contact primitive) and 15 (+base);
  * each level constructs, reset, step(random) finite;
  * fixed-stance constraint_residual ->
    h(14)=[h_box;h_hand;h_hand_R;h_foot_xyz] + g(3)=[g_bal;g_fric;g_tip];
  * the controller point equals the physical hand geom and reset has a positive approach gap;
  * manifold_geometry -> (Hnode+1, nu);
  * face selection (H4): different j logits -> different contact target;
    fixed rear face for H1;
  * H2 domain randomization (hand friction / box frictionloss) varies across seeds;
  * MGA ablations (no_softfeas/no_stiffness/no_tangent) change the plan;
  * the metrics plugin rolls + computes the full Exp II metric set end-to-end.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_humanoid_box_docker.py"
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import mujoco

from genedynamics.envs.factories import make_env
from genedynamics.experiments.plugins.environments._contact_task import HUMANOID_TASK
from genedynamics.experiments.plugins.methods.contact_receding import make_mga
from genedynamics.experiments.plugins.metrics.extractors import (
    humanoid_box_push_metrics_plugin,
)

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def test_h1_training_stratifies_domains_and_clears_terminal_task_memory():
    """Small array-only dynamics exercise the real Brax wrapper contract."""
    from brax.envs.base import State
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushDomainEnv
    from scripts.tasks.robot.humanoid.train_box_push_rl import wrap_h1_training
    import pytest

    class Domain:
        action_size = observation_size = 1
        backend = "generalized"
        dt = 0.02

        def reset(self, key):
            del key
            return State(
                pipeline_state=jnp.zeros(1), obs=jnp.zeros(1),
                reward=jnp.float32(0), done=jnp.float32(0), metrics={},
                info={"task_success": jnp.float32(0), "force_int": jnp.float32(0),
                      "task_fallen": jnp.float32(0), "success_padding": jnp.bool_(False),
                      "contact_acquired": jnp.float32(0), "step": jnp.int32(0),
                      "nested": {"memory": jnp.float32(0)}},
            )

        def step(self, state, action):
            frozen = state.info["task_success"] > 0
            # The last branch intentionally supplies simultaneous goal/fall
            # flags to verify that the training wrapper gives fall priority.
            success = ((action[0] > 0.5) | (action[0] < -1.5)).astype(jnp.float32)
            fallen = ((action[0] < -0.5) & ~frozen).astype(jnp.float32)
            value = state.pipeline_state + (~frozen).astype(jnp.float32)
            memory_step = (~frozen).astype(jnp.float32)
            goal = jnp.maximum(state.info["task_success"], success)
            return state.replace(
                pipeline_state=value, obs=value, reward=jnp.where(frozen, 2.0, 1.0),
                done=jnp.maximum(goal, fallen),
                info={**state.info, "task_success": goal,
                      "task_fallen": fallen, "success_padding": frozen,
                      "force_int": state.info["force_int"] + memory_step,
                      "contact_acquired": jnp.float32(1), "step": state.info["step"] + 1,
                      "nested": {"memory": state.info["nested"]["memory"] + memory_step}},
            )

    wrapped = wrap_h1_training(HumanoidBoxPushDomainEnv([Domain(), Domain()]), 3)
    with pytest.raises(ValueError, match="cover every training domain"):
        wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
    np.testing.assert_array_equal(state.info["_rl_domain_index"], [0, 1, 0, 1])
    step = jax.jit(wrapped.step)
    goal = step(state, jnp.ones((4, 1)))
    np.testing.assert_array_equal(goal.done, jnp.zeros(4))
    np.testing.assert_array_equal(goal.info["task_success"], jnp.ones(4))
    np.testing.assert_array_equal(goal.info["success_padding"], jnp.zeros(4, dtype=bool))
    np.testing.assert_array_equal(goal.metrics["h1_goal_completions"], jnp.ones(4))
    np.testing.assert_array_equal(goal.pipeline_state, jnp.ones((4, 1)))
    padding = step(goal, jnp.zeros((4, 1)))
    np.testing.assert_array_equal(padding.done, jnp.zeros(4))
    np.testing.assert_array_equal(padding.pipeline_state, goal.pipeline_state)
    np.testing.assert_array_equal(padding.reward, jnp.full(4, 2.0))
    np.testing.assert_array_equal(padding.info["force_int"], goal.info["force_int"])
    np.testing.assert_array_equal(padding.info["nested"]["memory"], goal.info["nested"]["memory"])
    np.testing.assert_array_equal(padding.metrics["h1_active_steps"], jnp.zeros(4))
    np.testing.assert_array_equal(padding.metrics["h1_success_padding_steps"], jnp.ones(4))
    terminal = step(padding, jnp.zeros((4, 1)))
    np.testing.assert_array_equal(terminal.done, jnp.ones(4))
    np.testing.assert_array_equal(terminal.info["truncation"], jnp.ones(4))
    np.testing.assert_array_equal(terminal.pipeline_state, jnp.zeros((4, 1)))
    for key, expected in (("h1_active_steps", 1), ("h1_success_padding_steps", 2),
                          ("h1_goal_completions", 1), ("h1_failures", 0)):
        np.testing.assert_array_equal(terminal.info["episode_metrics"][key], jnp.full(4, expected))
    for index in (0, 1):
        np.testing.assert_array_equal(
            terminal.info["episode_metrics"][f"h1_domain_{index}_steps"],
            (jnp.arange(4) % 2 == index).astype(jnp.float32),
        )

    restarted = step(terminal, jnp.zeros((4, 1)))
    np.testing.assert_array_equal(restarted.pipeline_state, jnp.ones((4, 1)))
    np.testing.assert_array_equal(restarted.info["task_success"], jnp.zeros(4))
    np.testing.assert_array_equal(restarted.info["task_fallen"], jnp.zeros(4))
    np.testing.assert_array_equal(restarted.info["success_padding"], jnp.zeros(4, dtype=bool))
    np.testing.assert_array_equal(restarted.info["force_int"], jnp.ones(4))
    np.testing.assert_array_equal(restarted.info["nested"]["memory"], jnp.ones(4))
    np.testing.assert_array_equal(restarted.info["step"], jnp.ones(4))
    timed_out = step(step(restarted, jnp.zeros((4, 1))), jnp.zeros((4, 1)))
    np.testing.assert_array_equal(timed_out.info["truncation"], jnp.ones(4))
    after_timeout = step(timed_out, jnp.zeros((4, 1)))
    np.testing.assert_array_equal(after_timeout.info["step"], jnp.ones(4))
    for index in (0, 1):
        assert float(jnp.sum(timed_out.info["episode_metrics"][f"h1_domain_{index}_steps"])) > 0

    mixed = step(state, jnp.asarray([[-1.0], [1.0], [0.0], [-2.0]]))
    np.testing.assert_array_equal(mixed.done, [1, 0, 0, 1])
    np.testing.assert_array_equal(mixed.info["truncation"], [0, 0, 0, 0])
    np.testing.assert_array_equal(mixed.metrics["h1_failures"], [1, 0, 0, 1])
    np.testing.assert_array_equal(mixed.metrics["h1_goal_completions"], [0, 1, 0, 0])
    np.testing.assert_array_equal(mixed.pipeline_state[:, 0], [0, 1, 1, 0])
    next_step = step(mixed, jnp.zeros((4, 1)))
    np.testing.assert_array_equal(next_step.info["step"], [1, 2, 2, 1])
    np.testing.assert_array_equal(next_step.info["task_success"], [0, 1, 0, 0])
    np.testing.assert_array_equal(next_step.info["task_fallen"], [0, 0, 0, 0])
    np.testing.assert_array_equal(next_step.info["success_padding"], [False, True, False, False])
    np.testing.assert_array_equal(next_step.info["nested"]["memory"], [1, 1, 2, 1])

    class OldDomain(Domain):
        def reset(self, key):
            initial = super().reset(key)
            return initial.replace(info={k: v for k, v in initial.info.items()
                                         if k != "success_padding"})

    old = wrap_h1_training(HumanoidBoxPushDomainEnv([OldDomain()]), 3)
    with pytest.raises(ValueError, match="termination contract missing"):
        old.reset(jax.random.split(jax.random.PRNGKey(0), 1))


def test_h1_training_refuses_missing_domain_coverage_and_divergent_export():
    import pytest
    from typing import NamedTuple
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _H1_TERMINATION_CONTRACT, _check_training_metrics, _validate_stage_for_export,
    )

    class Normalizer(NamedTuple):
        std: object

    params = (Normalizer(jnp.ones(2) * 0.05), {"weight": jnp.ones(1)})
    config = {"final_metrics": {"training/kl_mean": 0.02}}
    history = [{"stage": 0, "episode/h1_domain_0_steps": 5.0,
                "episode/h1_domain_1_steps": 4.0, "episode/h1_active_steps": 9.0,
                "episode/h1_success_padding_steps": 11.0,
                "episode/h1_goal_completions": 2.0, "episode/h1_failures": 1.0}]
    audit = _validate_stage_for_export(params, config, history, 0, 2, 0.05, 10)
    assert audit["stage"] == 0
    assert audit["domain_coverage_unit"] == "active_transitions_excluding_success_padding"
    assert audit["max_logged_episode_mean_transition_counts"]["h1_success_padding_steps"] == 11.0
    assert _H1_TERMINATION_CONTRACT["objective"] == "continuing_discounted_success_absorbing"
    assert _H1_TERMINATION_CONTRACT["time_limit"] == "truncation_with_existing_brax_bootstrap"
    assert _H1_TERMINATION_CONTRACT["environment_step_budget"] == "includes_success_padding"
    with pytest.raises(RuntimeError, match="no observed episode coverage"):
        _validate_stage_for_export(params, config, [{
            **history[0], "episode/h1_domain_1_steps": 0.0,
            "episode/h1_success_padding_steps": 100.0,
        }], 0, 2, 0.05, 10)
    with pytest.raises(RuntimeError, match="no observed episode coverage"):
        _validate_stage_for_export(params, config, history, 0, 3, 0.05, 10)
    with pytest.raises(RuntimeError, match="std floor"):
        _validate_stage_for_export((Normalizer(jnp.ones(2) * 1e-6),), config,
                                   history, 0, 2, 0.05, 10)
    unnormalized = _validate_stage_for_export(
        (Normalizer(jnp.ones(2) * 1e-6),),
        {**config, "normalize_observations": False}, history, 0, 2, 0.05, 10,
    )
    assert unnormalized["policy_observation_normalization"] == "disabled"
    assert unnormalized["policy_normalizer_floor_checked"] is False
    with pytest.raises(RuntimeError, match="divergent H1 training"):
        _check_training_metrics({"training/kl_mean": 4.36e12}, 10)
    with pytest.raises(RuntimeError, match="divergent H1 training"):
        _check_training_metrics({"training/v_loss": float("nan")}, 10)
    with pytest.raises(RuntimeError, match="no final PPO KL"):
        _validate_stage_for_export(params, {}, history, 0, 2, 0.05, 10)


def test_h1_training_domains_and_durations_follow_canonical_suites():
    import copy
    import pytest
    from genedynamics.experiments.framework.config import ExperimentConfig
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        CANONICAL_CONFIG, SCHEMA_SUITES, _domain_specs,
        _stage_episode_lengths, _training_suites,
    )

    config = ExperimentConfig.from_yaml(CANONICAL_CONFIG)
    available = {suite["name"]: suite for suite in config.suites}
    p2 = config.for_suite(available["p2_push_ood"])
    assert tuple(p2.env_params["h2_size_range"]) == (
        p2.env_params.get("box_half", 0.55),
        p2.env_params.get("box_half", 0.55),
    )
    assert tuple(p2.env_params["h2_boxfric_range"]) == (20.0, 25.0)
    assert p2.env_params["h2_boxfric_range"][0] > p2.env_params.get(
        "box_frictionloss", 15.0
    )
    assert p2.env_params["f_target"] - p2.env_params[
        "h2_boxfric_range"
    ][1] >= 10.0
    for schema, names in SCHEMA_SUITES.items():
        specs = _domain_specs(schema, config)
        for name, spec in zip(names, specs):
            expected = dict(config.for_suite(available[name]).env_params)
            if name == "p2_push_ood":
                expected["dr_seed"] = 101
            assert spec == expected

    updated = copy.deepcopy(config)
    walk = next(suite for suite in updated.suites if suite["name"] == "p4_walk_push")
    walk["n_steps"] = 300
    walk["env_params"]["push_dist"] = 0.5
    resolved = _training_suites("walk", updated)
    assert _stage_episode_lengths("walk", resolved) == [300]
    assert _stage_episode_lengths("walk", resolved, 50) == [50]
    assert _domain_specs("walk", updated)[0]["push_dist"] == 0.5
    assert _stage_episode_lengths("fixed", _training_suites("fixed", updated)) == [100, 100]
    updated.suites[0]["n_steps"] = 101
    with pytest.raises(ValueError, match="same episode length"):
        _stage_episode_lengths("fixed", _training_suites("fixed", updated))


def test_h1_training_seed_guard_follows_protocol_instead_of_historical_split():
    from types import SimpleNamespace
    import pytest
    from scripts.tasks.robot.humanoid.train_box_push_rl import _validate_training_seed

    config = SimpleNamespace(seeds=list(range(10)))
    for seed in (0, 9):
        with pytest.raises(ValueError, match="canonical formal evaluation seeds"):
            _validate_training_seed(seed, config)
    for seed in (10, 101, 110, 111):
        _validate_training_seed(seed, config)
    # A future frozen protocol can change the split without changing Python.
    with pytest.raises(ValueError, match="canonical formal evaluation seeds"):
        _validate_training_seed(101, SimpleNamespace(seeds=[101, 102]))


def test_h1_training_audit_preserves_roundup_steps_and_failure(tmp_path):
    import json
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _observed_training_steps, _write_training_status,
    )

    history = [
        {"stage": 0, "num_steps": 1000}, {"stage": 0, "num_steps": 2080},
        {"stage": 1, "num_steps": 0}, {"stage": 1, "num_steps": 1000,
         "training/v_loss": "nan"},
    ]
    counts = _observed_training_steps(history)
    assert counts == {"0": 2080, "1": 1000}
    path = tmp_path / "_training" / "fixed_smoke" / "training_status.json"
    payload = {"status": "failed", "failure": "RuntimeError: divergent training",
               "requested_environment_steps": 4096,
               "observed_stage_steps": counts, "progress_history": history,
               "performance_validated": False}
    _write_training_status(path, payload)
    assert json.loads(path.read_text()) == payload
    assert not path.with_suffix(".json.tmp").exists()


def test_h1_training_source_audit_covers_contact_gait_and_atacom():
    import hashlib
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        CANONICAL_CONFIG, _training_source_hashes,
    )

    root = CANONICAL_CONFIG.parents[3]
    fixed = _training_source_hashes(root)
    atacom = _training_source_hashes(root, atacom=True)
    for name in ("genedynamics/envs/domains/humanoid/box_push_brax.py",
                 "genedynamics/core/control/humanoid_contact.py",
                 "genedynamics/core/control/bipedal_gait.py"):
        assert fixed[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert set(fixed).issubset(atacom)
    assert "genedynamics/solvers/single/atacom/wrapper.py" in atacom


def test_h1_ppo_kwargs_keep_behavior_normalization_consistent_and_kl_observable():
    from scripts.tasks.robot.humanoid.train_box_push_rl import (
        _ppo_training_kwargs, wrap_h1_training,
    )

    for schema, atacom in (("fixed", False), ("atacom_p12", True), ("walk", False)):
        for low_memory in (False, True):
            for num_envs in (4, 8):
                kwargs = _ppo_training_kwargs(
                    schema=schema, atacom=atacom, num_envs=num_envs,
                    cpu_low_memory=low_memory,
                    normalizer_std_eps=0.05,
                )
                assert kwargs["learning_rate_schedule"] == "NONE"
                assert kwargs["max_grad_norm"] == 1.0
                assert kwargs["num_updates_per_batch"] == 2
                assert kwargs["wrap_env_fn"] is wrap_h1_training
                assert kwargs["normalize_observations_std_eps"] == 0.05
                assert kwargs["reward_scaling"] == (1e-4 if atacom else 0.01)
                assert "value_hidden_layer_sizes" not in kwargs
                assert kwargs["learning_rate"] == (2e-5 if schema == "walk" else 1e-4)
                if schema == "walk":
                    assert kwargs["init_noise_std"] == 0.25
                if low_memory:
                    assert kwargs["unroll_length"] == 10
                    assert kwargs["discounting"] == 0.99
                    assert kwargs["normalize_observations"] is False
                    assert kwargs["num_minibatches"] == 1
                    assert kwargs["batch_size"] == num_envs
                    assert kwargs["policy_hidden_layer_sizes"] == (16, 16)
                elif schema == "walk":
                    assert kwargs["unroll_length"] == 20
                    assert kwargs["discounting"] == 0.995
                    assert kwargs["normalize_observations"] is True
                    assert kwargs["policy_hidden_layer_sizes"] == (64, 64)
                else:
                    assert kwargs["learning_rate"] == 1e-4
                    assert kwargs["unroll_length"] == 10
                    assert kwargs["discounting"] == 0.99
                    assert kwargs["normalize_observations"] is False


def _check_level(level, use_base=False, want_nu=12, want_h=14):
    env = make_env(HUMANOID_TASK, level=level, use_base=use_base, dr_seed=1)
    x0 = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_allclose(x0.obs, env._get_obs(x0.pipeline_state, x0.info))
    task_start = x0.pipeline_state.qpos.size + x0.pipeline_state.qvel.size
    np.testing.assert_allclose(x0.obs[task_start + 4], env._bcfg.push_dist, atol=1e-6)
    a = jax.random.uniform(jax.random.PRNGKey(1), (env.action_size,), minval=-1.0, maxval=1.0)
    s1 = env.step(x0, a)
    finite = bool(jnp.all(jnp.isfinite(s1.obs)) and jnp.isfinite(s1.reward))
    h, g = env.constraint_residual(x0, a)
    ag = env.manifold_geometry(x0, jnp.zeros((CFG["Hnode"] + 1, env.action_size)), 0.0)
    ok = (env.action_size == want_nu and finite and h.shape == (want_h,) and g.shape == (3,)
          and ag.shape == (CFG["Hnode"] + 1, want_nu))
    tag = f"{level}{'(+base)' if use_base else ''}"
    print(f"  [{tag:9s}] action={env.action_size} finite={finite} h={tuple(h.shape)} "
          f"g={tuple(g.shape)} geom={tuple(ag.shape)} -> {'OK' if ok else 'FAIL'}")
    return ok


def _face_selection():
    env = make_env(HUMANOID_TASK, level="unjam")
    box = jnp.array([1.0, 0.0, 0.6])
    # rear-dominant vs left-dominant j logits -> different contact target
    def target(jlog):
        a = jnp.zeros(env.action_size).at[0:3].set(jlog)
        v, w, aa, bb, K, Fn = env._unpack(a)
        return np.asarray(env._contact_target(box, jnp.array([1., 0., 0., 0.]), w, aa, bb)[0])
    rear = target(jnp.array([10., 0., 0.]))
    left = target(jnp.array([0., 10., 0.]))
    d = float(np.max(np.abs(rear - left)))
    print(f"  face select: rear-target={np.round(rear,2)} left-target={np.round(left,2)} Δ={d:.3f}")
    # H1 ignores j (fixed rear)
    envh1 = make_env(HUMANOID_TASK, level="double_support")
    _, w1, _, _, _, _ = envh1._unpack(jnp.zeros(envh1.action_size).at[0:3].set(jnp.array([0., 10., 0.])))
    fixed = bool(np.allclose(np.asarray(w1), [1., 0., 0.]))
    print(f"  double_support fixed rear (j ignored): w_face={np.round(np.asarray(w1),2)} -> {fixed}")
    return d > 0.1 and fixed


def _h2_dr_varies():
    e1 = make_env(HUMANOID_TASK, level="heavy_dr", dr_seed=1)
    e2 = make_env(HUMANOID_TASK, level="heavy_dr", dr_seed=2)
    bf1, bf2 = float(e1.sys.dof_frictionloss[-3]), float(e2.sys.dof_frictionloss[-3])
    d = abs(float(e1._mu) - float(e2._mu)) > 1e-3 or abs(bf1 - bf2) > 1.0
    def target_inside_face(env):
        s = env.reset(jax.random.PRNGKey(0))
        c = env._hand_contact(s.pipeline_state, jnp.zeros(env.action_size), s.info)
        box_z = s.pipeline_state.x.pos[env._box_idx - 1, 2]
        return abs(float(c["p_surface"][2] - box_z)) <= float(env._half) + 1e-6
    targets_valid = target_inside_face(e1) and target_inside_face(e2)
    print(f"  H2 DR: seed1(mu={float(e1._mu):.2f},boxfric={bf1:.0f}) vs "
          f"seed2(mu={float(e2._mu):.2f},boxfric={bf2:.0f}) -> differs={d}, "
          f"targets-inside-face={targets_valid}")
    return d and targets_valid


def _physical_contact_geometry():
    env = make_env(HUMANOID_TASK, level="push_to_line", stiffness_mode="none")
    unjam = make_env(HUMANOID_TASK, level="unjam", stiffness_mode="none")
    x0 = env.reset(jax.random.PRNGKey(7))
    ps = x0.pipeline_state
    action = jnp.zeros(env.action_size).at[env.spec.nu_slice].set(-1.0)  # zero desired force
    contact = env._hand_contact(ps, action)
    controller_matches_geom = bool(jnp.allclose(
        contact["p_hand"], ps.geom_xpos[env._rhand_geom], atol=1e-6))
    hand_front = max(float(ps.geom_xpos[env._rhand_geom, 0]),
                     float(ps.geom_xpos[env._lhand_geom, 0])) + float(
                         env._robot_profile.controller_defaults[
                             "hand_forward_extent"
                         ]
                     )
    rear_face = float(ps.x.pos[env._box_idx - 1, 0] - env._half)
    gap = rear_face - hand_front
    forces = env._box_contact_forces(ps)
    no_reset_contact = float(forces["hand"]) < 1e-6 and float(forces["nonhand"]) < 1e-6
    p1 = make_env(
        HUMANOID_TASK, level="push_to_line", fixed_contact_target=True,
        fixed_force_target=True,
    )
    p1s = p1.reset(jax.random.PRNGKey(7))
    c_lo = p1._hand_contact(
        p1s.pipeline_state, jnp.zeros(p1.action_size).at[3:5].set(-1.0),
    )
    c_hi = p1._hand_contact(
        p1s.pipeline_state, jnp.zeros(p1.action_size).at[3:5].set(1.0),
    )
    p1_contact_locked = bool(jnp.allclose(c_lo["p_c"], c_hi["p_c"], atol=1e-6))
    mj = env.sys.mj_model
    lateral_locked = True
    for name in ("box_y", "box_yaw"):
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT.value, name)
        lateral_locked &= bool(env.sys.jnt_limited[jid]) and float(env.sys.jnt_range[jid, 1]) <= 1e-6
    walls_hidden = bool(
        jnp.all(env.sys.geom_rgba[env._wall_geoms, 3] == 0.0)
        and jnp.all(jnp.abs(env.sys.geom_pos[env._wall_geoms, 1]) >= 49.0)
        and np.all(env.sys.mj_model.geom_rgba[np.asarray(env._wall_geoms), 3] == 0.0)
        and np.all(np.abs(env.sys.mj_model.geom_pos[np.asarray(env._wall_geoms), 1]) >= 49.0)
    )
    walls_p3_only = bool(
        jnp.all(unjam.sys.geom_rgba[unjam._wall_geoms, 3] > 0.0)
        and jnp.all(jnp.abs(unjam.sys.geom_pos[unjam._wall_geoms, 1]) < 1.0)
        and np.all(unjam.sys.mj_model.geom_rgba[np.asarray(unjam._wall_geoms), 3] > 0.0)
        and np.all(np.abs(unjam.sys.mj_model.geom_pos[np.asarray(unjam._wall_geoms), 1]) < 1.0)
    )
    goal_site = mujoco.mj_name2id(
        mj, mujoco.mjtObj.mjOBJ_SITE.value, "goal_line"
    )
    box_joint = mujoco.mj_name2id(
        mj, mujoco.mjtObj.mjOBJ_JOINT.value, "box_x"
    )
    goal_x = (
        float(env._init_q[int(mj.jnt_qposadr[box_joint])])
        + env._bcfg.push_dist + float(env._half)
    )
    goal_line_synced = bool(
        np.isclose(float(env.sys.site_pos[goal_site, 0]), goal_x)
        and np.isclose(float(mj.site_pos[goal_site, 0]), goal_x)
        and float(mj.site_rgba[goal_site, 3]) > 0.0
    )
    p1_goal_site = mujoco.mj_name2id(
        p1.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "goal_line"
    )
    p3_goal_site = mujoco.mj_name2id(
        unjam.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "goal_line"
    )
    irrelevant_goal_lines_hidden = bool(
        float(p1.sys.mj_model.site_rgba[p1_goal_site, 3]) == 0.0
        and float(unjam.sys.mj_model.site_rgba[p3_goal_site, 3]) == 0.0
    )
    ok = controller_matches_geom and abs(gap - env._bcfg.approach_gap) < 1e-5 \
        and no_reset_contact and lateral_locked and p1_contact_locked \
        and walls_hidden and walls_p3_only and goal_line_synced \
        and irrelevant_goal_lines_hidden
    print(f"  contact geometry: point=geom {controller_matches_geom}, gap={gap:.4f} m, "
          f"reset_force=({float(forces['hand']):.3g},{float(forces['nonhand']):.3g}), "
          f"L1-y/yaw-locked={lateral_locked}, P1-contact-locked={p1_contact_locked} "
          f"walls-hidden={walls_hidden}, walls-P3-only={walls_p3_only}, "
          f"goal-line-synced={goal_line_synced}, "
          f"irrelevant-goal-lines-hidden={irrelevant_goal_lines_hidden} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def _robot_swap_contract():
    results = []
    for robot, joints in (("h1", 19), ("g1", 29)):
        env = make_env(HUMANOID_TASK, robot=robot, level="push_to_line")
        state = env.reset(jax.random.PRNGKey(17))
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        h, g = env.constraint_residual(state, action)
        valid = (
            env._robot_profile.num_actuated == joints
            and len(env._robot_binding.actuator_indices) == joints
            and env.action_size == 12
            and h.shape == (14,)
            and g.shape == (3,)
            and bool(jnp.isfinite(next_state.reward))
        )
        print(
            f"  robot swap {robot}: joints={joints} action={env.action_size} "
            f"h={h.shape} g={g.shape} finite={valid}"
        )
        results.append(valid)
    return all(results)


def _plan(env, sol, rng):
    b = sol._get_backend_impl()
    x0 = env.reset(jax.random.PRNGKey(1))
    return np.asarray(b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng))


def _ablations_active():
    rng = jax.random.PRNGKey(0)
    env0, full = make_mga(HUMANOID_TASK, "mga_base", level="unjam", **CFG)
    base = _plan(env0, full, rng)
    ok = True
    for m in ("mga_no_softfeas", "mga_no_stiffness", "mga_no_tangent"):
        e, s = make_mga(HUMANOID_TASK, m, level="unjam", **CFG)
        d = float(np.max(np.abs(_plan(e, s, rng) - base)))
        if m == "mga_no_tangent":
            # The zero plan is exactly on the clean stiffness/force manifold, so the final
            # stochastic plan delta can legitimately be tiny.  Test the actual operator on a
            # controlled off-manifold direction instead of classifying float32 noise as a no-op.
            fb = full._get_backend_impl()
            x0 = env0.reset(jax.random.PRNGKey(9))
            nodes = jnp.linspace(
                -0.5, 0.5, (CFG["Hnode"] + 1) * env0.action_size,
            ).reshape(CFG["Hnode"] + 1, env0.action_size)
            a_geom = full.geometry_fn(x0, nodes, 0.0)
            bundle = fb.manifold.geometry(a_geom, fb.topk_active, fb.eps_stab)
            projected = fb.manifold.project(a_geom, bundle, mode="metric")
            op_delta = float(jnp.max(jnp.abs(projected - a_geom)))
            active = (full.geometry_fn is not None and s.geometry_fn is None
                      and float(jnp.linalg.norm(a_geom)) > 1e-4 and op_delta > 1e-4)
            print(f"  mga vs {m}: max|Δplan|={d:.3e}, "
                  f"controlled projection Δ={op_delta:.3e} -> "
                  f"{'ACTIVE' if active else 'NO-OP!'}")
        else:
            active = d > 1e-4
            print(f"  mga vs {m}: max|Δplan|={d:.3e} -> "
                  f"{'ACTIVE' if active else 'NO-OP!'}")
        ok &= active
    return ok


def _metric_plugin_end_to_end():
    import types
    env = make_env(HUMANOID_TASK, level="unjam")
    x0 = env.reset(jax.random.PRNGKey(3))
    acts = [np.asarray(jax.random.uniform(jax.random.PRNGKey(i), (env.action_size,),
            minval=-1.0, maxval=1.0)) for i in range(5)]
    traj = types.SimpleNamespace(actions=acts)
    rec = humanoid_box_push_metrics_plugin().compute(
        traj, env, None, None, x0=x0, planning_time=0.3
    )
    # Five random steps end before the force ramp reaches its tracked window.
    # Steady-state/settling metrics are therefore unavailable by definition;
    # every metric that is observable on this short trajectory must stay finite.
    may_be_unavailable = {
        "steady_force_tracking_error",
        "steady_force_tracking_mae",
        "steady_normalized_force_tracking_mae",
        "force_settling_time",
    }
    nonfinite = {k for k, v in rec.items() if not np.isfinite(v)}
    valid = nonfinite <= may_be_unavailable and all(
        np.isnan(rec[k]) for k in nonfinite
    )
    print(f"  metric plugin: {len(rec)} metrics, nonfinite={sorted(nonfinite)}, "
          f"expected-only={valid}")
    print("    " + ", ".join(f"{k}={v:.3g}" for k, v in list(rec.items())[:6]))
    return len(rec) >= 12 and valid


def _h1_clean_manifold_array_env(level="unjam", *, use_base=False):
    """Actual clean-chart methods, without constructing physics or a renderer."""
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    env = object.__new__(HumanoidBoxPushEnv)
    env._bcfg = HumanoidBoxPushConfig(level=level, use_base=use_base, f_target=45.0)
    env._is_walk = level == "push_walk"
    env._face_select = level == "unjam"
    env._n_planner = 11
    env.spec = PrimitiveSpec(pos_dim=8 if use_base else 5, stiff_dim=3, feed_dim=1)
    return env



def _h1_contact_target_array_fixture():
    """Real target/observation methods; no contact solve or physics step."""
    from types import SimpleNamespace

    env = _h1_clean_manifold_array_env()
    env._half = jnp.float32(0.55)
    env._mu = jnp.float32(0.6)
    env._walk_requires_locomotion = False
    env._pelvis_idx, env._box_idx = 1, 2
    env._feet_site_id = jnp.asarray([0, 1])
    env._rhand_body, env._lhand_body = 1, 2
    env._rhand_geom, env._lhand_geom = 1, 2
    env._rhand_home = env._lhand_home = jnp.zeros(3)
    env._robot_profile = SimpleNamespace(
        controller_defaults={"hand_forward_extent": 0.033},
    )
    env.sys = SimpleNamespace(opt=SimpleNamespace(timestep=0.004))
    env._n_frames = 5
    env._stiffness = lambda raw: jnp.eye(3)
    env._box_contact_force = lambda ps: jnp.float32(0.)
    env._box_contact_forces = lambda ps: dict(
        hand=jnp.float32(0.), wall=jnp.float32(0.), nonhand=jnp.float32(0.),
    )
    env._one_hand = lambda ps, body, geom, p, n, K, F: {"p_c": p}
    ps = SimpleNamespace(
        x=SimpleNamespace(
            pos=jnp.asarray([[0., 0., 1.], [0.7, -0.1, 0.55]]),
            rot=jnp.tile(jnp.asarray([1., 0., 0., 0.]), (2, 1)),
        ),
        qpos=jnp.zeros(29), qvel=jnp.zeros(28),
        site_xpos=jnp.asarray([[0., 0.1, 0.], [0., -0.1, 0.]]),
    )
    return env, ps


def test_h1_p3_rear_two_hand_chart_respects_shared_face_span():
    from brax import math as brax_math
    from genedynamics.solvers.single.atacom.backends.atacom_jax import atacom_null_dim

    env, ps = _h1_contact_target_array_fixture()
    # A completed approach: interpolated free-space targets are not required
    # to lie on the box face before the approach has finished.
    info = {"step": jnp.int32(100), "contact_step": jnp.int32(100),
            "contact_acquired": jnp.float32(0.), "force_int": jnp.float32(0.)}
    assert env.action_size == 12
    assert env._get_obs(ps, info).shape == (76,)
    assert env.policy_interface is None
    assert env.manifold_constraint_size == 10 and atacom_null_dim(env) == 2
    assert env._manifold_res_node(jnp.zeros(12)).shape == (10,)
    for half in (0.48, 0.55):
        env._half = jnp.float32(half)
        # Use actual _half rather than the intentionally unchanged cfg.box_half.
        span = float(env._half) - 0.21 - env._bcfg.contact_edge_margin
        for offset in (-0.21, 0.21):
            env._bcfg.hand_offset = offset
            for yaw in (-0.4, 0.0, 0.4):
                quat = jnp.asarray([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
                ps.x.rot = ps.x.rot.at[1].set(quat)
                inverse = quat * jnp.asarray([1., -1., -1., -1.])
                for raw_a in (-1., -0.75, -0.5, 0., 0.25, 0.5, 1.):
                    action = jnp.zeros(12).at[3].set(raw_a).at[4].set(-raw_a)
                    _, weights, aa, _, _, _ = env._unpack(action)
                    np.testing.assert_allclose(weights, [1., 0., 0.], atol=5e-9)
                    shifted = np.clip(raw_a + env._bcfg.unjam_contact_a_bias, -1., 1.)
                    np.testing.assert_allclose(aa, 0.5 * (shifted + 1.), atol=1e-7)
                    contact = env._hand_contact(ps, action, info)
                    np.testing.assert_allclose(contact["approach_alpha"], 1.)
                    surface_points = []
                    for hand in (contact, contact["left"]):
                        surface = (hand["p_c"] - 0.033 * contact["n_c"]
                                   - ps.x.pos[1])
                        surface_points.append(brax_math.rotate(surface, inverse))
                    points = np.asarray(surface_points)
                    np.testing.assert_allclose(points[:, 0], -float(env._half), atol=2e-7)
                    limit = float(env._half) - env._bcfg.contact_edge_margin
                    assert np.max(np.abs(points[:, 1:])) <= limit + 2e-7
                    np.testing.assert_allclose(points[:, 1].mean(), shifted * span, atol=2e-7)
                    np.testing.assert_allclose(points[1] - points[0], [0., 2. * offset, 0.],
                                               atol=2e-7)
        # Preserve a continuous common coordinate, not independent hand clips.
        def centre(raw):
            action = jnp.zeros(12).at[3].set(raw)
            _, weights, aa, bb, _, _ = env._unpack(action)
            return env._contact_target(
                ps.x.pos[1], jnp.asarray([1., 0., 0., 0.]), weights, aa, bb,
            )[0][1]
        np.testing.assert_allclose(jax.grad(centre)(jnp.float32(-0.25)), span, atol=1e-7)
        np.testing.assert_allclose(jax.grad(centre)(jnp.float32(0.75)), 0., atol=1e-7)


def test_h1_other_task_contact_targets_retain_legacy_arithmetic():
    from brax import math as brax_math

    env, ps = _h1_contact_target_array_fixture()
    def legacy_target(box, quat, weights, aa, bb):
        cfg, half = env._bcfg, env._half
        ax = (2.0 * aa - 1.0) * half
        bz_raw = (cfg.hand_push_height - box[2]) + (2.0 * bb - 1.0) * cfg.contact_band
        z_lim = jnp.maximum(half - cfg.contact_edge_margin, 0.0)
        bz = jnp.clip(bz_raw, -z_lim, z_lim)
        offsets = jnp.stack([jnp.array([-half, ax, bz]),
                             jnp.array([ax, half, bz]), jnp.array([ax, -half, bz])])
        normals = jnp.asarray([[-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        point = box + brax_math.rotate(weights @ offsets, quat)
        normal = brax_math.rotate(weights @ normals, quat)
        return point, normal / (jnp.linalg.norm(normal) + 1e-9)
    for robot, level in (("h1", "push_to_line"), ("h1", "heavy_dr"),
                          ("h1", "push_walk"), ("g1", "unjam")):
        env._bcfg.robot, env._bcfg.level = robot, level
        for half in (0.48, 0.55):
            env._half = jnp.float32(half)
            for yaw in (0., 0.4):
                quat = jnp.asarray([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
                for weights in jnp.asarray([[1., 0., 0.], [0., 1., 0.],
                                            [0., 0., 1.], [0.6, 0.2, 0.2]]):
                    for aa in (0., 0.5, 1.):
                        expected = legacy_target(ps.x.pos[1], quat, weights, aa, 1. - aa)
                        actual = env._contact_target(ps.x.pos[1], quat, weights, aa, 1. - aa)
                        for new, old in zip(actual, expected):
                            np.testing.assert_array_equal(new, old)


def test_h1_p3_rear_two_hand_span_rejects_invalid_geometry():
    import pytest
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    # Constructor-only checks: no reset, controller rollout or MJX step.
    for overrides in (
        {"box_half": 0.20}, {"box_half": 0.25},
        {"box_half": float("nan")}, {"box_half": float("inf")},
        {"hand_offset": float("nan")}, {"hand_offset": float("inf")},
        {"contact_edge_margin": float("nan")}, {"contact_edge_margin": float("inf")},
    ):
        with pytest.raises(ValueError, match="finite positive two-hand contact span"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(level="unjam", **overrides))


def test_h1_p3_clean_chart_retains_both_contact_coordinates_across_release():
    from types import SimpleNamespace
    from genedynamics.solvers.single.mga.core.retraction import make_mga_cfs_filter

    for use_base in (False, True):
        env = _h1_clean_manifold_array_env(use_base=use_base)
        contact_start = 6 if use_base else 3
        free = slice(contact_start, contact_start + 2)
        nodes = jnp.tile(jnp.linspace(-0.4, 0.4, env.action_size), (3, 1))
        nodes = nodes.at[:, free].set(jnp.asarray([[-0.8, 0.3], [0.4, -0.6], [0.7, 0.8]]))
        jacobian = jax.jacfwd(env._manifold_res_node)(nodes[0])
        assert env.manifold_constraint_size == 10
        assert jacobian.shape == (10, env.action_size)
        assert np.linalg.matrix_rank(np.asarray(jacobian), tol=1e-6) == 10
        np.testing.assert_array_equal(jacobian[:, free], jnp.zeros((10, 2)))
        filt = make_mga_cfs_filter(env)
        previous = None
        for released in (0.0, 1.0):
            state = SimpleNamespace(info={"unjam_released": jnp.float32(released)})
            residual = env.manifold_residual(state, nodes)
            assert residual.shape == (30,)
            np.testing.assert_array_equal(
                env.manifold_residual_horizon(state, nodes, 0.0), residual,
            )
            np.testing.assert_array_equal(
                env.manifold_residual_horizon_controllable(state, nodes, 0.0), residual,
            )
            np.testing.assert_array_equal(
                env.manifold_geometry(state, nodes, 0.0)[:, free], jnp.zeros((3, 2)),
            )
            projected = filt(state, nodes, None, None)
            np.testing.assert_array_equal(projected[:, free], nodes[:, free])
            np.testing.assert_allclose(env.manifold_residual(state, projected), 0.0, atol=2e-6)
            if previous is not None:
                np.testing.assert_array_equal(projected, previous)
            previous = projected


def test_h1_other_clean_charts_preserve_existing_rows_and_dimensions():
    from genedynamics.solvers.single.atacom.backends.atacom_jax import atacom_null_dim

    for level, n_f, null_dim in (("push_to_line", 7, 5), ("heavy_dr", 7, 5),
                                 ("push_walk", 1, 22)):
        env = _h1_clean_manifold_array_env(level)
        action = jnp.linspace(-0.4, 0.4, env.action_size)
        force = jnp.asarray([(env._force_cmd(action[env.spec.nu_slice][0])
                              - env._bcfg.f_target) / (env._bcfg.f_max - env._bcfg.f_min)])
        expected = (force if env._is_walk else
                    jnp.concatenate([env._bcfg.s_scale * action[env.spec.s_slice], force]))
        assert env.manifold_constraint_size == n_f
        assert atacom_null_dim(env) == null_dim
        np.testing.assert_array_equal(env._manifold_res_node(action), expected)
        np.testing.assert_array_equal(env.manifold_residual(None, action[None]), expected)


def test_h1_p3_two_dimensional_atacom_transform_is_finite_without_physics():
    from genedynamics.solvers.single.atacom.backends.atacom_jax import (
        atacom_null_dim, init_slack, make_atacom_transform,
    )

    env = _h1_clean_manifold_array_env()
    # Isolate the real task-owned equality chart; regular negative slack rows
    # avoid making this array test a claim about the physical risk Jacobian.
    env.constraint_residual = lambda state, u: (
        jnp.zeros((14,), u.dtype), jnp.asarray([-0.5, -0.4, -0.3], u.dtype),
    )
    assert atacom_null_dim(env) == 2
    slack = init_slack(env, None)
    transform = make_atacom_transform(env)
    alpha = jnp.asarray([0.2, -0.3])
    action, slack_next = jax.jit(transform)(None, alpha, slack)
    assert action.shape == (12,) and slack_next.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert bool(jnp.all(jnp.isfinite(slack_next)))
    contact_jacobian = jax.jacfwd(lambda value: transform(None, value, slack)[0][3:5])(alpha)
    assert np.linalg.matrix_rank(np.asarray(contact_jacobian), tol=1e-6) == 2


def _paper_algorithm_contracts():
    from genedynamics.solvers.single.atacom.backends.atacom_jax import (
        atacom_null_dim,
    )

    expected = {
        "push_to_line": (7, 5),
        "unjam": (10, 2),
        "push_walk": (1, 22),
    }
    ok = True
    for level, (n_f, tangent) in expected.items():
        env = make_env(HUMANOID_TASK, level=level)
        state = env.reset(jax.random.PRNGKey(31))
        actions = jnp.zeros((5, env.action_size))
        prepared = env.prepare_realization_context(
            state, actions, gate_controllability=True,
        )
        residual = env.manifold_residual_horizon_controllable(
            prepared, actions, 0.0,
        )
        features = env.reliability_features_sequence(state, actions)
        emergency = env.emergency_plan(state, actions)
        valid = (
            env.manifold_constraint_size == n_f
            and atacom_null_dim(env) == tangent
            and features.shape == (24,)
            and residual.shape == (5 * n_f,)
            and emergency.shape == actions.shape
            and bool(jnp.isfinite(env.safety_index(state)))
        )
        print(
            f"  {level}: n_f={env.manifold_constraint_size}, "
            f"tangent={atacom_null_dim(env)}, features={features.shape}, "
            f"residual={residual.shape} -> {valid}"
        )
        ok &= valid
    return ok


def _atacom_finite_rollout():
    """ATACOM must stay finite under exploratory PPO actions."""
    from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper

    cases = (
        ("push_to_line", {}),
        ("unjam", {"push_dist": 0.03, "contact_band": 0.02}),
        ("push_walk", {"push_dist": 0.30}),
    )
    ok = True
    for level, overrides in cases:
        wrapped = AtacomEnvWrapper(
            make_env(HUMANOID_TASK, level=level, **overrides)
        )
        key = jax.random.PRNGKey(700 + len(level))
        state = wrapped.reset(key)
        step_fn = jax.jit(wrapped.step)
        failed_step = None
        for step in range(20):
            key, action_key = jax.random.split(key)
            alpha = jax.random.uniform(
                action_key, (wrapped.action_size,), minval=-1.0, maxval=1.0,
            )
            state = step_fn(state, alpha)
            leaves = {
                "obs": state.obs, "reward": state.reward,
                "q": state.pipeline_state.q, "qd": state.pipeline_state.qd,
                "slack": state.info["atacom_s"],
                "u": state.info["atacom_u"],
            }
            nonfinite = [
                name for name, value in leaves.items()
                if not bool(jnp.all(jnp.isfinite(value)))
            ]
            if nonfinite:
                failed_step = step
                print(f"    nonfinite={nonfinite}")
                break
        valid = failed_step is None
        print(f"  {level}: finite={valid}, failed_step={failed_step}")
        ok &= valid
    return ok


def _h1_terminal_array_fixture():
    """Exercise the actual H1 terminal/rollout hooks without MJX compilation."""
    from collections import namedtuple
    from types import SimpleNamespace
    from brax.envs.base import State
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    env = object.__new__(HumanoidBoxPushEnv)
    env._bcfg = HumanoidBoxPushConfig(push_dist=0.05)
    env.sys = SimpleNamespace(opt=SimpleNamespace(timestep=0.02))
    env._n_frames = 1
    env._walk_requires_locomotion = False
    env._torso_idx, env._box_idx, env._torso_z0 = 1, 2, 1.0
    Pose = namedtuple("TerminalPose", "pos rot")
    Physics = namedtuple("TerminalPhysics", "x")
    ps = Physics(Pose(jnp.asarray([[0., 0., 1.], [0., 0., 0.]]),
                      jnp.tile(jnp.asarray([1., 0., 0., 0.]), (2, 1))))
    env._get_obs = lambda ps, info: ps.x.pos[:, 0]
    env._task_reached = lambda ps, info: ps.x.pos[1, 0] >= 0.05
    env._box_contact_force = lambda ps: jnp.float32(65.)
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(65.), "nonhand": jnp.float32(0.)}
    env._contact_acquisition_detected = lambda ps, u, info, measured: jnp.bool_(True)
    env._manifold = lambda ps, u, info: (jnp.asarray([2.]), jnp.zeros(3))
    env._reward_done = lambda ps, u, info: (
        jnp.float32(0.3),
        (env._task_reached(ps, info) | env._has_fallen(ps)).astype(jnp.float32),
    )
    def safety_sample(current):
        # This synthetic task defines balance through its mocked manifold;
        # its tiny Physics lacks the real sampler's foot-site/pelvis arrays.
        forces = env._box_contact_forces(current)
        _, inequalities = env._manifold(current, None, None)
        return {
            "physics_hand_force": forces["hand"],
            "physics_nonhand_force": forces["nonhand"],
            "physics_safety_margins": env._safety_margins(current, forces, inequalities[0]),
        }

    env._physics_safety_sample = safety_sample
    state = State(pipeline_state=ps, obs=jnp.zeros(2), reward=jnp.float32(0),
                  done=jnp.float32(0), metrics={}, info={
                      "step": jnp.int32(0), "force_int": jnp.float32(0),
                      "contact_acquired": jnp.float32(0), "contact_step": jnp.int32(-1),
                      "task_success": jnp.float32(0), "task_fallen": jnp.float32(0),
                      "success_padding": jnp.bool_(False), "prev_action": jnp.zeros(2),
                      "nested": {"value": jnp.float32(7)},
                      **env._empty_physics_safety_samples(),
                  })

    def step(s, u):
        positions = s.pipeline_state.x.pos.at[1, 0].add(u[0])
        positions = positions.at[0, 2].add(u[1])
        moved = Physics(s.pipeline_state.x._replace(pos=positions))
        forces = env._box_contact_forces(moved)
        _, g = env._manifold(moved, u, s.info)
        samples = {
            "physics_hand_force": forces["hand"][None],
            "physics_nonhand_force": forces["nonhand"][None],
            "physics_safety_margins": env._safety_margins(moved, forces, g[0])[None],
        }
        return env._finish_step(s, moved, u, jnp.float32(3), samples)

    env.step = step
    return env, state


def test_h1_success_padding_preserves_completion_cost_physics_and_fall_priority():
    env, state = _h1_terminal_array_fixture()
    step = jax.jit(env.step)
    completed = step(state, jnp.asarray([0.05, 0.]))
    assert float(completed.done) == 1.0
    assert float(completed.info["task_success"]) == 1.0
    assert not bool(completed.info["success_padding"])
    np.testing.assert_allclose(completed.reward, 0.3)
    # Counterfactual tail commands cannot execute a later fall or erase the
    # unsafe 65 N contact at the true completion step.
    padding = step(completed, jnp.asarray([1., -1.]))
    assert bool(padding.info["success_padding"])
    assert float(padding.info["task_fallen"]) == 0.0
    np.testing.assert_array_equal(padding.pipeline_state.x.pos, completed.pipeline_state.x.pos)
    np.testing.assert_array_equal(padding.info["prev_action"], completed.info["prev_action"])
    assert float(env._box_contact_force(padding.pipeline_state)) == 65.
    assert int(padding.info["step"]) == 2
    np.testing.assert_allclose(padding.reward, 1.4)  # alive + saturated task progress
    h, _ = env.constraint_residual(padding, jnp.zeros(2))
    np.testing.assert_array_equal(h, [2.])  # physical constraint was NOT erased
    h_soft, g_soft = env.soft_feasibility_residual(padding, jnp.zeros(2))
    np.testing.assert_array_equal(h_soft, [0.])
    np.testing.assert_array_equal(g_soft, [0., 0., 0.])

    fallen_at_goal = step(state, jnp.asarray([0.05, -0.5]))
    assert float(fallen_at_goal.done) == 1.0
    assert float(fallen_at_goal.info["task_fallen"]) == 1.0
    assert float(fallen_at_goal.info["task_success"]) == 0.0
    assert not bool(fallen_at_goal.info["success_padding"])
    np.testing.assert_allclose(fallen_at_goal.reward, 0.3)


def test_h1_success_padding_aligns_dial_and_mga_objective_without_masking_risk():
    from genedynamics.solvers.common.env_rollout import (
        build_brax_rollout, build_brax_rollout_augmented,
    )

    env, state = _h1_terminal_array_fixture()
    early = jnp.asarray([[0.05, 0.], [1., -1.], [1., -1.], [1., -1.]])
    hold = jnp.zeros_like(early)
    actions = jnp.stack([early, hold])
    raw = build_brax_rollout(env)(state, actions, 0.)
    augmented = build_brax_rollout_augmented(env)(state, actions, 0., 0.1, 0.1)
    raw_microbatch = build_brax_rollout(
        env, candidate_batch_size=1
    )(state, actions, 0.)
    augmented_microbatch = build_brax_rollout_augmented(
        env, candidate_batch_size=1
    )(state, actions, 0., 0.1, 0.1)
    np.testing.assert_array_equal(raw_microbatch, raw)
    np.testing.assert_array_equal(augmented_microbatch, augmented)
    with pytest.raises(ValueError, match="candidate_batch_size"):
        build_brax_rollout(env, candidate_batch_size=0)
    np.testing.assert_allclose(raw[0], [0.3, 1.4, 1.4, 1.4])
    np.testing.assert_allclose(raw[1], [0.3] * 4)
    # The actual completion still pays lambda*2 + rho/2*4 = 0.4.
    np.testing.assert_allclose(augmented[0], [-0.1, 1.4, 1.4, 1.4], atol=1e-7)
    np.testing.assert_allclose(augmented[1], [-0.1] * 4, atol=1e-7)
    score, risk = jax.jit(env.sequence_score_risk)(state, early, 0.1, 0.1)
    np.testing.assert_allclose(score, jnp.mean(augmented[0]), atol=1e-7)
    assert float(risk[0]) == 1.0  # unsafe completion remains unsafe for the gate
    assert not bool(env.sequence_risk_is_safe(risk))
    hold_score, _ = env.sequence_score_risk(state, hold, 0.1, 0.1)
    assert float(score) > float(hold_score)
    assert float(jnp.mean(raw[0])) > float(jnp.mean(raw[1]))


def test_h1_physics_envelope_catches_interior_events_and_rejects_unknown_coverage():
    from dataclasses import replace

    env, state = _h1_terminal_array_fixture()
    env._n_frames = 5
    endpoint = jnp.asarray([-.5, -.01, -.2, -.4])
    margins = jnp.tile(endpoint, (5, 1)).at[1].set(jnp.asarray([2., .03, .4, .1]))
    state = state.replace(info={**state.info, **env._empty_physics_safety_samples(),
        "step": jnp.int32(1), "physics_samples_valid": jnp.bool_(True),
        "physics_hand_force": jnp.asarray([30., 180., 30., 30., 30.]),
        "physics_nonhand_force": jnp.asarray([0., 2.3, 0., 0., 0.]),
        "physics_safety_margins": margins})
    evaluate = jax.jit(lambda s: env._transition_safety_margins(s, endpoint))
    np.testing.assert_array_equal(evaluate(state), margins[1])
    # The real sequence-risk hook shares the interval envelope, not only its
    # safe last substep. Its fourth head remains endpoint force MAE.
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(30.), "nonhand": jnp.float32(0.)}
    env._manifold = lambda ps, u, info: (jnp.zeros(0), jnp.asarray([-.05, 0., 0.]))
    env.step = lambda s, u: s
    risk = jax.jit(env.sequence_risk)(state, jnp.zeros((2, 2)))
    np.testing.assert_allclose(risk[:3], [1., 1., .4])
    assert not bool(env.sequence_risk_is_safe(risk))
    unknown = state.replace(info={**state.info, "physics_samples_valid": jnp.bool_(False)})
    assert bool(jnp.all(jnp.isposinf(evaluate(unknown))))
    # Missing/native coverage is unknown, NOT observed positive training data.
    assert not bool(env.sequence_risk_is_safe(env.sequence_risk(unknown, jnp.zeros((1, 2)))))
    for extra in ({"step": jnp.int32(0)}, {"success_padding": jnp.bool_(True)}):
        np.testing.assert_array_equal(evaluate(unknown.replace(info={**unknown.info, **extra})), endpoint)
    nonfinite = state.replace(info={**state.info,
        "physics_safety_margins": margins.at[0, 0].set(jnp.nan)})
    assert bool(jnp.all(jnp.isposinf(evaluate(nonfinite))))
    env._bcfg = replace(env._bcfg, robot="g1")
    np.testing.assert_array_equal(env._transition_safety_margins(state, endpoint), endpoint)


def test_h1_physics_samples_retain_completion_but_not_padding_and_latch_substep_fall():
    env, state = _h1_terminal_array_fixture()
    env._n_frames = 5
    state = state.replace(info={**state.info, **env._empty_physics_safety_samples()})
    env._box_contact_force = lambda ps: jnp.float32(30.)
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(30.), "nonhand": jnp.float32(0.)}
    moved = state.pipeline_state._replace(x=state.pipeline_state.x._replace(
        pos=state.pipeline_state.x.pos.at[1, 0].set(.05)))
    samples = {
        "physics_hand_force": jnp.asarray([30., 180., 30., 30., 30.]),
        "physics_nonhand_force": jnp.zeros(5),
        "physics_safety_margins": jnp.tile(jnp.asarray([-.5, -.01, -.2, -.4]), (5, 1)).at[1, 0].set(2.),
    }
    finish = jax.jit(env._finish_step)
    completed = finish(state, moved, jnp.zeros(2), jnp.float32(3.), samples)
    assert bool(completed.info["task_success"]) and bool(completed.info["physics_samples_valid"])
    np.testing.assert_array_equal(completed.info["physics_hand_force"], samples["physics_hand_force"])
    np.testing.assert_allclose(completed.reward, .3)
    # A mid-interval fall is terminal even though the endpoint stands upright;
    # reaching the line cannot mask that event or claim success.
    transient_fall = {**samples, "physics_safety_margins": samples["physics_safety_margins"].at[2, 3].set(.1)}
    fallen = finish(state, moved, jnp.zeros(2), jnp.float32(3.), transient_fall)
    assert bool(fallen.done) and bool(fallen.info["task_fallen"])
    assert not bool(fallen.info["task_success"])
    np.testing.assert_allclose(fallen.reward, .3)  # termination changes; reward formula does not
    padding = finish(completed, moved, jnp.ones(2), jnp.float32(0.), transient_fall)
    assert bool(padding.info["success_padding"]) and not bool(padding.info["physics_samples_valid"])
    assert not bool(padding.info["task_fallen"])
    np.testing.assert_array_equal(padding.info["physics_hand_force"], jnp.zeros(5))
    np.testing.assert_array_equal(padding.info["physics_safety_margins"], jnp.zeros((5, 4)))
    endpoint = jnp.asarray([-.5, -.01, -.2, -.4])
    np.testing.assert_array_equal(env._transition_safety_margins(padding, endpoint), endpoint)
    native = finish(state, moved, jnp.zeros(2), jnp.float32(3.))
    assert not bool(native.info["physics_samples_valid"])


def test_h1_risk_certificate_uses_strict_physical_zero_and_preserves_g1():
    from dataclasses import replace

    env, _ = _h1_terminal_array_fixture()
    tiny_violation = jnp.asarray([0., 0., 5.e-9, 0.])
    assert bool(env.sequence_risk_is_safe(jnp.zeros(4)))
    assert not bool(env.sequence_risk_is_safe(tiny_violation))
    env._bcfg = replace(env._bcfg, robot="g1")
    assert bool(env.sequence_risk_is_safe(jnp.zeros(4)))
    assert bool(env.sequence_risk_is_safe(tiny_violation))


def test_h1_physics_sampler_preserves_fast_loop_arithmetic_and_observation():
    """Array pipeline verifies collection ordering without a physics rollout."""
    from collections import namedtuple
    from types import SimpleNamespace
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, state = _h1_terminal_array_fixture()
    del env.step
    del env._physics_safety_sample  # This test exercises the real production sampler.
    env._n_frames, env._debug = 5, False
    env.sys = SimpleNamespace(opt=SimpleNamespace(timestep=.004))
    env._pelvis_idx, env._feet_site_id = 1, jnp.asarray([0, 1])
    Physics = namedtuple("SafetySamplePhysics", "q qd x site_xpos")
    ps = Physics(jnp.zeros(1), jnp.zeros(1), state.pipeline_state.x, jnp.zeros((2, 3)))
    state = state.replace(pipeline_state=ps,
        info={**state.info, **env._empty_physics_safety_samples()})
    force_trace = jnp.asarray([20., 90., 25., 35., 30.])
    env._box_contact_forces = lambda ps: {
        "hand": force_trace[jnp.clip(ps.q[0].astype(jnp.int32) - 1, 0, 4)],
        "nonhand": jnp.float32(0.),
    }
    env._box_contact_force = lambda ps: env._box_contact_forces(ps)["hand"]
    env._control = lambda ps, u, info: ps.q + u[:1] + .1 * info["force_int"]
    env._update_force_integral = lambda ps, u, info: info["force_int"] + ps.q[0] * .01
    env._pipeline = SimpleNamespace(step=lambda sys, ps, tau, debug:
        ps._replace(q=ps.q + 1., qd=ps.qd + .004 * tau))
    action = jnp.asarray([.2, 0.])

    def original_loop(s, u):
        def body(carry, unused):
            current, integral = carry
            info = {**s.info, "force_int": integral}
            tau = env._control(current, u, info)
            current = env._pipeline.step(env.sys, current, tau, env._debug)
            return (current, env._update_force_integral(current, u, info)), None
        (current, integral), _ = jax.lax.scan(body,
            (s.pipeline_state, s.info["force_int"]), (), env._n_frames)
        return env._finish_step(s, current, u, integral)

    actual = jax.jit(env.step)(state, action)
    reference = jax.jit(original_loop)(state, action)
    for name in ("q", "qd"):
        np.testing.assert_array_equal(getattr(actual.pipeline_state, name), getattr(reference.pipeline_state, name))
    for name in ("obs", "reward", "done"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
    for key in state.info:
        if not key.startswith("physics_"):
            jax.tree_util.tree_map(np.testing.assert_array_equal, actual.info[key], reference.info[key])
    np.testing.assert_array_equal(actual.info["physics_hand_force"], force_trace)
    assert bool(actual.info["physics_samples_valid"])
    assert not bool(reference.info["physics_samples_valid"])
    assert actual.info["physics_safety_margins"].shape == (5, 4)
    # Non-fast control retains its old held-torque pipeline; no fake five-sample tape.
    env._bcfg.fast_force_loop = False
    env.pipeline_step = lambda ps, tau: ps._replace(q=ps.q + 5., qd=ps.qd + .02 * tau)
    nonfast = HumanoidBoxPushEnv.step(env, state, action)
    assert not bool(nonfast.info["physics_samples_valid"])


def test_h1_box_support_requires_current_right_hand_compression(monkeypatch):
    """The new mode bit must not alter historical force summary values."""
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid import box_push_brax as module

    env = NS(_box_geom=3, _rhand_geom=1, _lhand_geom=2,
             _wall_geoms=jnp.array([4, 5]), _floor_geom=0, sys=NS())
    # Right compression; reversed pair; exact touch; left-only; tensile;
    # unloaded; separated; non-box right-hand contact; wall; invalid body.
    geoms = jnp.array([[3, 1], [1, 3], [3, 1], [3, 2], [3, 1],
                       [3, 1], [3, 1], [0, 1], [3, 4], [3, 6]])
    distances = jnp.array([-.01, -.01, 0., -.01, -.01, -.01, .01, -.01, -.01, -.01])
    normals = jnp.array([2., 2., 2., 3., -2., 0., 2., 2., 4., 5.])
    monkeypatch.setattr(module._mjx_support, "contact_force",
                        lambda system, state, index: jnp.zeros(6).at[0].set(state.normal[index]))

    def evaluate(geom, distance, normal):
        ps = NS(contact=NS(geom=geom[None], dist=distance[None]), normal=normal[None])
        return module.HumanoidBoxPushEnv._box_contact_forces(env, ps)

    forces = jax.jit(jax.vmap(evaluate))(geoms, distances, normals)
    np.testing.assert_array_equal(forces["right_supported"],
                                  [True, True, True, False, False, False, False, False, False, False])
    # In particular, legacy summaries keep strict dist<0 and abs(fn); the
    # new contact-mode bit uses dist<=0 and positive compression instead.
    np.testing.assert_array_equal(forces["hand"], [2., 2., 0., 3., 2., 0., 0., 0., 0., 0.])
    np.testing.assert_array_equal(forces["wall"], [0., 0., 0., 0., 0., 0., 0., 0., 4., 0.])
    np.testing.assert_array_equal(forces["nonhand"], [0., 0., 0., 0., 0., 0., 0., 0., 0., 5.])


def test_h1_p3_soft_friction_uses_right_support_not_latched_acquisition():
    from dataclasses import replace

    env, state = _h1_terminal_array_fixture()
    env._bcfg = replace(env._bcfg, level="unjam")
    env._manifold = lambda ps, u, info: (jnp.array([.1, .2]), jnp.array([-.1, 2., .3]))
    env._box_contact_forces = lambda ps: {
        "hand": jnp.float32(8.), "wall": jnp.float32(0.), "nonhand": jnp.float32(0.),
        "right_supported": ps.x.pos[1, 1] > 0,
    }
    evaluate = jax.jit(lambda s: (env.constraint_residual(s, jnp.zeros(2)),
                                 env.soft_feasibility_residual(s, jnp.zeros(2))))
    # Hand force can be left-only, while an old acquisition latch remains set.
    # Conversely actual right support applies even before the latch is updated.
    for right_supported, acquired in ((False, 1.), (False, 0.), (True, 0.), (True, 1.)):
        ps = state.pipeline_state._replace(x=state.pipeline_state.x._replace(
            pos=state.pipeline_state.x.pos.at[1, 1].set(float(right_supported))))
        current = state.replace(pipeline_state=ps, info={**state.info, "contact_acquired": acquired})
        (h_raw, g_raw), (h_soft, g_soft) = evaluate(current)
        np.testing.assert_array_equal(h_soft, h_raw)
        np.testing.assert_array_equal(g_raw, jnp.array([-.1, 2., .3]))
        np.testing.assert_array_equal(g_soft, jnp.array([-.1, 2. if right_supported else 0., .3]))
        # Same fixed-shape raw hook is still consumed by ATACOM; 2 N has not
        # been divided by f_max or replaced by the measured support load.
        assert g_raw.shape == g_soft.shape == (3,)


def test_h1_p3_soft_scores_share_modes_but_keep_raw_padding_risk():
    from dataclasses import replace
    from genedynamics.solvers.common.env_rollout import build_brax_rollout_augmented

    env, state = _h1_terminal_array_fixture()
    env._bcfg = replace(env._bcfg, level="unjam")
    env._manifold = lambda ps, u, info: (jnp.array([2.]), jnp.array([.25, 2., 0.]))
    env._box_contact_forces = lambda ps: {
        "hand": jnp.float32(65.), "wall": jnp.float32(0.), "nonhand": jnp.float32(0.),
        "right_supported": ps.x.pos[1, 1] > 0,
    }

    def step(s, u):
        ps = s.pipeline_state._replace(x=s.pipeline_state.x._replace(
            pos=s.pipeline_state.x.pos.at[1, 1].set(u[0])))
        return s.replace(pipeline_state=ps, reward=jnp.float32(1.),
                         info={**s.info, "success_padding": u[1] > 0})

    env.step = step
    actions = jnp.array([[0., 0.], [1., 0.], [0., 0.], [0., 1.]])
    augmented = build_brax_rollout_augmented(env)(state, actions[None], 0., .1, .1)[0]
    score, risk = jax.jit(env.sequence_score_risk)(state, actions, .1, .1)
    np.testing.assert_allclose(augmented, [.571875, .171875, .571875, 1.], atol=1e-7)
    np.testing.assert_allclose(score, augmented.mean(), atol=1e-7)
    assert float(risk[0]) == 1.  # physical 65 N remains unsafe
    assert float(risk[2]) == 1.  # raw balance .25/.25, including padding
    assert not bool(env.sequence_risk_is_safe(risk))


def test_h1_soft_friction_change_preserves_other_task_residuals():
    from dataclasses import replace

    env, state = _h1_terminal_array_fixture()
    env._manifold = lambda ps, u, info: (jnp.array([.1, .2]), jnp.array([-.1, 2., .3]))

    def unused_forces(ps):
        raise AssertionError("non-P3 legacy AL must not request the new mode bit")

    env._box_contact_forces = unused_forces
    for robot, level in (("h1", "push_to_line"), ("h1", "heavy_dr"),
                         ("h1", "push_walk"), ("g1", "unjam")):
        env._bcfg = replace(env._bcfg, robot=robot, level=level)
        for padding in (False, True):
            current = state.replace(info={**state.info, "success_padding": jnp.bool_(padding)})
            h, g = env.constraint_residual(current, jnp.zeros(2))
            h_soft, g_soft = env.soft_feasibility_residual(current, jnp.zeros(2))
            np.testing.assert_array_equal(h_soft, jnp.zeros_like(h) if padding else h)
            np.testing.assert_array_equal(g_soft, jnp.zeros_like(g) if padding else g)


def test_h1_absorbing_reward_bound_refuses_invalid_weights_before_physics():
    import pytest
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    for params in ({"w_prog": -1.0}, {"w_force": float("nan")}, {"w_gait": float("inf")}):
        with pytest.raises(ValueError, match="reward weights"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(**params))
    for distance in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="push_dist"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(push_dist=distance))


def test_h1_safety_entry_points_share_force_contact_balance_and_fall_boundaries():
    """A 0.55 m torso must not pass a gate when the task has already fallen."""
    from types import SimpleNamespace

    env, state = _h1_terminal_array_fixture()
    env._is_walk = False
    env.spec = SimpleNamespace(total_width=2)
    env.step = lambda s, u: s  # static array states, not a physics approximation
    env._box_contact_forces = lambda ps: {
        "hand": ps.x.pos[1, 1], "nonhand": ps.x.pos[1, 2],
    }
    env._manifold = lambda ps, u, info: (jnp.zeros(0), jnp.asarray([ps.x.pos[1, 0], 0., 0.]))
    evaluate = jax.jit(lambda s: (
        env.sequence_score_risk(s, jnp.zeros((2, 2)))[1],
        env.safety_index(s), env.emergency_plan_should_override(s),
        env._has_fallen(s.pipeline_state),
    ))
    cases = (
        # hand force, nonhand force, balance residual, torso height, unsafe, fallen
        (60., .5, 0., .6, False, False),
        (60.01, 0., -.1, 1., True, False),
        (30., .5001, -.1, 1., True, False),
        (30., 0., .0001, 1., True, False),
        (30., 0., -.1, .55, True, True),
        (30., 0., -.1, 1., False, False),
    )
    for force, nonhand, balance, height, unsafe, fallen in cases:
        positions = state.pipeline_state.x.pos.at[0, 2].set(height)
        positions = positions.at[1].set(jnp.asarray([balance, force, nonhand]))
        ps = state.pipeline_state._replace(x=state.pipeline_state.x._replace(pos=positions))
        risk, index, override, is_fallen = evaluate(state.replace(pipeline_state=ps))
        assert bool(index > 0.0) == unsafe
        assert bool(override) == unsafe
        assert bool(env.sequence_risk_is_safe(risk)) == (not unsafe)
        assert bool(is_fallen) == fallen
    inverted = ps._replace(x=ps.x._replace(rot=ps.x.rot.at[0].set(jnp.asarray([0., 1., 0., 0.]))))
    risk, index, override, fallen = evaluate(state.replace(pipeline_state=inverted))
    assert bool(fallen) and bool(override) and float(index) > 0.
    assert not bool(env.sequence_risk_is_safe(risk))


def _h1_acquisition_array_fixture(half=(0.5, 0.5, 0.5), radii=(0.033, 0.033)):
    """Actual sphere/OBB geometry without constructing or stepping physics."""
    from collections import namedtuple
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )
    from genedynamics.envs.obstacles.convex import BoxObstacle

    env = object.__new__(HumanoidBoxPushEnv)
    env._bcfg = HumanoidBoxPushConfig()
    env._n_frames = 1
    env._box_geom, env._rhand_geom, env._lhand_geom = 0, 1, 2
    env.sys = NS(geom_size=jnp.asarray([half, (radii[0], 0, 0), (radii[1], 0, 0)]))
    env._acquisition_box = BoxObstacle(np.zeros(3), half)
    GeometryState = namedtuple("AcquisitionGeometryState", "geom_xpos geom_xmat")

    def pose(points, center=(0.0, 0.0, 0.0), rotation=None):
        rotation = jnp.eye(3) if rotation is None else jnp.asarray(rotation)
        center = jnp.asarray(center)
        world = jnp.asarray(points) @ rotation.T + center
        return GeometryState(jnp.vstack([center, world]), jnp.broadcast_to(rotation, (3, 3, 3)))

    # Deliberately reproduce the old early-approach p_c == p_hand condition.
    env._hand_contact = lambda ps, action, info: {
        "p_hand": ps.geom_xpos[1], "p_c": ps.geom_xpos[1],
        "n_c": jnp.asarray([-1.0, 0.0, 0.0]),
        "left": {"p_hand": ps.geom_xpos[2], "p_c": ps.geom_xpos[2]},
    }
    return env, pose


def test_h1_acquisition_uses_actual_two_hand_gap_and_measured_force():
    from brax.envs.base import State

    env, pose = _h1_acquisition_array_fixture()
    action = jnp.zeros(12)
    far = pose([[-0.633, -0.1, 0.0], [-0.633, 0.1, 0.0]])
    near = pose([[-0.543, -0.1, 0.0], [-0.543, 0.1, 0.0]])
    one_near = pose([[-0.543, -0.1, 0.0], [-0.563, 0.1, 0.0]])
    detect = jax.jit(lambda ps, u, force: env._contact_acquisition_detected(ps, u, {}, force))
    assert not bool(detect(far, action, 0.0))  # interpolated targets are nevertheless exactly at hands
    assert bool(detect(near, action, 0.0))
    assert not bool(detect(one_near, action, 0.0))
    assert not bool(detect(far, action, 0.4999))
    assert bool(detect(far, action, 0.5))
    assert bool(detect(near, jnp.ones(12), 0.0))  # selected face / stiffness cannot change acquisition
    np.testing.assert_allclose(env._hand_box_acquisition_gaps(near), [0.01, 0.01], atol=1e-7)

    # Exercise the actual task transition: no latch at reset-time p_c == hand,
    # then latch exactly once at a real near approach without changing the ramp.
    env._walk_requires_locomotion = False
    env._reward_done = lambda ps, u, info: (jnp.float32(0), jnp.float32(0))
    env._box_contact_force = lambda ps: jnp.float32(0)
    env._task_reached = lambda ps, info: jnp.bool_(False)
    env._has_fallen = lambda ps: jnp.bool_(False)
    env._get_obs = lambda ps, info: jnp.zeros(1)
    state = State(pipeline_state=far, obs=jnp.zeros(1), reward=jnp.float32(0),
                  done=jnp.float32(0), metrics={}, info={
                      "step": jnp.int32(0), "force_int": jnp.float32(0),
                      "contact_acquired": jnp.float32(0), "contact_step": jnp.int32(-1),
                      "task_success": jnp.float32(0), "prev_action": action,
                      "task_fallen": jnp.float32(0), "success_padding": jnp.bool_(False),
                      **env._empty_mga_execution_context(),
                      **env._empty_physics_safety_samples(),
                  })
    finish = jax.jit(env._finish_step)
    first = finish(state, far, action, jnp.float32(0))
    assert float(first.info["contact_acquired"]) == 0.0
    assert int(first.info["contact_step"]) == -1
    second = finish(first, near, action, jnp.float32(0))
    assert float(second.info["contact_acquired"]) == 1.0
    assert int(second.info["contact_step"]) == 2
    assert int(finish(second, near, action, jnp.float32(0)).info["contact_step"]) == 2


def test_h1_acquisition_finite_box_geometry_and_legacy_boundaries():
    env, pose = _h1_acquisition_array_fixture(half=(0.4, 0.6, 0.8), radii=(0.02, 0.05))
    action = jnp.zeros(12)
    # Different actual radii and DR half-extents, including a side face.
    points = [[-0.43, -0.1, 0.0], [0.0, 0.66, 0.0]]
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    gaps = jax.jit(env._hand_box_acquisition_gaps)(pose(points, (1.2, -0.3, 0.7), rotation))
    np.testing.assert_allclose(gaps, [0.01, 0.01], atol=2e-7)
    assert bool(env._contact_acquisition_detected(pose(points), action, {}, 0.0))
    # An infinite face projection would see a small normal gap; the sphere is
    # actually outside the finite face near its corner and is not close.
    corner = pose([[-0.43, 0.8, 0.0], [0.0, 0.66, 0.0]])
    assert float(env._hand_box_acquisition_gaps(corner)[0]) > 0.18
    assert not bool(env._contact_acquisition_detected(corner, action, {}, 0.0))
    overlap = pose([[-0.41, 0.0, 0.0], [0.0, 0.66, 0.0]])
    assert float(env._hand_box_acquisition_gaps(overlap)[0]) < 0.0
    assert not bool(env._contact_acquisition_detected(overlap, action, {}, 0.0))
    assert bool(env._contact_acquisition_detected(overlap, action, {}, 0.5))
    invalid = pose([[jnp.nan, 0.0, 0.0], [0.0, 0.66, 0.0]])
    assert not bool(env._contact_acquisition_detected(invalid, action, {}, 0.0))
    assert bool(env._contact_acquisition_detected(invalid, action, {}, 0.5))
    # Test the exact inclusive band separately from float32 world-coordinate
    # cancellation; negative gaps must be confirmed by measured force.
    env._hand_box_acquisition_gaps = lambda ps: jnp.asarray([0.0, env._bcfg.contact_acquire_gap])
    assert bool(env._contact_acquisition_detected(corner, action, {}, 0.0))
    env._hand_box_acquisition_gaps = lambda ps: jnp.asarray([-1e-7, 0.0])
    assert not bool(env._contact_acquisition_detected(corner, action, {}, 0.0))
    # Unsupported non-sphere/G1 branch keeps the old projected-gap arithmetic.
    env._acquisition_box = None
    assert bool(env._contact_acquisition_detected(corner, action, {}, 0.0))
    contact = env._hand_contact(corner, action, {})
    contact["p_c"] = contact["p_c"].at[0].add(0.03)
    env._hand_contact = lambda ps, u, info: contact
    legacy = jnp.maximum(
        jnp.dot(contact["p_hand"] - contact["p_c"], contact["n_c"]),
        jnp.dot(contact["left"]["p_hand"] - contact["left"]["p_c"], contact["n_c"]),
    ) <= env._bcfg.contact_acquire_gap
    np.testing.assert_array_equal(env._contact_acquisition_detected(corner, action, {}, 0.0), legacy)
    assert not bool(legacy)


def test_h1_walk_contact_diagnostics_keep_measured_and_nominal_support_separate():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env = object.__new__(HumanoidBoxPushEnv)
    env._floor_geom = 0
    env._foot_body_ids = jnp.asarray([2, 3])
    env._pelvis_idx, env._torso_idx, env._box_idx = 1, 1, 4
    env.sys = NS(geom_bodyid=jnp.asarray([0, 2, 3, 4, 5]))
    loads = jnp.asarray([300.0, 0.0])
    env._foot_contact_loads = lambda ps: loads
    ps = NS(contact=NS(
        geom=jnp.asarray([[0, 1], [1, 0], [0, 2], [1, 3], [0, -1]]),
        dist=jnp.asarray([-0.001, 0.002, 0.03, -0.2, -0.5])),
        subtree_com=jnp.asarray([[0., 0., 0.], [0.1, 0.0, 0.9],
                                 [0., 0., 0.], [0., 0., 0.], [9., 9., 9.]]),
        x=NS(pos=jnp.asarray([[0., 0., 0.98], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
             rot=jnp.tile(jnp.asarray([1., 0., 0., 0.]), (4, 1))),
        xd=NS(vel=jnp.asarray([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0.2, 0., 0.]])))
    out = jax.jit(lambda: env._walk_contact_diagnostics(ps))()
    np.testing.assert_array_equal(out["foot_normal_loads"], loads)
    np.testing.assert_allclose(out["foot_floor_clearance"], [-0.001, 0.03], atol=1e-8)
    np.testing.assert_array_equal(out["foot_ground_contact"], [True, False])
    np.testing.assert_allclose(out["robot_subtree_com"], [0.1, 0.0, 0.9])
    np.testing.assert_allclose([out["torso_up"], out["torso_height"], out["box_forward_velocity"]],
                               [1.0, 0.98, 0.2])
    ps.contact.geom = jnp.asarray([[0, 1], [1, 0], [3, 2], [1, 3], [0, -1]])
    assert np.isposinf(np.asarray(env._walk_contact_diagnostics(ps)["foot_floor_clearance"])[1])


def test_h1_acquisition_constructor_uses_actual_dr_geometry_without_steps():
    env = make_env(HUMANOID_TASK, level="heavy_dr", dr_seed=2, approach_gap=0.08)
    state = env.reset(jax.random.PRNGKey(0))
    assert env._acquisition_box is not None
    np.testing.assert_array_equal(env._acquisition_box.half_extents,
                                  np.asarray(env.sys.geom_size[env._box_geom]))
    gaps = env._hand_box_acquisition_gaps(state.pipeline_state)
    ps = state.pipeline_state
    hands = np.asarray([env._rhand_geom, env._lhand_geom])
    local = ((np.asarray(ps.geom_xpos)[hands] - np.asarray(ps.geom_xpos)[env._box_geom])
             @ np.asarray(ps.geom_xmat)[env._box_geom])
    expected = env._acquisition_box.sdf(local) - np.asarray(env.sys.geom_size)[hands, 0]
    np.testing.assert_allclose(gaps, expected, atol=2e-6)
    assert bool(jnp.all(gaps >= env._bcfg.approach_gap - 2e-6))
    assert not bool(env._contact_acquisition_detected(
        state.pipeline_state, jnp.zeros(env.action_size), state.info, jnp.float32(0)))



def test_h1_dial_walk_objective_rejects_invalid_scope_before_model_loading():
    import pytest
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    valid = dict(robot="h1", level="push_walk", walk_leg_control="joint_target",
                 walk_success_mode="locomotion", walk_objective_mode="dial")
    for change in (
        {"walk_objective_mode": "unknown"}, {"robot": "g1"},
        {"level": "push_to_line"}, {"walk_leg_control": "legacy"},
        {"walk_success_mode": "legacy"}, {"walk_velocity_ramp_time": 0.},
        {"walk_velocity_ramp_time": np.nan}, {"walk_height_target": -1.},
        {"walk_height_target": np.inf}, {"target_vx": -0.1}, {"target_vx": np.nan},
    ):
        with pytest.raises(ValueError, match="walk|target_vx"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(**{**valid, **change}))


def test_h1_dial_walk_reward_uses_config_clock_body_xy_and_vendor_frames():
    from collections import namedtuple
    from types import MethodType, SimpleNamespace as NS
    from brax.base import Motion, Transform
    from brax import math as bmath
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, _, info = _h1_measured_walk_array_fixture()
    cfg = env._bcfg
    for name in vars(cfg):
        if name.startswith("w_"):
            setattr(cfg, name, 0.)
    cfg.walk_objective_mode, cfg.walk_velocity_ramp_time = "dial", 2.
    cfg.target_vx, cfg.walk_height_target = .25, 1.2
    cfg.w_gait, cfg.w_vel, cfg.w_upright = 5., 1., .01
    cfg.w_height, cfg.w_yaw, cfg.w_angvel = .5, .1, (np.pi / 180.) ** 2
    cfg.w_alive = 1.
    env._torso_idx, env._box_idx, env._pelvis_idx, env._align_yaw = 2, 3, 1, 1.
    env._robot_binding = NS(dof_indices=tuple(range(19)))
    env.spec = NS(s_slice=slice(5, 11))
    env._manifold = lambda ps, u, info: (jnp.zeros(10), jnp.zeros(3))
    env._hand_contact = lambda ps, u, info: {}
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(0), "nonhand": jnp.float32(0)}
    env.requested_force_reference = lambda ps, info: jnp.float32(0)
    env._task_reached = lambda ps, info: jnp.bool_(False)
    env._has_fallen = MethodType(HumanoidBoxPushEnv._has_fallen, env)
    env._walk_velocity_target = MethodType(HumanoidBoxPushEnv._walk_velocity_target, env)
    env._walk_gait_error = lambda ps, info: jnp.asarray([.01, -.02])
    info.update(box_goal_x=jnp.float32(1.5), box_x0=jnp.float32(1.),
                walk_body_progress=jnp.float32(.3), walk_support_progress=jnp.float32(.2))
    pelvis_q = jnp.asarray([np.cos(.15), np.sin(.15), 0., 0.], jnp.float32)
    torso_q = jnp.asarray([np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)], jnp.float32)
    PS = namedtuple("WalkObjectivePhysics", "x xd qvel site_xpos")
    ps = PS(
        x=Transform(pos=jnp.asarray([[0., 0., 1.], [0., 0., .95], [1., 0., .5]]),
                    rot=jnp.stack([pelvis_q, torso_q, jnp.asarray([1., 0., 0., 0.])])),
        xd=Motion(ang=jnp.asarray([[0., 0., 0.], [.7, -.4, 1.1], [0., 0., 0.]]),
                  vel=jnp.zeros((3, 3))),
        qvel=jnp.zeros(19), site_xpos=jnp.asarray([[0., .2, 0.], [0., -.2, 0.]]),
    )
    reward = jax.jit(lambda state, memory: HumanoidBoxPushEnv._reward_done(
        env, state, jnp.zeros(23), memory)[0])
    for step, target in ((0, 0.), (50, .125), (100, .25), (150, .25)):
        at = {**info, "step": jnp.int32(step)}
        np.testing.assert_allclose(env._walk_velocity_target(at), [target, 0.], atol=2e-7)
        values = []
        for lateral in (-.3, .3):
            body_v = jnp.asarray([target + .1, lateral, 0.])
            world_v = bmath.rotate(body_v, torso_q)
            state = ps._replace(xd=ps.xd.replace(vel=ps.xd.vel.at[1].set(world_v)))
            actual = reward(state, at)
            expected = (1. - 5. * (.01 ** 2 + .02 ** 2) - (.1 ** 2 + .3 ** 2)
                        - .01 * (2. - 2. * np.cos(.3)) - .5 * (.95 - 1.2) ** 2
                        - .1 * (np.pi / 2) ** 2 - (np.pi / 180.) ** 2 * 1.1 ** 2)
            np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-6)
            values.append(actual)
        np.testing.assert_allclose(values[0], values[1], atol=2e-7)
    # The new parameter values are completely inactive in legacy arithmetic.
    cfg.walk_objective_mode = "legacy"
    legacy = jax.jit(lambda: HumanoidBoxPushEnv._reward_done(env, ps, jnp.zeros(23), info))()
    cfg.walk_velocity_ramp_time, cfg.walk_height_target, cfg.w_walk_effort = 7., 4., .9
    unchanged = jax.jit(lambda: HumanoidBoxPushEnv._reward_done(env, ps, jnp.zeros(23), info))()
    for actual, expected in zip(unchanged, legacy):
        np.testing.assert_array_equal(actual, expected)

    # DIAL's H1 push-crate objective uses a bounded unwanted-contact count.
    # P4 keeps a force-aware reward, but one extreme candidate must not set
    # the scale of the whole sampling batch.  The legacy tasks retain their
    # original unbounded arithmetic; physical safety margins are independent.
    for name in vars(cfg):
        if name.startswith("w_"):
            setattr(cfg, name, 0.)
    cfg.w_nonhand = 10.
    env._box_contact_forces = lambda ps: {
        "hand": jnp.float32(0), "nonhand": jnp.float32(120),
    }
    cfg.walk_objective_mode = "dial"
    bounded = HumanoidBoxPushEnv._reward_done(env, ps, jnp.zeros(23), info)[0]
    np.testing.assert_allclose(bounded, -10., atol=0.)
    cfg.walk_objective_mode = "legacy"
    unbounded = HumanoidBoxPushEnv._reward_done(env, ps, jnp.zeros(23), info)[0]
    np.testing.assert_allclose(unbounded, -40., atol=0.)


def test_h1_dial_walk_effort_uses_pre_state_and_preserves_success_padding():
    env, state = _h1_terminal_array_fixture()
    cfg = env._bcfg
    cfg.walk_objective_mode, cfg.w_walk_effort = "dial", .01
    cfg.level, cfg.walk_leg_control, cfg.walk_success_mode = "push_walk", "joint_target", "locomotion"
    env._walk_requires_locomotion = True
    env._walk_progress = lambda ps, info: {}
    env.joint_torque_range = jnp.tile(jnp.asarray([-40., 40.]), (19, 1))
    # The post-state box advances by .05: evaluating there would give 25,
    # rather than the first actually applied pre-state torque of 20.
    env._control = lambda ps, u, info: jnp.full(19, 20. + 100. * ps.x.pos[1, 0])
    np.testing.assert_allclose(env._walk_effort_penalty(state.pipeline_state, jnp.zeros(2), state.info),
                               -.0475, rtol=0., atol=1e-8)
    completed = jax.jit(env.step)(state, jnp.asarray([.05, 0.]))
    np.testing.assert_allclose(completed.reward, .3 - .0475, atol=1e-7)
    assert bool(completed.info["task_success"])
    padding = jax.jit(env.step)(completed, jnp.asarray([1., -1.]))
    np.testing.assert_array_equal(padding.pipeline_state.x.pos, completed.pipeline_state.x.pos)
    np.testing.assert_allclose(padding.reward, env._success_padding_reward(), atol=0.)
    assert bool(padding.info["success_padding"])
    # Config is static when traced. Use an independent legacy environment,
    # rather than mutating the already-jitted dial closure in place.
    legacy_env, legacy_state = _h1_terminal_array_fixture()
    legacy_env._bcfg.w_walk_effort = .01
    def forbidden_control(*args):
        raise AssertionError("legacy reward must not evaluate extra control")
    legacy_env._control = forbidden_control
    legacy = jax.jit(legacy_env.step)(legacy_state, jnp.asarray([.05, 0.]))
    np.testing.assert_allclose(legacy.reward, .3, atol=0.)


def test_h1_dial_walk_objective_keeps_91d_and_invalidates_old_policy_contract():
    import json
    import pytest
    from types import SimpleNamespace as NS
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.experiments.plugins.methods.contact_receding import _validate_policy_interface

    observed_env, ps, observed_cfg = _joint_target_observation_fixture()
    observed_cfg.walk_success_mode = "locomotion"
    legacy_obs = HumanoidBoxPushEnv._get_obs(observed_env, ps, {"step": 50})
    observed_cfg.walk_objective_mode, observed_cfg.walk_velocity_ramp_time = "dial", 2.
    for step, fraction in ((0, 0.), (25, .25), (50, .5), (100, 1.), (150, 1.)):
        obs = HumanoidBoxPushEnv._get_obs(observed_env, ps, {"step": step})
        assert obs.shape == (91,)
        np.testing.assert_allclose(obs[-2], fraction, atol=1e-7)
    changed_obs = HumanoidBoxPushEnv._get_obs(observed_env, ps, {"step": 50})
    np.testing.assert_array_equal(changed_obs[:-2], legacy_obs[:-2])
    np.testing.assert_array_equal(changed_obs[-1], legacy_obs[-1])
    assert float(legacy_obs[-2]) == 1. and float(changed_obs[-2]) == .5
    profile, cfg, _, wbc, _ = _joint_target_array_fixture()
    env = NS(_is_walk=True, _bcfg=cfg, _robot_profile=profile, _whole_body_controller=wbc,
             spec=PrimitiveSpec(pos_dim=5, stiff_dim=3, feed_dim=1),
             action_size=23, dt=.02, sys=NS(nq=29, nv=28), _gait="slow_walk",
             _gait_params={"slow_walk": jnp.asarray([.6, .8, .15])})
    old_contract = HumanoidBoxPushEnv.policy_interface.fget(env)
    cfg.walk_objective_mode, cfg.walk_velocity_ramp_time, cfg.walk_height_target = "dial", 2., 1.2
    cfg.target_vx, cfg.w_gait, cfg.w_vel, cfg.w_upright = .25, 5., 1., .01
    cfg.w_height, cfg.w_yaw, cfg.w_angvel, cfg.w_walk_effort = .5, .1, (np.pi / 180.) ** 2, .01
    contract = HumanoidBoxPushEnv.policy_interface.fget(env)
    assert json.loads(json.dumps(contract)) == contract
    assert contract["observation_layout"]["observation_size"] == 91
    assert contract["task"]["memory_contract"]["startup_elapsed_cap"] == 2.
    assert contract["task"]["locomotion_objective"]["target_vx"] == .25
    assert contract["action_layout"] == old_contract["action_layout"]
    _validate_policy_interface(NS(policy_interface=contract), {"policy_interface": contract})
    with pytest.raises(ValueError, match="policy interface mismatch"):
        _validate_policy_interface(NS(policy_interface=contract), {"policy_interface": old_contract})
    cfg.box_mass, cfg.mu_hand, cfg.dr_seed = 99., .2, 111
    assert HumanoidBoxPushEnv.policy_interface.fget(env) == contract
    cfg.walk_objective_mode = "legacy"
    assert HumanoidBoxPushEnv.policy_interface.fget(env) == old_contract


def test_h1_dial_walk_objective_real_reset_contract_without_steps():
    env = make_env(
        HUMANOID_TASK, robot="h1", level="push_walk", walk_leg_control="joint_target",
        walk_success_mode="locomotion", walk_objective_mode="dial", gait="slow_walk",
        walk_gait_reference="legacy", target_vx=.25, walk_velocity_ramp_time=2.,
        walk_height_target=1.2, w_gait=5., w_vel=1., w_upright=.01,
        w_height=.5, w_yaw=.1, w_angvel=(np.pi / 180.) ** 2, w_walk_effort=.01,
    )
    state = env.reset(jax.random.PRNGKey(110))
    assert state.obs.shape == (91,)
    assert env.policy_interface["task"]["memory_contract"]["startup_elapsed_cap"] == 2.
    assert env.policy_interface["task"]["locomotion_objective"]["target_vx"] == .25
    np.testing.assert_array_equal(env._walk_velocity_target(state.info), [0., 0.])
    assert float(state.obs[-2]) == 0.


def _h1_measured_walk_array_fixture():
    """Small post-physics states for the opt-in contact-event contract."""
    from collections import namedtuple
    from types import MethodType, SimpleNamespace as NS
    from brax.base import Motion, Transform
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    cfg = HumanoidBoxPushConfig(
        level="push_walk", walk_leg_control="joint_target",
        walk_success_mode="locomotion", gait="slow_walk",
    )
    env = NS(_bcfg=cfg, _is_walk=True, _walk_requires_locomotion=True,
             _feet_site_id=jnp.asarray([0, 1]), _pelvis_idx=1, _torso_idx=1,
             _box_idx=2, _torso_z0=0.98, dt=0.02, _gait="slow_walk",
             _gait_params={"slow_walk": jnp.asarray([0.6, 0.8, 0.15])},
             _gait_phase={"slow_walk": jnp.asarray([0.0, 0.5])})
    env._box_reached = lambda ps, info: jnp.bool_(True)
    env._walk_progress = MethodType(HumanoidBoxPushEnv._walk_progress, env)
    env._walk_foot_target = MethodType(HumanoidBoxPushEnv._walk_foot_target, env)
    env._walk_contact_diagnostics = lambda ps: {
        "foot_normal_loads": ps.loads, "foot_floor_clearance": ps.clearance,
        "foot_ground_contact": ps.clearance <= 0.0,
    }
    env._foot_contact_loads = lambda ps: ps.loads
    ContactState = namedtuple("MeasuredWalkingState", "site_xpos x xd loads clearance")

    def pose(x=(0.0, 0.0), site_z=(0.0, 0.0), loads=(250.0, 250.0),
             clearance=(-0.001, -0.001), body=0.25, height=0.98):
        return ContactState(
            site_xpos=jnp.asarray([[x[0], 0.2, site_z[0]], [x[1], -0.2, site_z[1]]]),
            x=Transform(pos=jnp.asarray([[body, 0.0, height], [1.0, 0.0, 0.5]]),
                        rot=jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1))),
            xd=Motion(ang=jnp.zeros((2, 3)), vel=jnp.zeros((2, 3))),
            loads=jnp.asarray(loads), clearance=jnp.asarray(clearance),
        )

    info = {
        "step": jnp.int32(0), "walk_foot_z0": jnp.zeros(2),
        "walk_body_x0": jnp.float32(0), "walk_support_x0": jnp.float32(0),
        "walk_landing_x": jnp.zeros(2), "walk_forward_steps": jnp.zeros(2, jnp.int32),
        "walk_swing_seen": jnp.zeros(2, jnp.bool_),
        "walk_foot_loaded": jnp.ones(2, jnp.bool_),
        "walk_swing_eligible": jnp.zeros(2, jnp.bool_),
        "walk_goal_hold_time": jnp.float32(0),
    }
    return env, pose, info


def test_h1_joint_target_gait_error_tracks_clearance_with_legacy_site_prefix():
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, pose, info = _h1_measured_walk_array_fixture()
    ps = pose(site_z=(0.09, 0.12), clearance=(-0.001, 0.025), loads=(300.0, 0.0))
    error = jax.jit(lambda ps, info: HumanoidBoxPushEnv._walk_gait_error(env, ps, info))
    expected = jax.jit(lambda ps, info: env._walk_foot_target(info) - ps.clearance)
    for step in (0, 31):
        at = {**info, "step": jnp.int32(step)}
        np.testing.assert_array_equal(error(ps, at), expected(ps, at))
    # Only the new joint-target locomotion contract uses measured clearance.
    for mode, strict in (("legacy", True), ("support_phase", True), ("joint_target", False)):
        env._bcfg.walk_leg_control = mode
        env._bcfg.walk_success_mode = "locomotion" if strict else "legacy"
        env._walk_requires_locomotion = strict
        legacy = jax.jit(lambda: HumanoidBoxPushEnv._walk_gait_error(env, ps, info))()
        expected_legacy = jax.jit(lambda: env._walk_foot_target(info) - ps.site_xpos[:, 2])()
        np.testing.assert_array_equal(legacy, expected_legacy)


def test_h1_joint_target_cpg_gait_error_removes_nonzero_site_home_only_for_clearance():
    from types import MethodType
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, pose, info = _h1_measured_walk_array_fixture()
    env._bcfg.walk_gait_reference = "cpg"
    env._feet_home = jnp.asarray([[0.0, 0.2, 0.035], [0.0, -0.2, 0.055]])
    env._walk_phases = MethodType(HumanoidBoxPushEnv._walk_phases, env)
    ps = pose(site_z=(0.09, 0.12), clearance=(-0.001, 0.025))
    error = jax.jit(lambda ps, info: HumanoidBoxPushEnv._walk_gait_error(env, ps, info))
    expected = jax.jit(lambda ps, info: (
        env._walk_foot_target(info) - env._feet_home[:, 2] - ps.clearance
    ))
    for step in (0, 31):
        at = {**info, "step": jnp.int32(step)}
        np.testing.assert_array_equal(error(ps, at), expected(ps, at))
    np.testing.assert_allclose(error(ps, info), -ps.clearance, atol=1e-8)
    # Non-strict and non-joint-target CPG modes still compare site heights,
    # including the old per-foot home offset, without an arithmetic change.
    for mode, strict in (("legacy", True), ("support_phase", True), ("joint_target", False)):
        env._bcfg.walk_leg_control = mode
        env._bcfg.walk_success_mode = "locomotion" if strict else "legacy"
        env._walk_requires_locomotion = strict
        actual = jax.jit(lambda: HumanoidBoxPushEnv._walk_gait_error(env, ps, info))()
        legacy = jax.jit(lambda: env._walk_foot_target(info) - ps.site_xpos[:, 2])()
        np.testing.assert_array_equal(actual, legacy)


def test_h1_joint_target_manifold_uses_clearance_without_changing_balance_inequalities():
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, pose, info = _h1_measured_walk_array_fixture()
    env._align_yaw, env._mu = 1.0, 0.6
    env._feet_home = jnp.asarray([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])
    env._walk_nominal_stance = lambda info: jnp.asarray([True, False])
    env._balance_safety_residual = lambda ps: jnp.float32(-0.1)
    env._hand_contact = lambda ps, action, info: {
        "p_hand": jnp.zeros(3), "p_c": jnp.zeros(3),
        "push_axis": jnp.asarray([1.0, 0.0, 0.0]),
        "n_c": jnp.asarray([-1.0, 0.0, 0.0]),
        "f_t": jnp.zeros(3), "f_n": jnp.float32(0),
    }
    ps = pose(site_z=(0.09, 0.12), clearance=(-0.001, 0.025), loads=(300.0, 0.0))
    direct_h, direct_g = jax.jit(lambda: HumanoidBoxPushEnv._manifold(env, ps, jnp.zeros(23), info))()
    np.testing.assert_allclose(direct_h[-2:], [-0.001, 0.0], atol=1e-8)
    env._bcfg.walk_leg_control = "legacy"
    legacy_h, legacy_g = jax.jit(lambda: HumanoidBoxPushEnv._manifold(env, ps, jnp.zeros(23), info))()
    np.testing.assert_allclose(legacy_h[-2:], [0.09, 0.0], atol=1e-8)
    np.testing.assert_array_equal(direct_h[:-2], legacy_h[:-2])
    np.testing.assert_array_equal(direct_g, legacy_g)


def test_h1_measured_walk_counts_real_steps_not_toe_roll_sliding_or_in_place():
    env, pose, info = _h1_measured_walk_array_fixture()
    update = jax.jit(lambda ps, info: {**info, **env._walk_progress(ps, info)})
    # A loaded toe/heel pivot can raise the site without lifting the capsule.
    info = update(pose(x=(0.12, 0.0), site_z=(0.09, 0.0)), info)
    np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])
    np.testing.assert_array_equal(info["walk_swing_eligible"], [False, False])
    np.testing.assert_array_equal(info["walk_forward_steps"], [0, 0])
    np.testing.assert_array_equal(info["walk_landing_x"], [0.0, 0.0])
    assert float(info["walk_support_progress"]) == 0.0
    # A torso far ahead of stationary feet receives only one bounded
    # potential increment; repeating the same forward-lean cannot farm it.
    np.testing.assert_allclose(info["walk_progress_potential"], env._bcfg.walk_step_min_distance)
    repeated = update(pose(x=(0.12, 0.0), site_z=(0.09, 0.0)), info)
    np.testing.assert_allclose(repeated["walk_progress_delta"], 0.0, atol=1e-8)
    assert float(repeated["walk_new_steps"]) == 0.0
    # Release is observed before the old 2 cm true-clearance threshold.
    info = update(pose(loads=(0.0, 250.0), clearance=(0.005, -0.001)), info)
    np.testing.assert_array_equal(info["walk_swing_eligible"], [True, False])
    np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])
    info = update(pose(x=(0.5, 0.0), loads=(0.0, 250.0), clearance=(0.02, -0.001)), info)
    np.testing.assert_array_equal(info["walk_swing_seen"], [True, False])
    assert float(info["walk_support_progress"]) == 0.0  # an airborne reach is not support advance
    info = update(pose(x=(0.2, 0.0), site_z=(0.08, 0.0)), info)
    np.testing.assert_array_equal(info["walk_forward_steps"], [1, 0])
    np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])
    np.testing.assert_allclose(info["walk_landing_x"], [0.2, 0.0])
    np.testing.assert_allclose(info["walk_support_progress"], 0.1)
    info = update(pose(x=(0.4, 0.0)), info)  # loaded sliding neither counts nor updates landing
    np.testing.assert_array_equal(info["walk_forward_steps"], [1, 0])
    np.testing.assert_allclose(info["walk_support_progress"], 0.1)
    # An in-place right lift/recontact updates contact history but not the count.
    info = update(pose(x=(0.4, 0.0), loads=(250.0, 0.0), clearance=(-0.001, 0.02)), info)
    info = update(pose(x=(0.4, 0.0)), info)
    np.testing.assert_array_equal(info["walk_forward_steps"], [1, 0])
    np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])


def test_h1_measured_walk_requires_release_edge_and_cancels_full_air_without_timer():
    env, pose, info = _h1_measured_walk_array_fixture()
    update = jax.jit(lambda ps, info: {**info, **env._walk_progress(ps, info)})
    # Starting with an already airborne foot cannot retroactively create a
    # supported release when the other foot happens to acquire ground contact.
    info["walk_foot_loaded"] = jnp.zeros(2, jnp.bool_)
    info = update(pose(loads=(0.0, 250.0), clearance=(0.03, -0.001)), info)
    np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])
    np.testing.assert_array_equal(info["walk_swing_eligible"], [False, False])
    info = update(pose(), info)
    for lift_gap in (0.005, 0.03):  # cancel both pending lift and already observed swing
        info = update(pose(loads=(0.0, 250.0), clearance=(lift_gap, -0.001)), info)
        assert bool(info["walk_swing_eligible"][0])
        info = update(pose(loads=(0.0, 0.0), clearance=(0.04, 0.02)), info)
        np.testing.assert_array_equal(info["walk_swing_eligible"], [False, False])
        np.testing.assert_array_equal(info["walk_swing_seen"], [False, False])
        info = update(pose(loads=(0.0, 250.0), clearance=(0.04, -0.001)), info)
        info = update(pose(x=(0.2, 0.0)), info)
        np.testing.assert_array_equal(info["walk_forward_steps"], [0, 0])
    # The agreed numerical load threshold is strictly zero, not a new N- or
    # body-weight-scaled deadband: any finite positive compression is loaded.
    info = update(pose(x=(0.2, 0.0), loads=(1e-7, 1e-7)), info)
    np.testing.assert_array_equal(info["walk_foot_loaded"], [True, True])
    info = update(pose(x=(0.2, 0.0), loads=(0.0, 1e-7), clearance=(0.02, -0.001)), info)
    assert bool(info["walk_swing_seen"][0])
    info = update(pose(x=(0.3, 0.0), loads=(1e-7, 1e-7)), info)
    np.testing.assert_array_equal(info["walk_forward_steps"], [1, 0])


def test_h1_measured_walk_counts_backward_backward_forward_as_zero_one():
    """Matched-load evidence reduced to pure arrays; no results-file dependency."""
    env, pose, info = _h1_measured_walk_array_fixture()
    origin = jnp.float32(0.059468016)
    info["walk_landing_x"] = jnp.full(2, origin)
    info["walk_support_x0"] = origin
    update = jax.jit(lambda ps, info: {**info, **env._walk_progress(ps, info)})
    frames = [
        # Right: observed release, physical lift, then a backward landing.
        ((.059468, 0.0), (300., 0.), (-.001, .001)),
        ((.059468, -.024635), (321., 0.), (-.001229, .023463)),
        ((.059468, -.201518), (195., 616.), (-.001518, -.008826)),
        # Left: the next supported exchange also lands backward.
        ((0.0, -.201518), (0., 397.), (.0002, -.00647)),
        ((-.239042, -.201518), (0., 227.), (.032449, -.002424)),
        ((-.2467386, -.201518), (204., 323.), (-.001269, -.002055)),
        # Right: true forward landing, although the tilted site remains high.
        ((-.23, -.17), (309., 0.), (-.00258, .00037)),
        ((-.23, -.095267), (124., 0.), (-.001255, .027374)),
        ((-.23, .084472), (143., 302.), (-.000819, -.003055)),
    ]
    for x, loads, clearance in frames:
        info = update(pose(x=x, site_z=(.06, .08766), loads=loads, clearance=clearance), info)
    np.testing.assert_array_equal(info["walk_forward_steps"], [0, 1])
    np.testing.assert_allclose(info["walk_landing_x"], [-.2467386, .084472], atol=1e-7)
    np.testing.assert_allclose(info["walk_support_progress"], (-.2467386 + .084472) / 2 - origin,
                               atol=1e-7)
    assert not bool(info["walk_goal_ready"])  # one partial recovery is not completed walking
    # Later full-air stumbling cannot revive that completed swing latch.
    info = update(pose(loads=(0., 37.), clearance=(.00009, -.00025)), info)
    info = update(pose(loads=(0., 0.), clearance=(.00119, .00366), height=.8056), info)
    info = update(pose(x=(-.179288, .17), loads=(0., 72.), clearance=(.028138, -.000548),
                       height=.6244), info)
    info = update(pose(x=(-.094949, .22), loads=(32., 0.), clearance=(-.001686, .003908),
                       height=.6405), info)
    np.testing.assert_array_equal(info["walk_forward_steps"], [0, 1])


def _walk_locomotion_contract():
    """Reaching the line, stance sliding, or in-place foot lifts are not walking."""
    env = make_env(
        HUMANOID_TASK, level="push_walk", push_dist=0.50, walk_success_mode="locomotion",
        walk_gait_reference="cpg", gait_hip_forward_sign=-1.0,
        stance_force_ankle_gain=-1.5, stance_force_hip_gain=0.007,
        stance_force_hip_deadband=15.0,
    )
    state = env.reset(jax.random.PRNGKey(0))
    ps0 = state.pipeline_state
    task_start = ps0.qpos.size + ps0.qvel.size
    np.testing.assert_allclose(state.obs[task_start + 4], 0.50, atol=1e-6)
    feet0 = ps0.site_xpos[env._feet_site_id]

    def pose(dx=(0.0, 0.0), height=(0.0, 0.0), body=0.25, speed=0.0):
        feet = feet0.at[:, 0].add(jnp.asarray(dx)).at[:, 2].add(jnp.asarray(height))
        pos = ps0.x.pos.at[env._box_idx - 1, 0].set(state.info["box_goal_x"])
        pos = pos.at[env._pelvis_idx - 1, 0].add(body)
        vel = ps0.xd.vel.at[env._box_idx - 1, 0].set(speed)
        return ps0.replace(
            site_xpos=ps0.site_xpos.at[env._feet_site_id].set(feet),
            x=ps0.x.replace(pos=pos), xd=ps0.xd.replace(vel=vel),
        )

    info = dict(state.info)
    at_line = pose(body=0.0)
    info.update(env._walk_progress(at_line, info))
    assert not bool(env._task_reached(at_line, info))
    slipped = pose(dx=(0.20, 0.20))
    info.update(env._walk_progress(slipped, info))
    assert not bool(jnp.any(info["walk_forward_steps"]))
    assert not bool(env._task_reached(slipped, info))
    info = dict(state.info)
    info.update(env._walk_progress(pose(dx=(0.20, 0.20), height=(0.03, 0.03)), info))
    info.update(env._walk_progress(pose(dx=(0.20, 0.20)), info))
    assert not bool(jnp.any(info["walk_forward_steps"]))
    # A lift and return at the same x also cannot count as a forward step.
    info = dict(state.info)
    info.update(env._walk_progress(pose(height=(0.03, 0.0)), info))
    info.update(env._walk_progress(pose(), info))
    assert not bool(jnp.any(info["walk_forward_steps"]))
    for feet_dx, feet_height in (
        ((0.20, 0.0), (0.03, 0.0)),
        ((0.20, 0.0), (0.0, 0.0)),
        ((0.20, 0.20), (0.0, 0.03)),
        ((0.20, 0.20), (0.0, 0.0)),
    ):
        ps = pose(dx=feet_dx, height=feet_height)
        info.update(env._walk_progress(ps, info))
    assert np.array_equal(np.asarray(info["walk_forward_steps"]), [1, 1])
    assert not bool(env._task_reached(ps, info))
    for _ in range(9):
        info.update(env._walk_progress(ps, info))
    assert bool(env._task_reached(ps, info))
    moving = pose(dx=(0.20, 0.20), speed=0.20)
    info.update(env._walk_progress(moving, info))
    assert not bool(env._task_reached(moving, info))
    assert float(info["walk_goal_hold_time"]) == 0.0
    assert float(env.walk_force_scale(at_line, info)) == 0.0
    assert float(env.walk_force_scale(ps0, info)) == 1.0
    assert float(env.requested_force_reference(at_line, {**info, "step": 100})) == 0.0
    # The CPG reference starts at reset foot height, without the legacy jog phase.
    np.testing.assert_allclose(env._walk_foot_target(state.info), env._feet_home[:, 2])
    # At phase zero only the left foot is in swing.  Its ground equality
    # must be inactive while the right stance foot retains its equality.
    both_lifted = pose(height=(0.03, 0.03))
    h, _ = env._manifold(both_lifted, jnp.zeros(env.action_size), state.info)
    np.testing.assert_allclose(np.asarray(h[-2:]), [0.0, 0.03], atol=1e-6)
    # A requested 30N with no physical hand contact must not trigger ankle
    # bracing.  The optional support_load key leaves the legacy contract intact.
    command_info = {**state.info, "step": 100, "contact_acquired": 1.0, "contact_step": 0}
    action = jnp.zeros(env.action_size)
    contact = env._hand_contact(ps0, action, command_info)
    assert float(contact["F_n"]) == 30.0
    assert float(contact["support_load"]) == 0.0
    wbc = env._whole_body_controller
    args = (action, command_info, env.spec.total_width)
    measured_tau = wbc.torque(ps0, contact, *args)
    legacy_contact = {k: v for k, v in contact.items() if k != "support_load"}
    legacy_tau = wbc.torque(ps0, legacy_contact, *args)
    zero_load_tau = wbc.torque(ps0, {**legacy_contact, "F_n": jnp.float32(0.0)}, *args)
    np.testing.assert_allclose(measured_tau, zero_load_tau, atol=1e-5)
    assert not np.allclose(measured_tau[:env._n_planner], legacy_tau[:env._n_planner])
    print("  locomotion: box-only/slide/in-place rejected; forward steps+dwell required -> True")
    return True


def test_h1_measured_support_is_scoped_to_strict_locomotion():
    """Fixed stance blends low-load realization with high-load anticipation."""
    env = make_env(
        HUMANOID_TASK, level="push_to_line", f_target=15.0,
        fixed_force_target=True,
        stance_force_ankle_gain=-1.5, stance_force_hip_gain=0.007,
        stance_force_hip_deadband=15.0,
    )
    state = env.reset(jax.random.PRNGKey(101))
    action = jnp.zeros(env.action_size)
    info = {**state.info, "step": 100, "contact_acquired": 1.0, "contact_step": 0}
    ps = state.pipeline_state
    contact = env._hand_contact(ps, action, info)
    assert float(contact["F_n"]) == 15.0
    assert float(contact["support_load"]) == 0.0
    wbc = env._whole_body_controller
    args = (action, info, env.spec.total_width)
    commanded_tau = wbc.torque(ps, contact, *args)
    zero_tau = wbc.torque(ps, {**contact, "F_n": jnp.float32(0.0)}, *args)
    np.testing.assert_allclose(commanded_tau, zero_tau, rtol=0.0, atol=1e-5)

    original_measurement = env._box_contact_force
    try:
        env._box_contact_force = lambda ps: jnp.float32(12.0)
        assert float(env._hand_contact(ps, action, info)["support_load"]) == 12.0
        env._bcfg.f_target = 22.5
        env._box_contact_force = lambda ps: jnp.float32(0.0)
        assert float(env._hand_contact(ps, action, info)["support_load"]) == pytest.approx(11.25)
        env._bcfg.f_target = 30.0
        assert float(env._hand_contact(ps, action, info)["support_load"]) == 30.0
    finally:
        env._box_contact_force = original_measurement

    # The old box-only P4 protocol is not silently changed by this repair.
    legacy_walk = make_env(HUMANOID_TASK, level="push_walk")
    walk_state = legacy_walk.reset(jax.random.PRNGKey(101))
    assert not legacy_walk._walk_requires_locomotion
    assert "support_load" not in legacy_walk._hand_contact(
        walk_state.pipeline_state, jnp.zeros(legacy_walk.action_size), walk_state.info,
    )


def _emergency_isotropic_unload():
    env = make_env(HUMANOID_TASK, level="push_to_line")
    state = env.reset(jax.random.PRNGKey(0))
    nodes = jnp.ones((3, env.action_size))
    emergency = env.emergency_plan(state, nodes)
    normal = np.asarray(env._stiffness(jnp.zeros((6,))))
    unload = np.asarray(env._stiffness(emergency[0, env.spec.s_slice]))
    np.testing.assert_allclose(unload, np.eye(3) * unload[0, 0], atol=1e-5)
    # H1 preserves the stiffness already applied at entry; cold prev_action=0
    # means nominal K. A commanded zero force is not a physical safety claim.
    np.testing.assert_array_equal(unload, normal)
    prepared = env._prepare_mga_execution_state(state, 1)
    np.testing.assert_array_equal(env.realized_hand_stiffness(prepared, emergency[0]), normal)
    assert float(env._force_cmd(emergency[0, env.spec.nu_slice][0])) == 0.0
    print("  emergency: zero force command and held entry stiffness -> True")
    return True


def _joint_target_array_fixture():
    """Real H1 profile plus small arrays; no simulator construction or steps."""
    from types import SimpleNamespace as NS
    from genedynamics.core.control.humanoid_contact import HumanoidWholeBodyController
    from genedynamics.robots.h1.profile import h1_profile

    profile = h1_profile()
    defaults = profile.controller_defaults
    home = jnp.asarray(defaults["home_qpos"][7:], jnp.float32)
    cfg = NS(walk_objective_mode="legacy", walk_box_goal_mode="position",
             walk_force_startup_mode="legacy",
             walk_leg_control="joint_target", leg_grav_comp=0.,
             f_target=30., stance_force_hip_deadband=15., dt=0.02, timestep=0.004,
             target_vx=0.15, gait_ramp_time=0.5, stance_hip_bias=-0.2,
             stance_ankle_bias=-0.2, stance_force_hip_gain=0.007, leg_scale=0.1,
             stance_force_reference=30., stance_force_ankle_gain=-1.5,
             stance_com_ankle_gain=375., stance_com_ankle_damping=45.,
             arm_null_damping=0.01, arm_posture_kp=2., arm_posture_kd=0.2,
             arm_grav_comp=1., use_base=False, stiffness_mode="log_spd", level="push_walk",
             walk_success_mode="locomotion", gait="slow_walk", walk_gait_reference="legacy",
             gait_swing_frac=0.45, gait_cadence=0.8, approach_time=0.3, force_ramp_time=0.3,
             walk_lift_height=0.02, walk_touchdown_height=0.005, walk_step_min_distance=0.04,
             walk_min_steps_per_foot=1, walk_success_hold_time=0.2, walk_min_body_progress=0.2,
             walk_min_support_progress=0.15, walk_success_max_box_speed=0.08,
             walk_safety_min_torso_up=0.9, walk_safety_min_height_ratio=0.7,
             policy_joint_reference_residual_scale=0.2, goal_eps=0.02)
    ps = NS(qpos=home, qvel=jnp.linspace(-0.1, 0.1, 19),
            qfrc_bias=jnp.linspace(1., 2., 19),
            x=NS(pos=jnp.array([[0.3, 0.1, 1.0]])),
            xd=NS(vel=jnp.array([[0.1, 0., 0.]])),
            site_xpos=jnp.array([[0., 0.2, 0.], [0., -0.2, 0.]]))
    def forbidden_gait(*args):
        raise AssertionError("joint_target must not evaluate the CPG")
    wbc = HumanoidWholeBodyController(
        binding=NS(qpos_indices=tuple(range(19)), dof_indices=tuple(range(19))),
        config=cfg, default_pose=home, stance_pose=home,
        kp=jnp.asarray(defaults["kp"]), kd=jnp.asarray(defaults["kd"]),
        torque_limits=jnp.full(19, 40.), arm_push_pose=home,
        left_arm=profile.joint_groups["left_arm"], right_arm=profile.joint_groups["right_arm"],
        sagittal_legs=(profile.joint_groups["left_sagittal_leg"],
                       profile.joint_groups["right_sagittal_leg"]),
        n_planner=11, pelvis_body_id=1, feet_site_ids=jnp.array([0, 1]),
        stance_com_x0=-0.06, is_walk=True, gait_controller=forbidden_gait,
        joint_limits=jnp.tile(jnp.array([-3., 3.]), (19, 1)),
        planner_joint_bounds=defaults["planner_joint_bounds"],
    )
    right = jnp.zeros((19, 3)).at[15:18].set(jnp.eye(3))
    left = jnp.zeros((19, 3)).at[11:14].set(jnp.eye(3))
    contact = {"F_n": jnp.float32(30.), "jacp": right,
               "wrench": jnp.array([2., -1., 3.]),
               "left": {"jacp": left, "wrench": jnp.array([-2., 1., 4.])}}
    return profile, cfg, ps, wbc, contact


def test_h1_joint_target_affine_pd_and_bounds_contract():
    """DIAL chart and leg PD, with independent FF and unchanged hand impedance."""
    from dataclasses import replace
    from types import SimpleNamespace as NS
    import pytest

    _, cfg, ps, wbc, contact = _joint_target_array_fixture()
    bounds = np.asarray(wbc.planner_joint_bounds)
    info = {"step": 50}
    target = jax.jit(lambda u: wbc.joint_targets(ps, contact, u, info, 12))
    for value, expected in ((-1., bounds[:, 0]), (0., bounds.mean(axis=1)),
                            (1., bounds[:, 1]), (2., bounds[:, 1])):
        np.testing.assert_allclose(target(jnp.full(23, value))[:11], expected, atol=1e-6)
    derivative = jax.jacfwd(target)(jnp.zeros(23))[:11]
    np.testing.assert_allclose(derivative[:, :12], 0., atol=0.)
    np.testing.assert_allclose(derivative[:, 12:], np.diag(np.diff(bounds, axis=1)[:, 0] / 2),
                               atol=1e-6)
    action = jnp.zeros(23).at[12:].set(wbc.planner_action_from_joints(ps.qpos[:11]) + 0.01)
    actual = jax.jit(lambda u: wbc.torque(ps, contact, u, info, 12))(action)
    expected_raw = wbc.kp * (target(action) - ps.qpos) - wbc.kd * ps.qvel
    brace = cfg.stance_force_ankle_gain * cfg.f_target
    for _, _, ankle in wbc.sagittal_legs:
        expected_raw = expected_raw.at[ankle].add(brace)
    expected = jnp.clip(expected_raw, -wbc.torque_limits, wbc.torque_limits)
    np.testing.assert_allclose(actual[:11], expected[:11], rtol=0., atol=1e-6)
    ff = replace(wbc, config=NS(**{**vars(cfg), "leg_grav_comp": 0.5}))
    with_ff = ff.torque(ps, contact, action, info, 12)
    expected_ff = jnp.clip(
        expected_raw.at[:11].add(0.5 * ps.qfrc_bias[:11]),
        -wbc.torque_limits, wbc.torque_limits,
    )
    np.testing.assert_allclose(with_ff[:11], expected_ff[:11], atol=1e-5)
    unloaded = {**contact, "F_n": jnp.float32(0.)}
    unloaded_actual = wbc.torque(ps, unloaded, action, info, 12)
    unloaded_expected = jnp.clip(
        wbc.kp * (target(action) - ps.qpos) - wbc.kd * ps.qvel,
        -wbc.torque_limits, wbc.torque_limits,
    )
    np.testing.assert_allclose(unloaded_actual[:11], unloaded_expected[:11], atol=2e-6)
    reference = jnp.stack([jnp.linspace(-0.4, 0.4, 11),
                           jnp.linspace(0.3, -0.3, 11)])
    referenced = replace(
        wbc,
        planner_action_reference=reference,
        planner_reference_residual_scale=0.05,
    )
    residual = jnp.zeros(23).at[12:].set(0.4)
    referenced_target = referenced.joint_targets(
        ps, contact, residual, {"step": 1}, 12
    )[:11]
    normalized = np.asarray(reference[1]) + 0.05 * 0.4
    expected_reference = (
        bounds[:, 0] + 0.5 * (normalized + 1.0)
        * (bounds[:, 1] - bounds[:, 0])
    )
    np.testing.assert_allclose(referenced_target, expected_reference, atol=1e-6)
    reanchored_target = referenced.joint_targets(
        ps, contact, residual, {"step": 1, "walk_reference_step": 0}, 12
    )[:11]
    normalized_reanchored = np.asarray(reference[0]) + 0.05 * 0.4
    expected_reanchored = (
        bounds[:, 0] + 0.5 * (normalized_reanchored + 1.0)
        * (bounds[:, 1] - bounds[:, 0])
    )
    np.testing.assert_allclose(
        reanchored_target, expected_reanchored, atol=1e-6
    )
    referenced_derivative = jax.jacfwd(
        lambda u: referenced.joint_targets(
            ps, contact, u, {"step": 1}, 12
        )
    )(jnp.zeros(23))[:11, 12:]
    np.testing.assert_allclose(
        referenced_derivative,
        np.diag(0.05 * np.diff(bounds, axis=1)[:, 0] / 2),
        atol=1e-6,
    )
    # The hand wrench/nullspace/gravity path is identical to legacy control.
    legacy = replace(wbc, config=NS(**{**vars(cfg), "walk_leg_control": "legacy"}),
                     gait_controller=lambda *args: jnp.zeros(19))
    legacy_tau = jax.jit(lambda u: legacy.torque(ps, contact, u, info, 12))(action)
    np.testing.assert_array_equal(actual[11:], legacy_tau[11:])
    released = {**contact, "arm_task_scale": jnp.float32(0.)}
    zero_wrench = {
        **contact,
        "wrench": jnp.zeros_like(contact["wrench"]),
        "left": {
            **contact["left"],
            "wrench": jnp.zeros_like(contact["left"]["wrench"]),
        },
    }
    np.testing.assert_allclose(
        wbc.torque(ps, released, action, info, 12)[11:],
        wbc.torque(ps, zero_wrench, action, info, 12)[11:],
        atol=1e-6,
    )
    assert not np.allclose(
        wbc.torque(ps, released, action, info, 12)[11:], actual[11:]
    )
    for invalid in (None, np.zeros((10, 2)), np.zeros((11, 2)),
                    np.full((11, 2), np.nan), np.tile([-4., 4.], (11, 1))):
        with pytest.raises(ValueError, match="planner_joint_bounds"):
            replace(wbc, planner_joint_bounds=invalid)


def test_h1_fixed_stance_shares_whole_body_brace_across_ankles():
    """The total hand-load brace must not be counted once per planted foot."""
    from dataclasses import replace
    from types import SimpleNamespace as NS

    _, cfg, ps, wbc, contact = _joint_target_array_fixture()
    fixed = replace(
        wbc,
        is_walk=False,
        torque_limits=jnp.full_like(wbc.torque_limits, 1.0e6),
        config=NS(**{
            **vars(cfg),
            "walk_leg_control": "legacy",
            "leg_grav_comp": 0.0,
            "stance_com_ankle_gain": 0.0,
            "stance_com_ankle_damping": 0.0,
        }),
    )
    action = jnp.zeros(12)
    no_load = {**contact, "F_n": jnp.float32(0.0)}
    loaded = fixed.torque(ps, contact, action, {"step": 0}, 12)
    unloaded = fixed.torque(ps, no_load, action, {"step": 0}, 12)
    brace = cfg.stance_force_ankle_gain * cfg.f_target
    for _, _, ankle in fixed.sagittal_legs:
        np.testing.assert_allclose(
            loaded[ankle] - unloaded[ankle], 0.5 * brace, atol=1e-6
        )

    # An explicit task-owned support split remains authoritative.
    one_foot = {
        **contact,
        "stance_support_weights": jnp.asarray([1.0, 0.0], jnp.float32),
    }
    explicit = fixed.torque(ps, one_foot, action, {"step": 0}, 12)
    left_ankle = fixed.sagittal_legs[0][2]
    right_ankle = fixed.sagittal_legs[1][2]
    np.testing.assert_allclose(
        explicit[left_ankle] - unloaded[left_ankle], brace, atol=1e-6
    )
    np.testing.assert_allclose(
        explicit[right_ankle] - unloaded[right_ankle], 0.0, atol=1e-6
    )


def test_h1_full_action_walk_reference_consumes_only_joint_coordinates(tmp_path):
    """An unloaded DIAL trace is a gait reference, never a contact reference."""
    import json

    actions = np.zeros((2, 23), dtype=np.float32)
    actions[:, :12] = np.asarray([.4, -.3, .2, -.1, .5, .8, -.7, .6,
                                  -.5, .4, -.3, .2], dtype=np.float32)
    actions[0, 12:] = np.linspace(-.4, .4, 11, dtype=np.float32)
    actions[1, 12:] = np.linspace(.3, -.3, 11, dtype=np.float32)
    path = tmp_path / "dial_actions.json"
    path.write_text(json.dumps({"actions": actions.tolist()}))

    env = make_env(
        HUMANOID_TASK,
        level="push_walk",
        walk_success_mode="locomotion",
        walk_leg_control="joint_target",
        walk_joint_reference_path=str(path),
    )
    np.testing.assert_allclose(env._walk_joint_reference, actions[:, 12:])
    assert not hasattr(env, "_walk_stiffness_reference")
    contract = env.policy_interface["action_layout"]
    assert contract["planner_mapping"] == "dial_reference_plus_bounded_normalized_residual"
    assert "stiffness_mapping" not in contract
    transform = env.policy_action_transform
    assert transform["leg_authority"] == (
        "local_residual_about_shared_low_level_reference"
    )
    np.testing.assert_allclose(
        transform["action_scale"][env.spec.total_width:], 0.2
    )

    state = env.reset(jax.random.PRNGKey(0))
    zero = jnp.zeros((env.action_size,), jnp.float32)
    np.testing.assert_allclose(
        env.realized_hand_stiffness(state, zero),
        env._stiffness(zero[env.spec.s_slice]),
    )


def test_h1_joint_target_initializer_emergency_and_policy_interface():
    """Same 23D width does not hide changed actions; cold and emergency plans use inverse FK coordinates."""
    import json
    import copy
    import pytest
    from types import SimpleNamespace as NS, MethodType
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.experiments.plugins.methods.contact_receding import _validate_policy_interface

    profile, cfg, ps, wbc, _ = _joint_target_array_fixture()
    env = NS(_is_walk=True, _walk_requires_locomotion=True,
             _bcfg=cfg, _n_planner=11, _whole_body_controller=wbc,
             _defaultN=wbc.default_pose,
             # This fixture isolates the joint-target chart.  The physical
             # pelvis/support capture law has its own integration test below;
             # keep it neutral here while exercising the referenced emergency
             # branch that now composes that law.
             _walk_roll_capture_residual=lambda state: jnp.zeros(11),
             _supports_mga_execution_context=False,  # no physical entry geometry in this leg-only fixture
             _robot_binding=NS(qpos_indices=tuple(range(7, 26))), _robot_profile=profile,
             spec=PrimitiveSpec(pos_dim=5, stiff_dim=3, feed_dim=1),
             action_size=23, dt=0.02, sys=NS(nq=29, nv=28), _gait="slow_walk",
             _gait_params={"slow_walk": jnp.array([0.6, 0.8, 0.15])})
    env._initialize_joint_target_plan = MethodType(HumanoidBoxPushEnv._initialize_joint_target_plan, env)
    state = NS(pipeline_state=NS(qpos=jnp.concatenate([
        jnp.array([0., 0., 0.98, 1., 0., 0., 0.]), ps.qpos, jnp.zeros(3),
    ])))
    nodes = jnp.arange(69, dtype=jnp.float32).reshape(3, 23) / 100.
    initialize = HumanoidBoxPushEnv.plan_initializer.fget(env)
    initialized = jax.jit(lambda y: initialize(state, y))(nodes)
    np.testing.assert_array_equal(initialized[:, :12], nodes[:, :12])
    home_command = np.asarray(wbc.planner_action_from_joints(ps.qpos[:11]))
    np.testing.assert_allclose(initialized[:, 12:], np.tile(home_command, (3, 1)), atol=1e-6)
    assert not np.allclose(home_command, 0.0)
    transform = HumanoidBoxPushEnv.policy_action_transform.fget(env)
    assert transform["schema_version"] == 2
    assert transform["leg_authority"] == "complete_robot_planner_joint_bounds"
    np.testing.assert_allclose(transform["action_bias"][:12], 0.0)
    np.testing.assert_allclose(transform["action_bias"][12:], home_command, atol=1e-6)
    scale = np.asarray(transform["action_scale"])[12:]
    np.testing.assert_allclose(scale, 1.0 + np.abs(home_command), atol=1e-6)
    # Residual extrema cover both DIAL normalized endpoints despite the safe
    # home command not generally lying at the interval midpoint.
    np.testing.assert_allclose(
        np.clip(home_command - scale, -1.0, 1.0), -np.ones(11), atol=1e-6)
    np.testing.assert_allclose(
        np.clip(home_command + scale, -1.0, 1.0), np.ones(11), atol=1e-6)
    # An ankle outside DIAL's sampling bounds gets the nearest representable
    # target, not a claim that the physical pose can be held exactly.
    state.pipeline_state.qpos = state.pipeline_state.qpos.at[11].set(-0.85)
    emergency = HumanoidBoxPushEnv.emergency_plan(env, state, nodes)
    np.testing.assert_allclose(emergency[:, 12 + 4], -1.)
    np.testing.assert_allclose(emergency[:, env.spec.nu_slice], -1.)
    np.testing.assert_array_equal(emergency[:, env.spec.s_slice],
                                  np.tile([-1., 0., 0., -1., 0., -1.], (3, 1)))
    # A referenced walking emergency keeps the shared DIAL gait continuous;
    # projecting a measured hold into this deliberately narrow residual chart
    # would saturate many joints and introduce a discontinuous body impulse.
    env._walk_joint_reference = jnp.stack([
        jnp.linspace(-.2, .2, 11),
        jnp.linspace(.2, -.2, 11),
    ])
    env._sag_legs = ((2, 3, 4), (7, 8, 9))
    cfg.walk_joint_reference_residual_scale = .1
    state.info = {"step": jnp.int32(1)}
    referenced = HumanoidBoxPushEnv.emergency_plan(env, state, nodes)
    expected_referenced = np.zeros((3, 11), dtype=np.float32)
    expected_referenced[:, [3, 8]] = 1.0
    np.testing.assert_array_equal(referenced[:, 12:], expected_referenced)
    cfg.emergency_knee_residual_levels = (1.0, 0.0, -0.5)
    env.emergency_plan = MethodType(HumanoidBoxPushEnv.emergency_plan, env)
    bank = HumanoidBoxPushEnv.emergency_plans(env, state, nodes)
    assert bank.shape == (9, 3, 23)
    np.testing.assert_array_equal(bank[0], referenced)
    np.testing.assert_array_equal(
        np.asarray(bank[:, 0, 12 + np.asarray([3, 8])]),
        np.asarray([
            [left, right]
            for left in cfg.emergency_knee_residual_levels
            for right in cfg.emergency_knee_residual_levels
        ]),
    )
    env._walk_joint_reference = None
    contract = HumanoidBoxPushEnv.policy_interface.fget(env)
    assert json.loads(json.dumps(contract)) == contract
    assert contract["action_layout"]["primitive_width"] == 12
    assert len(contract["action_layout"]["planner_joint_names"]) == 11
    assert len(contract["observation_layout"]["task_feature_names"]) == 34
    assert contract["observation_layout"]["observation_size"] == 91
    assert contract["task"]["memory_contract"]["startup_elapsed_cap"] == 0.6
    assert contract["task"]["memory_contract"]["loaded_normal_force_gt"] == 0.0
    old_contract = copy.deepcopy(contract)
    old_contract["observation_layout"]["version"] = "h1_joint_target_qpos_qvel_task21"
    old_contract["observation_layout"]["observation_size"] = 78
    old_contract["observation_layout"]["task_feature_names"] = (
        old_contract["observation_layout"]["task_feature_names"][:21]
    )
    del old_contract["task"]["memory_contract"]
    policy_env = NS(policy_interface=contract)
    _validate_policy_interface(policy_env, {"policy_interface": contract})
    with pytest.raises(ValueError, match="policy interface mismatch"):
        _validate_policy_interface(policy_env, {"policy_interface": old_contract})
    cfg.walk_step_min_distance = 0.05
    changed = HumanoidBoxPushEnv.policy_interface.fget(env)
    assert changed != contract
    assert changed["action_layout"] == contract["action_layout"]
    cfg.walk_step_min_distance = 0.04
    cfg.dr_seed, cfg.box_mass, cfg.mu_hand = 111, 40., 0.3
    assert HumanoidBoxPushEnv.policy_interface.fget(env) == contract
    cfg.walk_success_mode = "legacy"
    old_contract["task"]["walk_success_mode"] = "legacy"
    assert HumanoidBoxPushEnv.policy_interface.fget(env) == old_contract
    cfg.walk_leg_control = "legacy"
    assert HumanoidBoxPushEnv.plan_initializer.fget(env) is None
    assert HumanoidBoxPushEnv.policy_interface.fget(env) is None
    legacy = HumanoidBoxPushEnv.emergency_plan(env, state, nodes)
    np.testing.assert_array_equal(legacy[:, 12:], 0.)


def test_p4_receding_risk_certifies_executed_interval_and_shifted_backup():
    """P4 preserves one safe successor; fixed tasks retain full-horizon risk."""
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    transition_risks = jnp.asarray([
        [0.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6],
        [1.0, 1.0, 0.4, 0.8],
    ])

    def aggregate(walk):
        env = NS(_walk_requires_locomotion=walk)
        return HumanoidBoxPushEnv._aggregate_sequence_risks(
            env, transition_risks
        )

    np.testing.assert_allclose(aggregate(True), [0.0, 0.0, 0.0, 8.0 / 15.0])
    np.testing.assert_allclose(aggregate(False), [1.0, 1.0, 0.4, 8.0 / 15.0])

    three_step = NS(
        _walk_requires_locomotion=True,
        _bcfg=NS(emergency_backup_steps=3),
    )
    np.testing.assert_allclose(
        HumanoidBoxPushEnv._aggregate_sequence_risks(
            three_step, transition_risks
        ),
        [1.0, 1.0, 0.4, 8.0 / 15.0],
    )




def test_h1_synchronized_force_startup_rejects_unsupported_scope():
    import pytest
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )
    valid = dict(robot="h1", level="push_walk", walk_leg_control="joint_target",
                 walk_success_mode="locomotion", walk_force_startup_mode="synchronized")
    for change in ({"walk_force_startup_mode": "invalid"}, {"robot": "g1"},
                   {"level": "push_to_line"}, {"level": "heavy_dr"}, {"level": "unjam"},
                   {"walk_leg_control": "legacy"}, {"walk_success_mode": "legacy"}):
        with pytest.raises(ValueError, match="force.startup|walk_force_startup_mode"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(**{**valid, **change}))


def test_h1_force_startup_legacy_normal_and_benchmark_bitwise_contract():
    """Original default expressions, including the static no-info query."""
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushConfig

    env, ps = _h1_contact_target_array_fixture()
    action = jnp.linspace(-0.7, 0.7, 12)
    for level in ("push_to_line", "heavy_dr", "unjam", "push_walk"):
        env._bcfg = HumanoidBoxPushConfig(
            level=level, approach_time=.3, force_ramp_time=.3, f_target=30., f_max=60.)
        env._is_walk = level == "push_walk"
        env._face_select = level == "unjam"
        for step, contact_step, acquired in ((0, 0, 0.), (7, 1, 1.), (7, 20, 1.),
                                              (16, 1, 1.), (21, 20, 1.), (100, 1, 1.)):
            info = dict(step=jnp.int32(step), contact_step=jnp.int32(contact_step),
                        contact_acquired=jnp.float32(acquired), force_int=jnp.float32(2.))
            elapsed = jnp.maximum(jnp.asarray(info.get("step", 0), jnp.float32)
                                  - jnp.asarray(info.get("contact_step", info.get("step", 0)),
                                                jnp.float32), 0.) * env.dt
            old_scale = jnp.asarray(info.get("contact_acquired", 0.), jnp.float32) * jnp.clip(
                elapsed / max(float(env._bcfg.force_ramp_time), 1e-6), 0., 1.)
            np.testing.assert_array_equal(env._normal_force_startup_scale(info), old_scale)
            actual = env._normal_hand_contact(ps, action, info)
            bound = env._normal_force_startup_scale
            try:
                env._normal_force_startup_scale = lambda info: old_scale
                expected = env._normal_hand_contact(ps, action, info)
            finally:
                env._normal_force_startup_scale = bound
            for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(got, want)
            old_global = jnp.clip(
                (jnp.asarray(info["step"], jnp.float32) * env.dt - env._bcfg.approach_time)
                / max(float(env._bcfg.force_ramp_time), 1e-6), 0., 1.)
            np.testing.assert_array_equal(
                env.requested_force_reference(ps, info),
                jnp.float32(env._bcfg.f_target) * old_global * env.walk_force_scale(ps, info))
        assert float(env._normal_force_startup_scale(None)) == 1.
        assert float(env._normal_hand_contact(ps, action, None)["force_scale"]) == 1.


def test_h1_synchronized_force_startup_minimum_and_selected_observation_clock():
    from dataclasses import replace
    from types import MethodType
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, ps = _h1_contact_target_array_fixture()
    env._bcfg = replace(
        env._bcfg, level="push_walk", walk_leg_control="joint_target",
        walk_success_mode="locomotion", walk_force_startup_mode="synchronized",
        approach_time=.3, force_ramp_time=.3, f_target=30., f_max=60., f_min=0.,
        kp_force=.3, ki_force=1.)
    env._is_walk, env._face_select, env._walk_requires_locomotion = True, False, True
    env._box_contact_force = lambda ps: jnp.float32(0.)
    # This unit isolates the force-startup clock; physical foot-load support is
    # covered independently by the controller/contact diagnostics tests.
    env._foot_contact_loads = lambda ps: jnp.zeros(2)
    env._support_feedback = lambda ps, loads: {
        "stance_support_weights": jnp.zeros(2),
        "support_reference_xy": ps.x.pos[env._pelvis_idx - 1, :2],
    }
    action = jnp.zeros(23).at[11].set(.5)  # 45 N action-selected command, not nominal 30 N.
    observer, ops, ocfg = _joint_target_observation_fixture()
    ocfg.walk_success_mode, ocfg.walk_force_startup_mode = "locomotion", "synchronized"
    for name in ("_global_force_startup_scale", "_normal_force_startup_scale"):
        setattr(observer, name, MethodType(getattr(HumanoidBoxPushEnv, name), observer))
    for step, contact_step, acquired, scale in (
            (7, 1, 1., 0.), (16, 1, 1., 1./15.),
            (21, 20, 1., 1./15.), (100, 1, 0., 0.), (100, 1, 1., 1.)):
        info = dict(step=jnp.int32(step), contact_step=jnp.int32(contact_step),
                    contact_acquired=jnp.float32(acquired), force_int=jnp.float32(0.))
        contact = env._normal_hand_contact(ps, action, info)
        np.testing.assert_allclose(contact["force_scale"], scale, atol=2e-7, rtol=0.)
        np.testing.assert_allclose(contact["F_n"], 45.*scale, atol=1e-5, rtol=0.)
        np.testing.assert_allclose(contact["F_eff"], 1.3*45.*scale, atol=1e-5, rtol=0.)
        obs = HumanoidBoxPushEnv._get_obs(observer, ops, info)
        assert obs.shape == (91,)
        np.testing.assert_array_equal(obs[-1], env._normal_force_startup_scale(info))
        synced_reference = env.requested_force_reference(ps, info)
        old_cfg = env._bcfg
        env._bcfg = replace(old_cfg, walk_force_startup_mode="legacy")
        legacy = env._normal_hand_contact(ps, action, info)
        np.testing.assert_array_equal(env.requested_force_reference(ps, info), synced_reference)
        for name in ("p_c", "p_surface", "n_c", "approach_alpha", "F_n_cmd"):
            np.testing.assert_array_equal(contact[name], legacy[name])
        np.testing.assert_array_equal(contact["left"]["p_c"], legacy["left"]["p_c"])
        env._bcfg = old_cfg
        ocfg.walk_force_startup_mode = "legacy"
        old_obs = HumanoidBoxPushEnv._get_obs(observer, ops, info)
        np.testing.assert_array_equal(obs[:-1], old_obs[:-1])
        ocfg.walk_force_startup_mode = "synchronized"
    # Coincident halfway ramps: min is one-half, never the product one-quarter.
    env._bcfg = replace(env._bcfg, approach_time=.2, force_ramp_time=.4)
    half = dict(step=jnp.int32(20), contact_step=jnp.int32(10), contact_acquired=1.)
    np.testing.assert_allclose(env._normal_force_startup_scale(half), .5, atol=1e-7)
    assert float(env._normal_force_startup_scale(None)) == 1.
    assert float(env._normal_force_startup_scale({})) == 0.


def test_h1_synchronized_force_startup_interface_rejects_same_width_legacy_policy():
    import copy
    import json
    import pytest
    from types import SimpleNamespace as NS
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.experiments.plugins.methods.contact_receding import _validate_policy_interface

    profile, cfg, _, wbc, _ = _joint_target_array_fixture()
    env = NS(_is_walk=True, _bcfg=cfg, _n_planner=11, _whole_body_controller=wbc,
             _robot_profile=profile, spec=PrimitiveSpec(pos_dim=5, stiff_dim=3, feed_dim=1),
             action_size=23, dt=.02, sys=NS(nq=29, nv=28), _gait="slow_walk",
             _gait_params={"slow_walk": jnp.array([.6, .8, .15])})
    legacy = HumanoidBoxPushEnv.policy_interface.fget(env)
    cfg.walk_force_startup_mode = "synchronized"
    synced = HumanoidBoxPushEnv.policy_interface.fget(env)
    assert synced["observation_layout"]["observation_size"] == 91
    assert json.loads(json.dumps(synced)) == synced
    _validate_policy_interface(NS(policy_interface=synced), {"policy_interface": synced})
    with pytest.raises(ValueError, match="policy interface mismatch"):
        _validate_policy_interface(NS(policy_interface=synced),
                                   {"policy_interface": legacy, "observation_size": 91})
    with pytest.raises(ValueError, match="model/execution"):
        _validate_policy_interface(NS(policy_interface=synced), {"policy_interface": synced},
                                   execution_env=NS(policy_interface=legacy))
    semantic_difference = copy.deepcopy(synced)
    semantic_difference["task"].pop("force_startup")
    semantic_difference["task"]["memory_contract"]["contact_clock"] = (
        legacy["task"]["memory_contract"]["contact_clock"])
    assert semantic_difference == legacy
    cfg.walk_force_startup_mode = "legacy"
    assert HumanoidBoxPushEnv.policy_interface.fget(env) == legacy


def test_h1_walk_coast_goal_scope_and_model_coefficients():
    import pytest
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )
    valid = dict(robot="h1", level="push_walk", walk_leg_control="joint_target",
                 walk_success_mode="locomotion", walk_box_goal_mode="coast",
                 box_mass=8., box_frictionloss=8.)
    for change in ({"walk_box_goal_mode": "invalid"}, {"robot": "g1"},
                   {"level": "push_to_line"}, {"level": "unjam"},
                   {"walk_leg_control": "legacy"}, {"walk_success_mode": "legacy"}):
        with pytest.raises(ValueError, match="coast|walk_box_goal_mode"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(**{**valid, **change}))
    # Constructor only: check the actual task-applied arrays before any reset/step.
    for change in ({"box_mass": 0.}, {"box_mass": -1.},
                   {"box_frictionloss": 0.}, {"box_frictionloss": np.inf}):
        with pytest.raises(ValueError, match="positive finite executed"):
            HumanoidBoxPushEnv(HumanoidBoxPushConfig(**{**valid, **change}))


def test_h1_walk_coast_goal_signed_stopping_location_without_physics():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    env = object.__new__(HumanoidBoxPushEnv)
    env._box_idx, env._walk_coast_deceleration = 1, 1.
    def location(x, vx):
        ps = NS(x=NS(pos=jnp.asarray([[x, 0., .55]])),
                xd=NS(vel=jnp.asarray([[vx, 0., 0.]])))
        return env._walk_coasting_box_x(ps)
    fn = jax.jit(location)
    for vx, expected_offset in ((0., 0.), (.25, .03125), (-.25, -.03125),
                                (1.30469465, 1.30469465 ** 2 / 2.)):
        np.testing.assert_allclose(fn(jnp.float32(.39173126), jnp.float32(vx)),
                                   .39173126 + expected_offset, atol=2e-7, rtol=0.)
    # Resetting the location itself, unlike adding a signed position error,
    # gives translation-invariant cost to a correspondingly translated goal.
    np.testing.assert_allclose(fn(10.4, .25) - 10., fn(.4, .25), atol=7e-7)
    # At rest the original position expression is exactly recovered.
    np.testing.assert_array_equal(fn(jnp.float32(.4), jnp.float32(0.)), jnp.float32(.4))


def test_h1_walk_coast_goal_tapers_force_against_stopping_location():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env = object.__new__(HumanoidBoxPushEnv)
    env._box_idx, env._walk_coast_deceleration = 1, 1.
    env._walk_requires_locomotion = True
    env._bcfg = NS(walk_box_goal_mode="coast", goal_eps=.02,
                   walk_stop_distance=.12, walk_approach_force_floor=.35)
    info = {"box_goal_x": jnp.float32(.30)}
    def scale(x, vx):
        ps = NS(x=NS(pos=jnp.asarray([[x, 0., .55]])),
                xd=NS(vel=jnp.asarray([[vx, 0., 0.]])))
        return env.walk_force_scale(ps, info)
    # Same physical position: a box already coasting to the goal must unload,
    # while a stationary box still receives approach force.  The Cartesian
    # target is recomputed from the measured box face, so its tracking gain
    # remains fully active while the tapered force request is positive.  It
    # releases discretely only at the coast condition instead of becoming a
    # progressively weaker hidden contact tracker.
    assert float(scale(.20, np.sqrt(.20))) == 0.
    assert float(scale(.20, 0.)) > 0.
    arm_scale = jax.jit(lambda x, vx: HumanoidBoxPushEnv.walk_arm_task_scale(
        env,
        NS(x=NS(pos=jnp.asarray([[x, 0., .55]])),
           xd=NS(vel=jnp.asarray([[vx, 0., 0.]]))),
        info,
    ))
    assert float(arm_scale(.20, np.sqrt(.20))) == 0.
    assert float(arm_scale(.20, 0.)) == 1.
    intermediate_force_scale = float(scale(.20, .20))
    assert 0.0 < intermediate_force_scale < 1.0
    assert float(arm_scale(.20, .20)) == 1.0
    env._bcfg.walk_box_goal_mode = "position"
    np.testing.assert_allclose(scale(.20, np.sqrt(.20)), scale(.20, 0.), atol=0.)
    ps = NS(x=NS(pos=jnp.asarray([[.20, 0., .55]])),
            xd=NS(vel=jnp.asarray([[np.sqrt(.20), 0., 0.]])))
    assert float(HumanoidBoxPushEnv.walk_arm_task_scale(env, ps, info)) == 1.


def test_h1_walk_coast_release_uses_task_owned_geometric_retract():
    """The coast threshold must detach, not merely set desired force to zero."""
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env = object.__new__(HumanoidBoxPushEnv)
    env._walk_requires_locomotion = True
    env._box_idx, env._walk_coast_deceleration = 1, 1.
    env._bcfg = NS(
        robot="h1", fast_force_loop=True,
        walk_box_goal_mode="coast", goal_eps=.02, walk_stop_distance=.12,
        walk_approach_force_floor=.35, emergency_retract_force=20.,
    )
    env._acquisition_box = object()
    env._mga_inspection_mode = lambda info: jnp.asarray(
        info.get("mga_execution_mode", 0), jnp.int32
    )
    env._empty_mga_execution_context = lambda: {
        "mga_execution_mode": jnp.int32(0),
        "mga_emergency_zero_force": jnp.bool_(False),
    }
    ordinary = {
        "wrench": jnp.asarray([7., 0., 0.]),
        "left": {
            "wrench": jnp.asarray([7., 0., 0.]),
            "p_hand": jnp.asarray([1., .2, 1.]),
            "p_c": jnp.asarray([1.1, .2, 1.]),
        },
        "p_hand": jnp.asarray([1., -.2, 1.]),
        "p_c": jnp.asarray([1.1, -.2, 1.]),
        "F_n": jnp.float32(14.), "F_n_cmd": jnp.float32(14.),
        "F_eff": jnp.float32(14.), "force_scale": jnp.float32(1.),
        "arm_task_scale": jnp.float32(1.),
        "arm_control_scale": jnp.float32(1.),
        "arm_posture_position_scale": jnp.float32(1.),
    }
    env._normal_hand_contact = lambda *args, **kwargs: ordinary
    env._mga_unload_info_is_ready = lambda info: jnp.bool_(False)
    ps_release = NS(
        x=NS(pos=jnp.asarray([[.20, 0., .55]])),
        xd=NS(vel=jnp.asarray([[np.sqrt(.20), 0., 0.]])),
    )
    info = {"box_goal_x": jnp.float32(.30), "mga_execution_mode": jnp.int32(0)}
    contact = env._hand_contact(ps_release, jnp.zeros(1), info)
    np.testing.assert_array_equal(contact["wrench"], [-10., 0., 0.])
    np.testing.assert_array_equal(contact["left"]["wrench"], [-10., 0., 0.])
    np.testing.assert_array_equal(contact["p_c"], contact["p_hand"])
    assert float(contact["F_n"]) == 0.
    assert float(contact["arm_posture_position_scale"]) == 0.

    ps_push = NS(
        x=NS(pos=jnp.asarray([[.20, 0., .55]])),
        xd=NS(vel=jnp.asarray([[0., 0., 0.]])),
    )
    pushing = env._hand_contact(ps_push, jnp.zeros(1), info)
    np.testing.assert_array_equal(pushing["wrench"], [7., 0., 0.])


def test_h1_normal_recovery_reuses_bounded_gait_roll_capture():
    from types import SimpleNamespace as NS
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.robots.h1.profile import h1_profile

    env = object.__new__(HumanoidBoxPushEnv)
    env._is_walk = True
    env._n_planner = 11
    env._walk_joint_reference = jnp.zeros((10, 11))
    env._roll_hips = (1, 6)
    env._sag_legs = ((2, 3, 4), (7, 8, 9))
    env._pelvis_idx = 1
    env._torso_idx = 1
    env._robot_profile = h1_profile()
    env.spec = PrimitiveSpec(pos_dim=5, stiff_dim=3, feed_dim=1)
    env._whole_body_controller = NS(
        planner_joint_bounds=jnp.asarray([
            [-.3, .3], [-.3, .3], [-1., 1.], [0., 1.74], [-.6, .4],
            [-.3, .3], [-.3, .3], [-1., 1.], [0., 1.74], [-.6, .4],
            [-.5, .5],
        ])
    )
    env._bcfg = NS(
        walk_leg_control="joint_target", emergency_reference_rewind_steps=6,
        gait_roll_capture_gain=1.5, gait_roll_capture_lead=.1,
        gait_roll_limit=.3, walk_joint_reference_residual_scale=.2,
        walk_min_steps_per_foot=1, gait_hip_forward_sign=1.,
        gait_capture_gain=.25, gait_capture_lead=.1, gait_hip_amp=.15,
    )
    pelvis_y, support_y, pelvis_vy = .08, -.02, .20
    state = NS(info={"walk_forward_steps": jnp.zeros((2,), jnp.int32)},
               pipeline_state=NS(
        x=NS(rot=jnp.asarray([
            [1., 0., 0., 0.]
        ]), pos=jnp.asarray([[0., pelvis_y, 1.]])),
        xd=NS(ang=jnp.zeros((1, 3)), vel=jnp.asarray([[0., pelvis_vy, 0.]])),
    ))
    env._foot_contact_loads = lambda ps: jnp.asarray([300., 300.])
    env._support_feedback = lambda ps, loads: {
        "support_reference_xy": jnp.asarray([0., support_y])
    }
    nodes = jnp.zeros((3, 23)).at[:, 15].set(.25).at[:, 20].set(-.25)
    recovery = env.normal_recovery_plan(state, nodes)
    capture_rad = np.clip(
        1.5 * (pelvis_y - support_y + .1 * pelvis_vy), -.3, .3
    )
    radians_per_action = .5 * (.3 - (-.3)) * .2
    expected = np.clip(capture_rad / radians_per_action, -1., 1.)
    assert expected > 0.0
    np.testing.assert_allclose(recovery[:, 13], expected, atol=1e-7)
    np.testing.assert_allclose(recovery[:, 18], expected, atol=1e-7)
    np.testing.assert_allclose(recovery[:, 15], [.25, .125, 0.], atol=1e-7)
    np.testing.assert_allclose(recovery[:, 20], [-.25, -.125, 0.], atol=1e-7)
    mask = np.ones(23, dtype=bool)
    mask[[13, 15, 18, 20]] = False
    np.testing.assert_array_equal(np.asarray(recovery)[:, mask], 0.)

    momentum_state = NS(
        info=state.info,
        pipeline_state=NS(
            x=state.pipeline_state.x,
            xd=NS(
                ang=jnp.asarray([[0., .72, 0.]]),
                vel=state.pipeline_state.xd.vel,
            ),
        ),
    )
    bank = env.normal_recovery_plans(momentum_state, nodes)
    assert bank.shape == (8, 3, 23)
    np.testing.assert_array_equal(bank[0], recovery)
    pitch_radians_per_action = .5 * (1. - (-1.)) * .2
    pitch_residual = .1 * .72 / pitch_radians_per_action
    expected_capture = recovery.at[1, 14].add(pitch_residual)
    expected_capture = expected_capture.at[1, 19].add(pitch_residual)
    np.testing.assert_allclose(bank[1], expected_capture, atol=1e-7)
    expected_roll_release = recovery.at[1:, jnp.asarray([13, 18])].set(0.0)
    np.testing.assert_array_equal(bank[2], expected_roll_release)
    expected_combined = expected_roll_release.at[1, 14].add(pitch_residual)
    expected_combined = expected_combined.at[1, 19].add(pitch_residual)
    np.testing.assert_allclose(bank[3], expected_combined, atol=1e-7)
    for bank_index, target in ((4, -.5), (6, -1.)):
        expected_release = recovery.at[
            1:, jnp.asarray([13, 18])
        ].set(target)
        np.testing.assert_array_equal(bank[bank_index], expected_release)
        expected_combined = expected_release.at[1, 14].add(pitch_residual)
        expected_combined = expected_combined.at[1, 19].add(pitch_residual)
        np.testing.assert_allclose(
            bank[bank_index + 1], expected_combined, atol=1e-7
        )

    # NORMAL rescue expresses a short DIAL phase lag in the same bounded
    # residual chart and uses the solver-provided horizon to locate node
    # knots.  It must not mutate the task clock or assume Hsample=16.
    reference = jnp.arange(40, dtype=jnp.float32)[:, None] * jnp.linspace(
        0.001, 0.004, 11, dtype=jnp.float32
    )[None, :]
    env._walk_joint_reference = reference
    rescue_state = NS(
        info={**momentum_state.info, "walk_reference_step": jnp.int32(7)},
        pipeline_state=momentum_state.pipeline_state,
    )
    rescue = env.normal_rescue_plans(
        rescue_state, nodes, horizon_steps=8
    )
    assert rescue.shape == (3, 3, 23)
    node_offsets = jnp.asarray([0, 4, 8], jnp.int32)
    current = node_offsets + 7
    base = env.normal_recovery_plans(rescue_state, nodes)[-1]
    for index, lag in enumerate((1, 2, 3)):
        delayed = jnp.maximum(current - lag, 0)
        delta = (reference[delayed] - reference[current]) / .2
        expected_rescue = base.at[:, env.spec.total_width:].add(delta)
        np.testing.assert_allclose(
            rescue[index], jnp.clip(expected_rescue, -1., 1.), atol=1e-7
        )


def test_h1_walk_prior_waits_for_bilateral_supported_exchange():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env = object.__new__(HumanoidBoxPushEnv)
    env._walk_requires_locomotion = True
    env._bcfg = NS(walk_min_steps_per_foot=1)
    assert not bool(env.mga_prior_applicable(NS(
        info={"walk_forward_steps": jnp.asarray([1, 0], jnp.int32)}
    )))
    assert bool(env.mga_prior_applicable(NS(
        info={"walk_forward_steps": jnp.asarray([1, 1], jnp.int32)}
    )))
    env._walk_requires_locomotion = False
    assert bool(env.mga_prior_applicable(NS(info={})))


def test_h1_walk_coast_prediction_does_not_override_physical_safety():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env = object.__new__(HumanoidBoxPushEnv)
    env._walk_requires_locomotion = True
    env._box_idx, env._walk_coast_deceleration = 1, 1.0
    env._is_walk, env._n_planner = False, 0
    env.spec = NS(total_width=1)
    env._bcfg = NS(
        walk_box_goal_mode="coast", goal_eps=.03, walk_stop_distance=.12,
    )
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(0.),
                                          "nonhand": jnp.float32(0.)}
    env._manifold = lambda ps, u, info: (jnp.zeros(0), jnp.zeros(1))
    env._safety_margins = lambda ps, forces, g: jnp.full((4,), -.1)

    def should_release(box_x, box_vx, success=0.):
        ps = NS(
            x=NS(pos=jnp.asarray([[box_x, 0., .5]])),
            xd=NS(vel=jnp.asarray([[box_vx, 0., 0.]])),
        )
        state = NS(
            pipeline_state=ps,
            info={"box_goal_x": jnp.float32(.30),
                  "task_success": jnp.float32(success)},
        )
        return bool(env.emergency_plan_should_override(state))

    # Neither entering the force-taper band nor predicting a stop inside the
    # goal corridor is itself a safety certificate for a walking contact.
    assert not should_release(.20, .10)
    assert not should_release(.20, .40)
    assert not should_release(.20, .40, success=1.)


def test_h1_walk_coast_goal_changes_only_box_reward_and_policy_contract():
    from copy import copy
    from dataclasses import replace
    import pytest
    from genedynamics.experiments.plugins.methods.contact_receding import _validate_policy_interface
    env = make_env(
        HUMANOID_TASK, robot="h1", level="push_walk",
        walk_leg_control="joint_target", walk_success_mode="locomotion",
        walk_objective_mode="dial", walk_box_goal_mode="coast",
        gait="slow_walk", walk_gait_reference="legacy",
        box_mass=8., box_frictionloss=8., leg_grav_comp=0.,
    )
    state = env.reset(jax.random.PRNGKey(0))
    assert env._walk_coast_deceleration == 1.
    assert float(env.sys.body_mass[env._box_idx]) == 8.
    # The native scene template is not the task-modified execution model.
    assert float(env.sys.mj_model.body_mass[env._box_idx]) == 30.
    old = copy(env)
    old._bcfg = replace(env._bcfg, walk_box_goal_mode="position")
    action = env.plan_initializer(state, jnp.zeros((2, env.action_size)))[0]
    original_ps = state.pipeline_state
    box = env._box_idx - 1
    for speed in (0., .25, -.25):
        ps = original_ps.replace(xd=original_ps.xd.replace(
            vel=original_ps.xd.vel.at[box, 0].set(jnp.float32(speed))))
        previous, previous_done = old._reward_done(ps, action, state.info)
        current, current_done = env._reward_done(ps, action, state.info)
        x = ps.x.pos[box, 0]
        goal = state.info["box_goal_x"]
        expected_delta = env._bcfg.w_box * (
            (x - goal) ** 2 - (x + speed * abs(speed) / 2. - goal) ** 2)
        np.testing.assert_allclose(current - previous, expected_delta, atol=2e-6)
        np.testing.assert_array_equal(current_done, previous_done)
        # Same post-state through the task transition: not a physics rollout.
        tape = {
            "physics_hand_force": jnp.zeros(5),
            "physics_nonhand_force": jnp.zeros(5),
            "physics_safety_margins": -jnp.ones((5, 4)),
        }
        for memory in (state.info, {**state.info, "task_success": jnp.float32(1.)}):
            before = state.replace(info=memory)
            left = old._finish_step(before, ps, action, memory["force_int"], tape)
            right = env._finish_step(before, ps, action, memory["force_int"], tape)
            for lval, rval in zip(
                jax.tree_util.tree_leaves(left.replace(reward=jnp.float32(0))),
                jax.tree_util.tree_leaves(right.replace(reward=jnp.float32(0)))):
                np.testing.assert_array_equal(lval, rval)
            if float(memory["task_success"]) > .5 or speed == 0.:
                np.testing.assert_array_equal(left.reward, right.reward)
    previous_interface, current_interface = old.policy_interface, env.policy_interface
    assert previous_interface["observation_layout"] == current_interface["observation_layout"]
    assert state.obs.shape == (91,)
    def checkpoint_for(task_env, interface):
        transform = task_env.policy_action_transform
        return {
            "policy_interface": interface, "observation_size": 91,
            "policy_action_transform": transform,
            "action_bias": transform["action_bias"],
            "action_scale": transform["action_scale"],
        }
    _validate_policy_interface(old, checkpoint_for(old, previous_interface))
    _validate_policy_interface(env, checkpoint_for(env, current_interface))
    with pytest.raises(ValueError, match="policy interface mismatch"):
        _validate_policy_interface(env, checkpoint_for(env, previous_interface))
    with pytest.raises(ValueError, match="model/execution"):
        _validate_policy_interface(env, checkpoint_for(env, current_interface), execution_env=old)
    assert "box_goal_objective" not in previous_interface["task"]
    objective = current_interface["task"].pop("box_goal_objective")
    assert objective["mode"] == "friction_only_coast" and objective["weight"] == env._bcfg.w_box
    assert previous_interface == current_interface


def test_h1_joint_target_nominal_stance_matches_reward_reference():
    """DIAL's nominal stance is not the residual CPG's opposite phase convention."""
    from types import SimpleNamespace as NS, MethodType
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    cfg = NS(walk_leg_control="joint_target", walk_gait_reference="legacy",
             gait_swing_frac=0.45, gait_cadence=0.8)
    env = NS(_bcfg=cfg, dt=0.02, _gait="slow_walk",
             _gait_params={"slow_walk": jnp.array([0.6, 0.8, 0.15])},
             _gait_phase={"slow_walk": jnp.array([0., 0.5])})
    env._walk_foot_target = MethodType(HumanoidBoxPushEnv._walk_foot_target, env)
    env._walk_phases = MethodType(HumanoidBoxPushEnv._walk_phases, env)
    stance = jax.jit(lambda step: HumanoidBoxPushEnv._walk_nominal_stance(env, {"step": step}))
    np.testing.assert_array_equal(stance(jnp.int32(0)), [True, False])
    for step in (10, 25, 40, 62):
        expected = env._walk_foot_target({"step": step}) <= 1e-6
        np.testing.assert_array_equal(stance(jnp.int32(step)), expected)
    cfg.walk_leg_control = "legacy"
    np.testing.assert_array_equal(HumanoidBoxPushEnv._walk_nominal_stance(env, {"step": 0}),
                                  [False, True])


def _joint_target_observation_fixture():
    from types import SimpleNamespace as NS, MethodType
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    cfg = NS(walk_objective_mode="legacy", walk_box_goal_mode="position",
             walk_force_startup_mode="legacy",
             walk_leg_control="joint_target", walk_gait_reference="legacy", gait_cadence=3.7,
             f_target=30., f_max=60., support_radius=0.2, force_int_max=10., level="push_walk",
             walk_success_mode="legacy", walk_min_steps_per_foot=1, walk_success_hold_time=0.2,
             approach_time=0.3, force_ramp_time=0.3, gait_ramp_time=0.5)
    env = NS(_bcfg=cfg, _is_walk=True, dt=0.02, _gait="slow_walk", _mu=0.6,
             _gait_params={"slow_walk": jnp.array([0.6, 0.8, 0.15])},
             _box_idx=2, _pelvis_idx=1, _feet_site_id=jnp.array([0, 1]),
             _box_contact_forces=lambda ps: {"hand": 0., "wall": 0., "nonhand": 0.},
             _corridor_clearance=lambda ps: 0.)
    env._walk_phase_features = MethodType(HumanoidBoxPushEnv._walk_phase_features, env)
    env._walk_task_memory_features = MethodType(HumanoidBoxPushEnv._walk_task_memory_features, env)
    ps = NS(qpos=jnp.arange(29, dtype=jnp.float32), qvel=jnp.arange(28, dtype=jnp.float32),
            x=NS(pos=jnp.array([[0., 0., 1.], [1., 0., 0.5]]),
                 rot=jnp.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])),
            site_xpos=jnp.array([[0., -0.2, 0.], [0., 0.2, 0.]]))
    return env, ps, cfg


def test_h1_joint_target_observation_exposes_reward_clock_only_in_new_mode():
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, ps, cfg = _joint_target_observation_fixture()
    info = {"step": 25, "box_goal_x": 1.5}
    obs = HumanoidBoxPushEnv._get_obs(env, ps, info)
    angle = 2.0 * np.pi * 0.8 * 0.02 * 25  # reward cadence, not cfg.gait_cadence=3.7
    np.testing.assert_allclose(obs[-2:], [np.sin(angle), np.cos(angle)], atol=1e-6)
    assert obs.shape == (78,)
    cfg.walk_leg_control = "legacy"
    legacy = HumanoidBoxPushEnv._get_obs(env, ps, info)
    assert legacy.shape == (76,)
    np.testing.assert_array_equal(obs[:-2], legacy)
    cfg.walk_leg_control, cfg.walk_gait_reference = "joint_target", "cpg"
    cpg = HumanoidBoxPushEnv._get_obs(env, ps, info)
    angle = 2.0 * np.pi * 3.7 * 0.02 * 25
    np.testing.assert_allclose(cpg[-2:], [np.sin(angle), np.cos(angle)], atol=1e-6)


def test_h1_strict_joint_target_observation_separates_task_history():
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, ps, cfg = _joint_target_observation_fixture()
    cfg.walk_success_mode = "locomotion"
    info = {"step": jnp.int32(25), "box_goal_x": jnp.float32(1.5),
            "contact_acquired": jnp.float32(1.), "contact_step": jnp.int32(20),
            "walk_swing_seen": jnp.zeros(2, bool), "walk_foot_loaded": jnp.zeros(2, bool),
            "walk_swing_eligible": jnp.zeros(2, bool), "walk_landing_x": jnp.zeros(2),
            "walk_forward_steps": jnp.zeros(2, jnp.int32), "walk_goal_hold_time": jnp.float32(0.)}
    observe = jax.jit(lambda memory: HumanoidBoxPushEnv._get_obs(env, ps, memory))
    baseline = observe(info)
    assert baseline.shape == (91,)
    # Identical physical state/phase can have different event history.  None
    # of these distinctions is present in the original 78D prefix.
    changes = {
        "walk_swing_seen": jnp.array([True, False]),
        "walk_foot_loaded": jnp.array([False, True]),
        "walk_swing_eligible": jnp.array([True, False]),
        "walk_landing_x": jnp.array([-0.04, 0.08]),
        "walk_forward_steps": jnp.array([3, 0], jnp.int32),
        "walk_goal_hold_time": jnp.float32(0.3),
    }
    for name, value in changes.items():
        changed = observe({**info, name: value})
        np.testing.assert_array_equal(changed[:78], baseline[:78])
        assert not np.array_equal(changed[78:], baseline[78:]), name
    combined = observe({**info, **changes})
    np.testing.assert_allclose(combined[78:],
                               [1., 0., 0., 1., 1., 0., .04, -.08, 1., 0., 1., 5./6., 1./3.],
                               atol=1e-6)
    # The parent reset supplies its clock before installing task-owned info.
    missing = HumanoidBoxPushEnv._get_obs(env, ps, {"step": jnp.int32(0)})
    np.testing.assert_array_equal(missing[78:], 0.)
    np.testing.assert_array_equal(env._walk_task_memory_features(ps, {}), 0.)
    assert np.isfinite(missing).all()
    cfg.walk_success_mode = "legacy"
    legacy = HumanoidBoxPushEnv._get_obs(env, ps, info)
    assert legacy.shape == (78,)
    np.testing.assert_array_equal(baseline[:78], legacy)
    cfg.walk_leg_control = "legacy"
    np.testing.assert_array_equal(HumanoidBoxPushEnv._get_obs(env, ps, info), baseline[:76])


def test_h1_strict_joint_target_observation_has_independent_saturated_clocks():
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv

    env, ps, cfg = _joint_target_observation_fixture()
    cfg.walk_success_mode = "locomotion"
    observe = lambda info: HumanoidBoxPushEnv._get_obs(env, ps, info)
    # 125 control steps are exactly two slow_walk periods.  Preserve contact
    # age while changing absolute startup time: periodic phase cannot do this.
    early = observe({"step": 5, "contact_acquired": 1., "contact_step": 0})
    late = observe({"step": 130, "contact_acquired": 1., "contact_step": 125})
    np.testing.assert_allclose(early[:78], late[:78], atol=2e-6)
    np.testing.assert_allclose(early[-2:], [1./6., 1./3.], atol=1e-6)
    np.testing.assert_allclose(late[-2:], [1., 1./3.], atol=1e-6)
    # Same physical state and global time; only the contact-latched clock differs.
    acquired_now = observe({"step": 130, "contact_acquired": 1., "contact_step": 130})
    acquired_long_ago = observe({"step": 130, "contact_acquired": 1., "contact_step": 100})
    np.testing.assert_array_equal(acquired_now[:-1], acquired_long_ago[:-1])
    np.testing.assert_array_equal([acquired_now[-1], acquired_long_ago[-1]], [0., 1.])
    no_contact = observe({"step": 130, "contact_acquired": 0., "contact_step": -1})
    assert float(no_contact[-1]) == 0.
    # One bounded startup coordinate reconstructs both approach interpolation
    # and the time-indexed reference force until both have saturated.
    for step in (0, 5, 15, 20, 30, 100):
        features = env._walk_task_memory_features(ps, {"step": step})
        represented_time = float(features[-2]) * 0.6
        actual_time = step * env.dt
        actual = [np.clip(actual_time / .3, 0., 1.), np.clip((actual_time - .3) / .3, 0., 1.)]
        encoded = [np.clip(represented_time / .3, 0., 1.),
                   np.clip((represented_time - .3) / .3, 0., 1.)]
        np.testing.assert_allclose(encoded, actual, atol=1e-6)
    cfg.walk_gait_reference, cfg.gait_ramp_time = "cpg", 2.
    np.testing.assert_allclose(env._walk_task_memory_features(ps, {"step": 25})[-2], .25,
                               atol=1e-6)


def test_h1_strict_joint_target_observation_real_reset_shape():
    """Real construction/reset only, without a physics step or controller call."""
    env = make_env(HUMANOID_TASK, level="push_walk", walk_leg_control="joint_target",
                   walk_success_mode="locomotion", push_dist=0.5, gait="slow_walk")
    state = env.reset(jax.random.PRNGKey(110))
    assert state.obs.shape == (91,)
    assert not bool(state.info["physics_samples_valid"])
    np.testing.assert_array_equal(state.info["physics_hand_force"], jnp.zeros(env._n_frames))
    np.testing.assert_array_equal(state.info["physics_nonhand_force"], jnp.zeros(env._n_frames))
    assert state.info["physics_safety_margins"].shape == (env._n_frames, 4)
    assert env.reliability_contract()["schema"]["version"] == 4
    assert env.action_size == 23
    assert int(state.info["mga_execution_mode"]) == 0
    assert int(state.info["mga_execution_request"]) == -1
    assert not bool(state.info["mga_unload_entry_prepared"])
    assert state.info["mga_unload_targets"].shape == (2, 3)
    assert state.info["mga_unload_stiffness_matrix"].shape == (3, 3)
    assert env.policy_interface["observation_layout"]["observation_size"] == 91
    assert env.policy_interface["task"]["walk_safety_envelope"] == {
        "min_torso_up": 0.9,
        "min_height_ratio": 0.7,
        "fixed_stance_proxy_applies": False,
    }
    assert "update_action" not in env.geometry_reliability(state)
    saved_reference = env._walk_joint_reference
    env._walk_joint_reference = jnp.zeros((2, env._n_planner), jnp.float32)
    gate = env.geometry_reliability(state)
    assert "update_action" in gate
    update_gate = np.asarray(gate["update_action"])
    # No measured box contact at reset: retain searchable contact geometry,
    # but preserve nominal stiffness/force and locally bounded leg authority
    # about the DIAL reference.
    np.testing.assert_array_equal(update_gate[:env.spec.s_slice.start], 1.0)
    np.testing.assert_array_equal(update_gate[env.spec.s_slice], 0.0)
    np.testing.assert_array_equal(update_gate[env.spec.nu_slice], 0.0)
    np.testing.assert_array_equal(update_gate[env.spec.total_width:], 0.0)
    candidates = jnp.linspace(
        -0.9, 0.9, 3 * env.action_size, dtype=jnp.float32
    ).reshape(3, env.action_size)
    projected = env.project_mga_candidate(state, candidates)
    np.testing.assert_allclose(
        projected[:, :env.spec.total_width],
        candidates[:, :env.spec.total_width],
    )
    np.testing.assert_array_equal(
        projected[:, env.spec.total_width:],
        np.zeros((3, env._n_planner), dtype=np.float32),
    )
    # Once the task has observed a supported left-leg swing, both legs may use
    # the policy chart so the opposite leg can maintain support through the
    # exchange.  The torso remains on the shared gait; all candidate sources
    # receive the same clipped authority.
    swing_state = state.replace(info={
        **state.info,
        "walk_swing_seen": jnp.asarray([True, False]),
    })
    projected_swing = env.project_mga_candidate(swing_state, candidates)
    residual_limit = env._bcfg.policy_joint_reference_residual_scale
    paired_leg_width = (env._n_planner - 1) // 2
    expected_swing = np.zeros((3, env._n_planner), dtype=np.float32)
    expected_swing[:, :2 * paired_leg_width] = np.clip(
        np.asarray(
            candidates[
                :, env.spec.total_width:env.spec.total_width + 2 * paired_leg_width
            ]
        ),
        -residual_limit,
        residual_limit,
    )
    np.testing.assert_allclose(
        projected_swing[:, env.spec.total_width:], expected_swing
    )
    planner = tuple(env._robot_profile.joint_groups["planner"])
    emergency = env.emergency_plan(state, candidates)
    capture = env._walk_roll_capture_residual(state)
    for hip in env._roll_hips:
        coordinate = env.spec.total_width + planner.index(hip)
        np.testing.assert_allclose(
            emergency[:, coordinate], capture[planner.index(hip)], atol=1e-7
        )
    projected_emergency = env.project_mga_candidate(state, emergency)
    # Candidate projection has NORMAL semantics and must not infer execution
    # mode from the emergency plan's -1 force coordinate.  The backend owns
    # explicit emergency provenance and therefore never sends its task-owned
    # bank through this local proposal tube.
    np.testing.assert_array_equal(
        projected_emergency[:, env.spec.total_width:],
        np.zeros((3, env._n_planner), dtype=np.float32),
    )
    emergency_bank = env.emergency_plans(state, candidates)
    assert emergency_bank.shape == (16, 3, env.action_size)
    projected_emergency_bank = env.project_mga_candidate(state, emergency_bank)
    np.testing.assert_array_equal(
        projected_emergency_bank[..., env.spec.total_width:],
        np.zeros((16, 3, env._n_planner), dtype=np.float32),
    )
    env._walk_joint_reference = saved_reference
    # Dynamic walking does not reject a valid single-support posture merely
    # because the pelvis is away from the two-foot midpoint.  It does reject
    # an early torso-envelope violation before the terminal fall threshold.
    assert float(env._balance_safety_residual(state.pipeline_state)) < 0.0
    assert float(env._walk_terminal_recovery_residual(
        state.pipeline_state
    )) < 0.0
    tilted_rot = state.pipeline_state.x.rot.at[env._torso_idx - 1].set(
        jnp.asarray([jnp.cos(.3), jnp.sin(.3), 0., 0.])
    )
    tilted = state.pipeline_state.replace(
        x=state.pipeline_state.x.replace(rot=tilted_rot)
    )
    assert float(env._balance_safety_residual(tilted)) > 0.0
    assert float(env._walk_terminal_recovery_residual(tilted)) > 0.0
    env._walk_requires_locomotion = False
    shifted_pos = state.pipeline_state.x.pos.at[env._pelvis_idx - 1, 0].add(1.0)
    shifted = state.pipeline_state.replace(
        x=state.pipeline_state.x.replace(pos=shifted_pos)
    )
    assert float(env._balance_safety_residual(shifted)) > 0.0
    env._walk_requires_locomotion = True
    memory = np.asarray(state.obs[78:])
    np.testing.assert_array_equal(memory[:2], 0.)
    np.testing.assert_array_equal(memory[2:4], np.asarray(state.info["walk_foot_loaded"], np.float32))
    np.testing.assert_array_equal(memory[4:], 0.)
    env._bcfg.walk_success_mode = "legacy"
    try:
        np.testing.assert_array_equal(env._get_obs(state.pipeline_state, state.info), state.obs[:78])
        assert env.policy_interface["observation_layout"]["observation_size"] == 78
    finally:
        env._bcfg.walk_success_mode = "locomotion"


def _h1_forward_gait_sign():
    """The opt-in sign must move H1's swing foot forward, verified by MuJoCo FK."""
    from genedynamics.core.control.bipedal_gait import BipedalGait, GaitParams
    env = make_env(HUMANOID_TASK, level="push_walk")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe("home").qpos
    mujoco.mj_forward(model, data)
    foot_id = int(env._feet_site_id[0])
    x0 = float(data.site_xpos[foot_id, 0])
    hip_local = env._sag_legs[0][0]
    data.qpos[env._robot_binding.qpos_indices[hip_local]] += 0.01
    mujoco.mj_forward(model, data)
    dx_positive = float(data.site_xpos[foot_id, 0]) - x0
    assert dx_positive < 0.0
    gait = BipedalGait(
        *env._sag_legs, env._robot_profile.num_actuated,
        params=GaitParams(hip_forward_sign=-1.0, stance_sweep=True),
    )
    start = gait(jnp.float32(0.0), 0.0, 0.0, 0.0)[hip_local]
    end = gait(jnp.float32(0.99 * gait.p.swing_frac / gait.p.cadence), 0.0, 0.0, 0.0)[hip_local]
    assert float(end) < float(start)
    touchdown_time = gait.p.swing_frac / gait.p.cadence
    before = gait(jnp.float32(touchdown_time - 1e-5), 0.0, 0.0, 0.0)[hip_local]
    after = gait(jnp.float32(touchdown_time + 1e-5), 0.0, 0.0, 0.0)[hip_local]
    assert abs(float(after - before)) < 1e-3
    assert GaitParams().hip_forward_sign == 1.0
    assert GaitParams().stance_sweep is False
    print(f"  H1 gait: +0.01 hip rad moves foot {dx_positive:.5f} m; forward sign -1 -> True")
    return True


def test_h1_measured_support_allocation_and_shared_reference():
    """Array-only support contract: actual loads, total authority, flight limit."""
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
    from genedynamics.core.control.humanoid_contact import HumanoidWholeBodyController

    ps = NS(x=NS(pos=jnp.array([[0.1, 0.05, 1.0]])),
            xd=NS(vel=jnp.array([[0.1, 0.0, 0.0]])),
            site_xpos=jnp.array([[-0.2, 0.1, 0.0], [0.6, -0.1, 0.0]]),
            qpos=jnp.zeros(8), qvel=jnp.zeros(8), qfrc_bias=jnp.zeros(8))
    env = NS(_robot_weight=500.0, _pelvis_idx=1, _feet_site_id=jnp.array([0, 1]))
    support = jax.jit(lambda loads: HumanoidBoxPushEnv._support_feedback(env, ps, loads))
    cases = (
        ([250., 250.], [1., 1.], [0.2, 0.0]),
        ([500., 0.], [2., 0.], [-0.2, 0.1]),
        ([0., 500.], [0., 2.], [0.6, -0.1]),
        ([125., 375.], [0.5, 1.5], [0.4, -0.05]),
        ([1000., 0.], [2., 0.], [-0.2, 0.1]),
        ([100., 0.], [0.4, 0.], [0.04, 0.06]),
        ([0., 0.], [0., 0.], [0.1, 0.05]),
        ([-20., 0.], [0., 0.], [0.1, 0.05]),
    )
    for loads, weights, reference in cases:
        result = support(jnp.array(loads))
        np.testing.assert_allclose(result["stance_support_weights"], weights, atol=1e-6)
        np.testing.assert_allclose(result["support_reference_xy"], reference, atol=1e-6)
    near_flight = support(jnp.array([1e-3, 0.]))
    np.testing.assert_allclose(near_flight["support_reference_xy"], [0.1, 0.05], atol=1e-6)
    assert float(jnp.sum(near_flight["stance_support_weights"])) < 1e-5

    cfg = NS(f_target=30., stance_force_hip_deadband=0., dt=0.02,
             target_vx=0.15, gait_ramp_time=0.1, stance_hip_bias=0.,
             stance_force_hip_gain=0., stance_ankle_bias=0., leg_scale=0.1,
             leg_grav_comp=0., stance_force_reference=30., stance_force_ankle_gain=0.,
             stance_com_ankle_gain=3., stance_com_ankle_damping=2.,
             arm_null_damping=0.01, arm_posture_kp=0., arm_posture_kd=0., arm_grav_comp=0.)
    # Expose both capture offsets directly without a separate fake controller.
    def capture_reference(t, dx, vx, target_vx, dy, vy):
        del t, vx, target_vx, vy
        return jnp.zeros(8).at[0].set(dx).at[3].set(dy)
    wbc = HumanoidWholeBodyController(
        binding=NS(qpos_indices=tuple(range(8)), dof_indices=tuple(range(8))),
        config=cfg, default_pose=jnp.zeros(8), stance_pose=jnp.zeros(8),
        kp=jnp.zeros(8), kd=jnp.zeros(8), torque_limits=jnp.ones(8) * 100.,
        arm_push_pose=jnp.zeros(8), right_arm=(6,), left_arm=(7,),
        sagittal_legs=((0, 1, 2), (3, 4, 5)), n_planner=6,
        pelvis_body_id=1, feet_site_ids=jnp.array([0, 1]), stance_com_x0=0.03,
        is_walk=True, gait_controller=capture_reference,
    )
    hand = {"jacp": jnp.zeros((8, 3)), "wrench": jnp.zeros(3)}
    legacy = {**hand, "left": hand, "F_n": jnp.float32(0.)}
    info, action = {"step": 10}, jnp.zeros(6)
    loaded = {**legacy, **support(jnp.array([500., 0.]))}
    target = wbc.joint_targets(ps, loaded, action, info, 0)
    np.testing.assert_allclose(target[jnp.array([0, 3])], [0.3, -0.05], atol=1e-6)
    tau = wbc.torque(ps, loaded, action, info, 0)
    # Same left-foot reference as capture, with double the per-ankle authority.
    np.testing.assert_allclose(tau[jnp.array([2, 5])], [2.02, 0.], atol=1e-6)
    flight = {**legacy, **support(jnp.zeros(2))}
    np.testing.assert_allclose(wbc.joint_targets(ps, flight, action, info, 0), 0., atol=1e-6)
    np.testing.assert_allclose(wbc.torque(ps, flight, action, info, 0), 0., atol=1e-6)
    # Optional hooks absent: unchanged both-feet mean and both-ankle feedback.
    np.testing.assert_allclose(
        wbc.joint_targets(ps, legacy, action, info, 0)[jnp.array([0, 3])],
        [-0.1, 0.05], atol=1e-6,
    )
    np.testing.assert_allclose(
        wbc.torque(ps, legacy, action, info, 0)[jnp.array([2, 5])],
        [-0.19, -0.19], atol=1e-6,
    )


def test_h1_foot_loads_ignore_nonfloor_inactive_and_tensile_contacts(monkeypatch):
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid import box_push_brax as module
    env = NS(_floor_geom=0, _foot_body_ids=jnp.array([1, 2]),
             sys=NS(geom_bodyid=jnp.array([0, 1, 2, 3])))
    ps = NS(contact=NS(
        geom=jnp.array([[0, 1], [2, 0], [1, 3], [0, 1], [1, 0], [0, 1], [0, 3]]),
        dist=jnp.array([-0.01, 0., -0.01, 0.01, -0.01, -0.01, -0.01]),
    ))
    normals = jnp.array([100., 250., 999., 999., -20., 25., 999.])
    monkeypatch.setattr(module._mjx_support, "contact_force",
                        lambda system, state, index: jnp.zeros(6).at[0].set(normals[index]))
    loads = jax.jit(lambda: module.HumanoidBoxPushEnv._foot_contact_loads(env, ps))()
    np.testing.assert_array_equal(loads, [125., 250.])


def _h1_support_phase_control():
    """Measured support is phase-independent; nominal foot-level retains residuals."""
    from dataclasses import replace
    env = make_env(
        HUMANOID_TASK, level="push_walk", walk_success_mode="locomotion",
        walk_leg_control="support_phase_foot_level", gait_hip_forward_sign=-1.0,
    )
    state = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros(env.action_size)
    ps = state.pipeline_state
    info = {**state.info, "step": 10}
    contact = env._hand_contact(ps, action, info)
    assert np.array_equal(contact["swing_feet"], [True, False])
    different_phase = env._hand_contact(ps, action, {**info, "step": 40})
    np.testing.assert_array_equal(contact["stance_support_weights"],
                                  different_phase["stance_support_weights"])
    np.testing.assert_array_equal(contact["support_reference_xy"],
                                  different_phase["support_reference_xy"])
    np.testing.assert_allclose(
        env._robot_weight,
        env.sys.mj_model.body_subtreemass[env._pelvis_idx]
        * np.linalg.norm(env.sys.mj_model.opt.gravity), rtol=1e-6,
    )
    assert env._robot_weight < (float(np.sum(env.sys.mj_model.body_mass))
                               * np.linalg.norm(env.sys.mj_model.opt.gravity))
    lifted = ps.replace(contact=ps.contact.replace(dist=jnp.ones_like(ps.contact.dist)))
    np.testing.assert_array_equal(env._feet_on_ground(lifted), [False, False])
    airborne = jax.jit(env._hand_contact)(lifted, action, info)
    np.testing.assert_array_equal(airborne["stance_support_weights"], [0, 0])
    np.testing.assert_allclose(airborne["support_reference_xy"],
                               ps.x.pos[env._pelvis_idx - 1, :2], atol=1e-6)
    wbc = env._whole_body_controller
    args = (action, info, env.spec.total_width)
    joint_targets = jax.jit(wbc.joint_targets, static_argnums=(4,))
    target = joint_targets(ps, contact, *args)
    hip, knee, ankle = env._sag_legs[0]
    expected = np.clip(
        float(contact["swing_foot_pitch_target"] - target[hip] - target[knee]),
        float(env.physical_joint_range[ankle, 0]) + 0.02,
        float(env.physical_joint_range[ankle, 1]) - 0.02,
    )
    np.testing.assert_allclose(target[ankle], expected, atol=1e-6)
    assert np.all(np.asarray(target[:env._n_planner]) >=
                  np.asarray(env.physical_joint_range[:env._n_planner, 0]) + 0.01999)
    assert np.all(np.asarray(target[:env._n_planner]) <=
                  np.asarray(env.physical_joint_range[:env._n_planner, 1]) - 0.01999)
    # Foot levelling defines the nominal reference, not an action projection:
    # the swing ankle retains both signs of its independent residual authority.
    amplitude = 0.5
    increment = env._bcfg.leg_scale * amplitude
    assert float(env.physical_joint_range[ankle, 0]) + 0.02 < float(target[ankle]) - increment
    assert float(target[ankle]) + increment < float(env.physical_joint_range[ankle, 1]) - 0.02
    for sign in (-1.0, 1.0):
        nonzero = action.at[env.spec.total_width + ankle].set(sign * amplitude)
        changed = joint_targets(ps, contact, nonzero, info, env.spec.total_width)
        expected_delta = jnp.zeros_like(target).at[ankle].set(sign * increment)
        np.testing.assert_allclose(changed-target, expected_delta, atol=1e-6)
    # Hip residuals also stay independent; levelling must not cancel them by
    # injecting an opposite, unrequested ankle residual after the fact.
    hip_action = action.at[env.spec.total_width + hip].set(amplitude)
    changed = joint_targets(ps, contact, hip_action, info, env.spec.total_width)
    np.testing.assert_allclose(changed-target,
                               jnp.zeros_like(target).at[hip].set(increment), atol=1e-6)
    # Isolate the additive COM feedback from PD and gravity compensation.
    cfg = replace(env._bcfg, leg_grav_comp=0.0, stance_force_ankle_gain=0.0,
                  stance_com_ankle_gain=1.0, stance_com_ankle_damping=0.0)
    test_wbc = replace(wbc, config=cfg, kp=jnp.zeros_like(wbc.kp),
                       kd=jnp.zeros_like(wbc.kd),
                       torque_limits=jnp.ones_like(wbc.torque_limits) * 10000.0)
    displaced = ps.replace(x=ps.x.replace(
        pos=ps.x.pos.at[env._pelvis_idx - 1, 0].add(0.1)))
    stance_contact = {**contact, "stance_support_weights": jnp.array([0., 1.])}
    supported = test_wbc.torque(displaced, stance_contact, *args)
    airborne_tau = test_wbc.torque(displaced,
        {**stance_contact, "stance_support_weights": jnp.zeros(2)}, *args)
    delta = np.asarray(supported - airborne_tau)
    assert abs(float(delta[env._sag_legs[0][2]])) < 1e-6
    assert abs(float(delta[env._sag_legs[1][2]])) > 0.05
    # Missing optional keys retains the old both-ankle feedback path.
    legacy = {k: v for k, v in contact.items()
              if k not in {"stance_support_weights", "support_reference_xy",
                           "swing_feet", "swing_foot_pitch_target"}}
    legacy_tau = test_wbc.torque(displaced, legacy, *args)
    assert abs(float(legacy_tau[env._sag_legs[0][2]])) > 0.05
    assert abs(float(legacy_tau[env._sag_legs[1][2]])) > 0.05
    print("  support: actual load independent of planned phase; ankle residual retained -> True")
    return True


def test_h1_walk_reference_clock_uses_measured_monotone_phase_and_holds_unload():
    from types import SimpleNamespace as NS
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushEnv,
    )

    env = NS(
        _bcfg=NS(emergency_reference_rewind_steps=4),
        _walk_joint_reference=jnp.asarray([
            [0.0, 0.0], [0.25, 0.0], [0.5, 0.0],
            [0.75, 0.0], [1.0, 0.0],
        ], jnp.float32),
        _robot_binding=NS(qpos_indices=(0, 1)),
        _n_planner=2,
        _whole_body_controller=NS(
            planner_action_from_joints=lambda joints: joints
        ),
    )
    env._mga_inspection_mode = HumanoidBoxPushEnv._mga_inspection_mode
    anchor, error = HumanoidBoxPushEnv._walk_reference_phase_anchor(
        env, NS(qpos=jnp.asarray([0.52, 0.0], jnp.float32)),
        {"walk_reference_step": jnp.int32(4)},
    )
    assert int(anchor) == 2
    assert float(error) == pytest.approx(np.sqrt(0.0002), abs=1e-7)
    next_clock = HumanoidBoxPushEnv._next_walk_reference_step
    normal = {
        "walk_reference_step": jnp.int32(112),
        "mga_execution_mode": jnp.int32(0),
        "mga_execution_request": jnp.int32(0),
    }
    assert int(next_clock(env, normal)) == 113
    entering = {**normal, "mga_execution_request": jnp.int32(1)}
    assert int(next_clock(env, entering, jnp.int32(110))) == 110
    continuing = {
        **entering,
        "walk_reference_step": jnp.int32(110),
        "walk_reference_recovery_anchor": jnp.int32(110),
        "mga_execution_mode": jnp.int32(1),
    }
    assert int(next_clock(env, continuing, jnp.int32(109))) == 110
    recovering = {**continuing, "mga_execution_request": jnp.int32(0)}
    assert int(next_clock(env, recovering, jnp.int32(109))) == 111
    reentering = {
        **recovering,
        "walk_reference_step": jnp.int32(111),
        "mga_execution_mode": jnp.int32(0),
        "mga_execution_request": jnp.int32(1),
    }
    next_reference, next_anchor = HumanoidBoxPushEnv._next_walk_reference_clock(
        env, reentering, jnp.int32(109)
    )
    assert int(next_reference) == 110
    assert int(next_anchor) == 110


def _native_walk_probe(mode="support_phase_foot_level", *, n_steps=300, with_box=False,
                       diagnostic=None, pulse_from=None, pulse_step=20, output_tag=None,
                       com_audit_from=None, env_overrides=None):
    """Low-memory feasibility screen; NOT a substitute for MJX validation.

    Native MuJoCo owns physics only.  State conversion follows Brax's MJX
    pipeline and reuses the actual task control, force PI, and state transition.
    No independently implemented gait/controller or alternate success metric.
    ``diagnostic=standing/gait/residual_pulse`` additionally releases Cartesian
    hand control, retaining the existing full arm posture PD.  These deliberately
    isolated probes are NOT the complete pushing task or its final controller.
    The pulse case replays one saved pre-fall state, with zero and six signed
    pulses (left/right hip-knee and bilateral ankles); it is not a gain search.
    Explicit ``env_overrides`` are saved with the resolved parameters; the
    standing/no-contact diagnostic contract still takes precedence.
    """
    import copy
    import json
    import math
    import time
    from dataclasses import replace
    from pathlib import Path
    from brax.base import Motion, Transform
    from brax.mjx.base import State as PipelineState
    from brax.mjx.pipeline import _reformat_contact
    from mujoco import mjx
    from genedynamics.experiments.framework.config import ExperimentConfig

    if diagnostic not in {None, "standing", "gait", "residual_pulse"}:
        raise ValueError("diagnostic must be standing, gait, or residual_pulse")
    if diagnostic and with_box:
        raise ValueError("released-hand diagnostics cannot claim full-task validation")
    if diagnostic == "residual_pulse" and env_overrides:
        raise ValueError("matched-state pulses must retain the anchor's environment parameters")
    tag = output_tag or diagnostic or f"{mode}_{'push' if with_box else 'isolated'}"
    output = Path("results/_development/humanoid_mga_repair") / f"native_walk_{tag}"
    if (output / "results.json").exists():
        raise FileExistsError(f"Preserve the previous probe; choose a new output_tag: {output}")
    started = time.monotonic()
    config = ExperimentConfig.from_yaml(Path("configs/humanoid/push_to_line/main/mga.yaml"))
    config = config.for_suite(next(s for s in config.suites if s["name"] == "p4_walk_push"))
    params = {**config.env_params, "push_dist": 0.50, "walk_success_mode": "locomotion",
              "walk_gait_reference": "cpg", "walk_leg_control": mode,
              "gait_hip_forward_sign": -1.0, "gait_stance_sweep": True,
              "w_walk_progress": 5.0, "target_vx": 0.15}
    params.update(env_overrides or {})
    if not with_box:
        # Keep the Cartesian target at home as well as moving the box away;
        # otherwise impedance would pull toward the distant box at large force.
        params.update(approach_gap=2.0, approach_time=1e6, f_target=0.0, f_min=0.0, f_max=0.0)
    anchor = None
    if diagnostic in {"standing", "gait"}:
        params["walk_leg_control"] = "legacy"
        if diagnostic == "gait":
            params["walk_leg_control"] = (env_overrides or {}).get("walk_leg_control", "legacy")
    if diagnostic == "standing":
        params.update(gait_hip_amp=0.0, gait_knee_amp=0.0, gait_capture_gain=0.0,
                      gait_forward_gain=0.0, gait_roll_amp=0.0, gait_roll_capture_gain=0.0)
    if diagnostic == "residual_pulse":
        if pulse_from is None:
            raise ValueError("residual_pulse requires an explicit saved standing/gait report")
        source = json.loads(Path(pulse_from).read_text())
        if source.get("diagnostic") not in {"standing", "gait"}:
            raise ValueError("pulse anchor must use the same released-hand controller")
        anchor = next(r for r in source["rows"] if r["step"] == int(pulse_step))
        if anchor["done"]:
            raise ValueError("pulse anchor must be pre-terminal")
        params = dict(source["env_params"])
        n_steps = 10  # 0.2s fixed pulses; no optimizing their duration.
    env = make_env(config.env_name, **params)
    state = env.reset(jax.random.PRNGKey(0))
    model = copy.copy(env.sys.mj_model)
    # sys overrides are consumed by MJX but are not all copied to mj_model.
    for field in ("jnt_range", "jnt_limited", "dof_frictionloss", "body_mass", "body_inertia",
                  "geom_size", "geom_friction", "body_pos", "pair_solref", "geom_pos",
                  "geom_rgba", "site_pos"):
        getattr(model, field)[:] = np.asarray(getattr(env.sys, field))
    model.opt.timestep = float(env.sys.opt.timestep)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    data.qpos[:] = np.asarray(state.pipeline_state.qpos)
    data.qvel[:] = np.asarray(state.pipeline_state.qvel)
    mujoco.mj_forward(model, data)

    def pipeline_state():
        native = mjx.put_data(model, data)
        x = Transform(pos=native.xpos[1:], rot=native.xquat[1:])
        cvel = Motion(vel=native.cvel[1:, 3:], ang=native.cvel[1:, :3])
        offset = Transform.create(pos=native.xpos[1:] - native.subtree_com[env.sys.body_rootid[1:]])
        args = dict(native.__dict__)
        args["contact"] = _reformat_contact(env.sys, native.contact)
        return PipelineState(q=native.qpos, qd=native.qvel, x=x,
                             xd=offset.vmap().do(cvel), **args)

    wbc = env._whole_body_controller
    if diagnostic:
        # With the Cartesian Jacobian released, its nullspace becomes identity:
        # the existing posture branch supplies ordinary PD at profile gains.
        arm_ids = np.asarray(env._right_arm + env._left_arm)
        arm_kp, arm_kd = np.asarray(wbc.kp)[arm_ids], np.asarray(wbc.kd)[arm_ids]
        np.testing.assert_allclose(arm_kp, arm_kp[0])
        np.testing.assert_allclose(arm_kd, arm_kd[0])
        wbc = replace(wbc, config=replace(env._bcfg, arm_posture_kp=float(arm_kp[0]),
                                         arm_posture_kd=float(arm_kd[0])))

    @jax.jit
    def control(ps, action, info):
        contact = env._hand_contact(ps, action, info)
        if diagnostic:
            contact = {**contact, "wrench": jnp.zeros_like(contact["wrench"]),
                       "jacp": jnp.zeros_like(contact["jacp"]),
                       "left": {**contact["left"],
                                "wrench": jnp.zeros_like(contact["left"]["wrench"]),
                                "jacp": jnp.zeros_like(contact["left"]["jacp"])}}
        return (wbc.torque(ps, contact, action, info, env.spec.total_width),
                wbc.joint_targets(ps, contact, action, info, env.spec.total_width),
                contact.get("stance_support_weights", jnp.ones(2)),
                env._foot_contact_loads(ps),
                contact.get("support_reference_xy",
                            ps.site_xpos[env._feet_site_id, :2].mean(axis=0)))

    def foot_loads(*, active_only=False):
        load = np.zeros(2)
        force = np.zeros(6)
        foot_bodies = np.asarray(env._foot_body_ids)
        for i in range(data.ncon):
            pair = np.asarray(data.contact[i].geom)
            if env._floor_geom not in pair:
                continue
            if active_only and data.contact[i].dist > 0.0:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            for side, body in enumerate(foot_bodies):
                if body in model.geom_bodyid[pair]:
                    load[side] += max(float(force[0]), 0.0)
        return load

    def restore_info(template, saved):
        if isinstance(template, dict):
            return {k: restore_info(v, saved[k]) for k, v in template.items()}
        return jnp.asarray(saved, dtype=getattr(template, "dtype", None))

    integral = jax.jit(env._update_force_integral)
    finish = jax.jit(env._finish_step)
    forces = jax.jit(env._box_contact_forces)
    if anchor is not None:
        data.qpos[:] = anchor["q"]
        data.qvel[:] = anchor["qd"]
        data.time = anchor["time"]
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        state = state.replace(info=restore_info(state.info, anchor["task_info"]))
    state = state.replace(pipeline_state=pipeline_state())
    start_state, start_data = state, copy.copy(data)
    com_audit = []
    if com_audit_from is not None:
        audit_report = json.loads(Path(com_audit_from).read_text())
        audit_data = copy.copy(start_data)
        checkpoints = {1, 20, 50, 100, audit_report["rows"][-1]["step"]}
        for row in audit_report["rows"]:
            if row["step"] not in checkpoints:
                continue
            audit_data.qpos[:] = row["q"]
            audit_data.qvel[:] = row["qd"]
            mujoco.mj_forward(model, audit_data)
            mujoco.mj_subtreeVel(model, audit_data)
            com_audit.append({"step": row["step"],
                              "pelvis_xyz": audit_data.xpos[env._pelvis_idx].tolist(),
                              "robot_com": audit_data.subtree_com[env._pelvis_idx].tolist(),
                              "robot_com_velocity": audit_data.subtree_linvel[env._pelvis_idx].tolist()})
    feet_start = np.asarray(state.pipeline_state.site_xpos[env._feet_site_id])
    actions = {"zero": jnp.zeros(env.action_size)}
    if diagnostic == "residual_pulse":
        directions = {}
        for side, (hip, knee, _) in zip(("left", "right"), env._sag_legs):
            directions[f"{side}_hip_knee"] = jnp.zeros(env.action_size).at[
                env.spec.total_width + hip].set(-1.0).at[env.spec.total_width + knee].set(1.0)
        ankle_ids = jnp.asarray([leg[2] for leg in env._sag_legs])
        directions["bilateral_ankles"] = jnp.zeros(env.action_size).at[
            env.spec.total_width + ankle_ids].set(1.0)
        for name, direction in directions.items():
            actions[f"{name}_plus"] = direction
            actions[f"{name}_minus"] = -direction
    branches = {}
    foot_load_contract_passed = True
    foot_load_error_max = 0.0
    for name, action in actions.items():
        state, data = start_state, copy.copy(start_data)
        mujoco.mj_subtreeVel(model, data)
        previous_com_velocity = data.subtree_linvel[env._pelvis_idx].copy()
        rows = []
        for step in range(int(n_steps)):
            ps, force_int = state.pipeline_state, state.info["force_int"]
            tau_history, load_history, control_load_errors = [], [], []
            for _ in range(env._n_frames):
                info = {**state.info, "force_int": force_int}
                # Compare the exact pre-control state, not native force after
                # mj_step against controller force from the previous substep.
                native_control_loads = foot_loads(active_only=True)
                control_state_time = float(data.time)
                tau, q_target, support_weights, control_loads, support_reference = control(
                    ps, action, info
                )
                load_error = float(np.max(np.abs(
                    np.asarray(control_loads) - native_control_loads
                )))
                control_load_errors.append(load_error)
                foot_load_error_max = max(foot_load_error_max, load_error)
                foot_load_contract_passed = foot_load_contract_passed and bool(np.allclose(
                    control_loads, native_control_loads, rtol=1e-5, atol=1e-3
                ))
                data.ctrl[:] = np.asarray(tau)
                tau_history.append(np.asarray(tau))
                mujoco.mj_step(model, data)
                load_history.append(foot_loads())
                ps = pipeline_state()
                force_int = integral(ps, action, info)
            state = finish(state, ps, action, force_int)
            contact_force = forces(ps)
            q = np.asarray(ps.qpos)
            w, x, y, z = q[3:7]
            feet = np.asarray(ps.site_xpos[env._feet_site_id])
            mujoco.mj_subtreeVel(model, data)
            com_velocity = data.subtree_linvel[env._pelvis_idx].copy()
            task_step = int(state.info["step"])
            row = {"step": task_step, "time": task_step * env.dt,
                   "body_progress": float(state.info["walk_body_progress"]),
                   "support_progress": float(state.info["walk_support_progress"]),
                   "forward_steps": np.asarray(state.info["walk_forward_steps"]).tolist(),
                   "root_xyz": np.asarray(ps.x.pos[env._pelvis_idx - 1]).tolist(),
                   "root_pitch": math.asin(float(np.clip(2 * (w*y-z*x), -1, 1))),
                   "root_generalized_accel": data.qacc[:6].tolist(),
                   "robot_com": data.subtree_com[env._pelvis_idx].tolist(),
                   "robot_com_velocity": com_velocity.tolist(),
                   "robot_com_accel": ((com_velocity-previous_com_velocity)/env.dt).tolist(),
                   "feet_xyz": feet.tolist(), "feet_delta": (feet-feet_start).tolist(),
                   "feet_normal_force": load_history[-1].tolist(),
                   "feet_peak_normal_force": np.max(load_history, axis=0).tolist(),
                   "support_weights": np.asarray(support_weights).tolist(),
                   "control_state_time": control_state_time,
                   "control_foot_loads": np.asarray(control_loads).tolist(),
                   "native_control_foot_loads": native_control_loads.tolist(),
                   "control_foot_load_error_max": max(control_load_errors),
                   "support_reference_xy": np.asarray(support_reference).tolist(),
                   "applied_tau": tau_history[-1].tolist(),
                   "peak_abs_tau": np.max(np.abs(tau_history), axis=0).tolist(),
                   "q_target": np.asarray(q_target).tolist(),
                   "hand_force": float(contact_force["hand"]),
                   "nonhand_force": float(contact_force["nonhand"]),
                   "success": float(state.info["task_success"]), "done": float(state.done),
                   "q": q.tolist(), "qd": np.asarray(ps.qvel).tolist(),
                   "task_info": jax.tree_util.tree_map(lambda v: np.asarray(v).tolist(), state.info)}
            rows.append(row)
            previous_com_velocity = com_velocity
            if (step + 1) % 25 == 0 or bool(state.done) or step + 1 == n_steps:
                print({"branch": name, **{k: row[k] for k in (
                    "step", "body_progress", "support_progress", "forward_steps",
                    "root_pitch", "feet_normal_force", "done")}}, flush=True)
            if bool(state.done):
                break
        branches[name] = {"action": np.asarray(action).tolist(), "rows": rows}
    output.mkdir(parents=True, exist_ok=True)
    report = {"backend": "native_mujoco_screen_only", "env_params": params, "seed": 0,
              "env_overrides": dict(env_overrides or {}), "requested_control_steps": int(n_steps),
              "diagnostic": diagnostic, "released_cartesian_hands": bool(diagnostic),
              "anchor_report": pulse_from, "anchor_step": pulse_step if anchor else None,
              "com_audit_report": com_audit_from, "robot_com_audit": com_audit,
              "foot_load_contract_passed": foot_load_contract_passed,
              "foot_load_error_max": foot_load_error_max,
              "physics_timestep": model.opt.timestep, "rows": branches["zero"]["rows"],
              "branches": branches, "wall_seconds": time.monotonic()-started}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Native screen saved: {output / 'results.json'}", flush=True)
    assert foot_load_contract_passed, "Native/MJX foot-load mismatch; report saved for audit"
    return report


def main():
    print("== levels =="); a = all([
        _check_level("push_to_line", want_nu=12), _check_level("heavy_dr", want_nu=12),
        _check_level("unjam", want_nu=12), _check_level("unjam", use_base=True, want_nu=15),
        _check_level("push_walk", want_nu=23, want_h=10)])
    print("== physical contact geometry =="); b = _physical_contact_geometry()
    print("== robot swap contract =="); c = _robot_swap_contract()
    print("== face selection =="); d = _face_selection()
    print("== heavy_dr domain randomization =="); e = _h2_dr_varies()
    print("== ablations active (humanoid, unjam) =="); f = _ablations_active()
    print("== metric plugin end-to-end =="); g = _metric_plugin_end_to_end()
    print("== paper algorithm contracts =="); h = _paper_algorithm_contracts()
    print("== ATACOM finite rollout =="); i = _atacom_finite_rollout()
    print("== locomotion success contract =="); j = _walk_locomotion_contract()
    print("== emergency isotropic unload =="); k = _emergency_isotropic_unload()
    print("== H1 forward gait sign =="); l = _h1_forward_gait_sign()
    print("== H1 support-phase control =="); m = _h1_support_phase_control()
    ok = a and b and c and d and e and f and g and h and i and j and k and l and m
    print("RESULT:", "HUMANOID OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


def _h1_unload_context_array_fixture(*, decoder_scale=1.0):
    """Actual context helpers on synthetic arrays: no model or physics step."""
    from collections import namedtuple
    from types import SimpleNamespace
    from brax.envs.base import State
    from genedynamics.core.control.stiffness import PrimitiveSpec
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        HumanoidBoxPushConfig, HumanoidBoxPushEnv,
    )

    env = object.__new__(HumanoidBoxPushEnv)
    env._bcfg = HumanoidBoxPushConfig(robot="h1", fast_force_loop=True)
    env._acquisition_box = object()
    env._is_walk = False
    env.spec = PrimitiveSpec(pos_dim=5, stiff_dim=3, feed_dim=1)
    env._rhand_geom, env._lhand_geom, env._box_geom = 0, 1, 2
    env.sys = SimpleNamespace(geom_size=jnp.asarray([
        [.03, 0., 0.], [.03, 0., 0.], [.55, .55, .55],
    ]))
    env._stiffness = lambda raw: jnp.diag(jnp.exp(raw[jnp.asarray([0, 3, 5])])) * decoder_scale
    env._unpack = lambda action: (None, None, None, None,
                                  env._stiffness(action[env.spec.s_slice]), None)
    Physics = namedtuple("UnloadContextPhysics", "geom_xpos geom_xmat qvel")
    ps = Physics(jnp.asarray([[-.574, -.21, .3], [-.579, .21, .3], [0., 0., 0.]]),
                 jnp.tile(jnp.eye(3)[None], (3, 1, 1)),
                 jnp.zeros(76))
    prev = jnp.zeros(12).at[env.spec.s_slice].set(jnp.asarray([.1, .2, -.1, .3, .2, -.2]))
    state = State(ps, jnp.zeros(76), jnp.float32(0.), jnp.float32(0.), {}, {
        "step": jnp.int32(47), "prev_action": prev, "force_int": jnp.float32(-5.),
        **env._empty_mga_execution_context(),
    })
    return env, state


def test_h1_unload_context_seals_actual_geometry_and_stiffness_across_models():
    execution, state = _h1_unload_context_array_fixture()
    nominal, _ = _h1_unload_context_array_fixture(decoder_scale=4.)
    entry = jax.jit(execution._prepare_mga_execution_state)(state, 1)
    assert bool(execution._mga_unload_entry_is_ready(entry))
    assert int(entry.info["mga_execution_mode"]) == 0  # preparation is not execution
    assert int(entry.info["mga_execution_request"]) == 1
    np.testing.assert_allclose(entry.info["mga_unload_targets"],
                               [[-.594, -.21, .3], [-.599, .21, .3]], atol=1e-7)
    np.testing.assert_array_equal(entry.info["mga_unload_normals"], [[-1., 0., 0.]] * 2)
    np.testing.assert_array_equal(entry.info["mga_unload_stiffness_raw"],
                                  state.info["prev_action"][execution.spec.s_slice])
    # The nominal model has a different box pose AND stiffness decoder.
    moved = entry.pipeline_state._replace(
        geom_xpos=entry.pipeline_state.geom_xpos.at[2, 0].add(.1))
    transferred = entry.replace(pipeline_state=moved)
    repeated = nominal._prepare_mga_execution_state(transferred, 1)
    for key in execution._mga_unload_entry_keys():
        np.testing.assert_array_equal(repeated.info[key], entry.info[key])
    candidate = jnp.ones(12)
    np.testing.assert_array_equal(nominal.realized_hand_stiffness(repeated, candidate),
                                  entry.info["mga_unload_stiffness_matrix"])
    assert not np.array_equal(nominal._stiffness(entry.info["mga_unload_stiffness_raw"]),
                              entry.info["mga_unload_stiffness_matrix"])
    emergency = execution.emergency_plan(state, jnp.ones((5, 12)))
    np.testing.assert_array_equal(emergency[:, execution.spec.s_slice],
        np.tile(np.asarray(state.info["prev_action"][execution.spec.s_slice]), (5, 1)))


def test_h1_unload_context_cancellation_continuation_and_normal_reentry():
    env, state = _h1_unload_context_array_fixture()
    env._bcfg.emergency_min_dwell_steps = 2
    entry = env._prepare_mga_execution_state(state, 1)
    cancelled = env._prepare_mga_execution_state(entry, 0)
    assert not bool(cancelled.info["mga_unload_entry_prepared"])
    assert int(cancelled.info["mga_execution_mode"]) == 0
    committed_info = env._mga_finish_context(entry.info, dict(entry.info), jnp.bool_(False))
    committed = entry.replace(info=committed_info)
    assert int(committed.info["mga_execution_mode"]) == 1
    assert int(committed.info["mga_execution_request"]) == -1
    assert int(committed.info["mga_unload_age"]) == 1
    assert not bool(env.normal_recovery_ready(committed))
    assert int(env._mga_inspection_mode(committed.info)) == 1
    continued = env._prepare_mga_execution_state(committed, 1)
    continued = continued.replace(info=env._mga_finish_context(
        continued.info, dict(continued.info), jnp.bool_(False)
    ))
    assert int(continued.info["mga_execution_mode"]) == 1
    assert int(continued.info["mga_unload_age"]) == 2
    assert bool(env.normal_recovery_ready(continued))
    changed = committed.replace(pipeline_state=committed.pipeline_state._replace(
        geom_xpos=committed.pipeline_state.geom_xpos.at[2, 0].add(.05)))
    for prepared in (env._prepare_mga_execution_state(changed, 1),
                     env._prepare_mga_execution_state(
                         env._prepare_mga_execution_state(changed, 0), 1)):
        for key in env._mga_unload_entry_keys():
            np.testing.assert_array_equal(prepared.info[key], entry.info[key])
    # Ordinary step entry defaults NORMAL, despite committed UNLOAD diagnostics.
    normal = env._mga_interval_state(changed)
    assert int(normal.info["mga_execution_request"]) == 0
    assert bool(normal.info["mga_unload_entry_prepared"])
    finished = normal.replace(info=env._mga_finish_context(
        normal.info, dict(normal.info), jnp.bool_(False)))
    assert int(finished.info["mga_execution_mode"]) == 0
    assert int(finished.info["mga_unload_age"]) == 0
    assert not bool(finished.info["mga_unload_entry_prepared"])
    new_entry = env._prepare_mga_execution_state(finished, 1)
    assert bool(new_entry.info["mga_unload_entry_prepared"])
    # Absorbing padding does not commit a counterfactual mode request.
    padded_info = env._mga_finish_context(normal.info, dict(normal.info), jnp.bool_(True))
    assert int(padded_info["mga_execution_mode"]) == 1
    assert int(padded_info["mga_unload_age"]) == 1
    assert bool(padded_info["mga_unload_entry_prepared"])


def test_h1_fixed_task_unload_has_continuous_certified_normal_recovery():
    env, state = _h1_unload_context_array_fixture()
    env._is_walk = False
    env._bcfg.f_min = 0.0
    env._bcfg.f_max = 60.0
    env._bcfg.f_target = 45.0
    entry = env._prepare_mga_execution_state(state, 1)
    nodes = env.emergency_plan(state, jnp.ones((4, 12)))

    recovery = env.normal_recovery_plan(entry, nodes)
    bank = env.normal_recovery_plans(entry, nodes)

    assert recovery.shape == nodes.shape
    assert bank.shape == (3, *nodes.shape)
    np.testing.assert_array_equal(bank[0], recovery)
    np.testing.assert_allclose(
        bank[1, :, env.spec.nu_slice.start], [-1.0, -0.75, -0.5, -0.25],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        bank[2, :, env.spec.nu_slice], -np.ones((4, 1))
    )
    np.testing.assert_array_equal(
        recovery[:, env.spec.s_slice],
        np.tile(np.asarray(entry.info["mga_unload_stiffness_raw"]), (4, 1)),
    )
    np.testing.assert_allclose(
        recovery[:, env.spec.nu_slice.start], [-1.0, -0.5, 0.0, 0.5],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        recovery[:, env.spec.r_slice], np.zeros((4, env.spec.pos_dim))
    )


def test_h1_fixed_displacement_task_has_geometry_then_load_rescue_bank():
    env, state = _h1_unload_context_array_fixture()
    env._is_walk = False
    env._bcfg.fixed_force_target = False
    env._bcfg.f_min = 0.0
    env._bcfg.f_max = 60.0
    env._bcfg.f_target = 45.0
    nodes = jnp.linspace(-0.8, 0.8, 48, dtype=jnp.float32).reshape(4, 12)

    bank = env.normal_rescue_plans(state, nodes, horizon_steps=16)

    assert bank.shape == (10, *nodes.shape)
    b_index = env.spec.r_slice.stop - 1
    geometry_unchanged = np.ones(nodes.shape[-1], dtype=bool)
    geometry_unchanged[b_index] = False
    for candidate in bank[:2]:
        np.testing.assert_array_equal(
            candidate[:, geometry_unchanged], nodes[:, geometry_unchanged]
        )
        np.testing.assert_array_equal(candidate[0], nodes[0])
    np.testing.assert_allclose(bank[:2, -1, b_index], [-0.5, -1.0], atol=1e-7)

    load_unchanged = np.ones(nodes.shape[-1], dtype=bool)
    load_unchanged[env.spec.nu_slice] = False
    for candidate in bank[2::2]:
        np.testing.assert_array_equal(
            candidate[:, load_unchanged], nodes[:, load_unchanged]
        )
        np.testing.assert_array_equal(
            candidate[0, env.spec.nu_slice], nodes[0, env.spec.nu_slice]
        )
    for candidate in bank[3::2]:
        np.testing.assert_array_equal(
            candidate[:, load_unchanged], nodes[:, load_unchanged]
        )
    expected_force = np.asarray([0.875, 0.75, 0.5, 0.0]) * 45.0
    expected_nu = 2.0 * expected_force / 60.0 - 1.0
    np.testing.assert_allclose(
        bank[2::2, -1, env.spec.nu_slice].reshape(-1), expected_nu, atol=1e-7
    )
    np.testing.assert_allclose(
        bank[3::2, :, env.spec.nu_slice.start],
        np.repeat(expected_nu[:, None], nodes.shape[0], axis=1),
        atol=1e-7,
    )

    env._bcfg.fixed_force_target = True
    assert env.normal_rescue_plans(
        state, nodes, horizon_steps=16
    ) is None


@pytest.mark.parametrize("invalid", ["corner", "interior", "deep", "nonfinite", "raw_outside", "bad_matrix"])
def test_h1_unload_context_rejects_invalid_geometry_or_stiffness_without_repair(invalid):
    env, state = _h1_unload_context_array_fixture()
    ps, info = state.pipeline_state, dict(state.info)
    if invalid == "corner":
        ps = ps._replace(geom_xpos=ps.geom_xpos.at[0, 1].set(-.58))
    elif invalid == "interior":
        ps = ps._replace(geom_xpos=ps.geom_xpos.at[0, 0].set(-.54))
    elif invalid == "deep":
        ps = ps._replace(geom_xpos=ps.geom_xpos.at[0, 0].set(-.555))
    elif invalid == "nonfinite":
        ps = ps._replace(geom_xpos=ps.geom_xpos.at[0, 0].set(jnp.nan))
    elif invalid == "raw_outside":
        info["prev_action"] = info["prev_action"].at[env.spec.s_slice.start].set(1.01)
    else:
        env._stiffness = lambda raw: -jnp.eye(3)
    entry = env._prepare_mga_execution_state(state.replace(pipeline_state=ps, info=info), 1)
    assert bool(entry.info["mga_unload_entry_prepared"])
    assert not bool(env._mga_unload_entry_is_ready(entry))
    # An invalid-but-prepared entry must remain invalid in a valid nominal model.
    nominal, nominal_state = _h1_unload_context_array_fixture()
    transferred = entry.replace(pipeline_state=nominal_state.pipeline_state)
    repeated = nominal._prepare_mga_execution_state(transferred, 1)
    assert not bool(nominal._mga_unload_entry_is_ready(repeated))
    for key in env._mga_unload_entry_keys():
        np.testing.assert_array_equal(repeated.info[key], entry.info[key])


def test_h1_unload_context_missing_seal_executes_zero_force_before_step():
    env, state = _h1_unload_context_array_fixture()
    for committed, pending in ((0, 1), (1, -1), (1, 1)):
        missing = state.replace(info={**state.info, "mga_execution_mode": jnp.int32(committed),
                                      "mga_execution_request": jnp.int32(pending)})
        prepared = env._prepare_mga_execution_state(missing, 1)
        assert not bool(prepared.info["mga_unload_entry_prepared"])
        assert not bool(env._mga_unload_entry_is_ready(prepared))
    calls = []
    env.step = lambda s, a: (calls.append("physics"), s)[1]
    context = env.mga_execution_context
    malformed = state.replace(info={**state.info, "mga_execution_request": jnp.int32(1)})
    executed = context["step"](malformed, jnp.zeros(12), 1)
    assert calls == ["physics"]
    assert bool(executed.info["mga_emergency_zero_force"])
    with pytest.raises(TypeError, match="must run on the host"):
        jax.jit(lambda action: context["step"](state, action, 0))(jnp.zeros(12))
    assert calls == ["physics"]


def test_h1_unload_normal_and_g1_preserve_previous_control_arithmetic():
    env, state = _h1_unload_context_array_fixture()
    one_hand = {"F_n": jnp.float32(7.125),
                "wrench": jnp.asarray([1.25, -3.5, 2.25])}
    expected = {**one_hand, "left": dict(one_hand)}
    env._normal_hand_contact = lambda *args, **kwargs: expected
    env._unload_hand_contact = lambda ps, action, info, ordinary: {
        "F_n": jnp.float32(0.), "wrench": jnp.zeros(3),
        "left": {"F_n": jnp.float32(0.), "wrench": jnp.zeros(3)},
    }
    action = jnp.arange(12, dtype=jnp.float32) / 17.
    normal = env._prepare_mga_execution_state(state, 0)
    observed = jax.jit(env._hand_contact)(normal.pipeline_state, action, normal.info)
    for key, value in one_hand.items():
        np.testing.assert_array_equal(observed[key], value)
    np.testing.assert_array_equal(env.realized_hand_stiffness(normal, action), env._unpack(action)[4])
    env._bcfg.robot = "g1"
    assert env.mga_execution_context is None
    g1 = env._hand_contact(state.pipeline_state, action, state.info)
    for key, value in one_hand.items():
        np.testing.assert_array_equal(g1[key], value)
    old_emergency = env.emergency_plan(state, jnp.zeros((5, 12)))
    np.testing.assert_array_equal(old_emergency[:, env.spec.s_slice], np.tile([-1., 0., 0., -1., 0., -1.], (5, 1)))


def test_h1_locomotion_emergency_uses_cartesian_zero_wrench_branch():
    env, state = _h1_unload_context_array_fixture()
    env._walk_requires_locomotion = True
    entry = env._prepare_mga_execution_state(state, 1)
    assert bool(env._mga_unload_entry_is_ready(entry))
    assert bool(entry.info["mga_emergency_zero_force"])
    ordinary = {
        "F_n": jnp.float32(3.), "wrench": jnp.ones(3),
        "p_hand": jnp.zeros(3), "p_surface": jnp.zeros(3),
        "left": {"F_n": jnp.float32(3.), "wrench": jnp.ones(3),
                 "p_hand": jnp.zeros(3)},
    }
    env._normal_hand_contact = lambda *args, **kwargs: ordinary
    expected = {"wrench": jnp.asarray([-2., 0., 0.]),
                "left": {"wrench": jnp.asarray([-2., 0., 0.])}}
    env._unload_hand_contact = lambda *args, **kwargs: pytest.fail(
        "locomotion emergency must not use a moving world-frame spring"
    )
    env._zero_force_hand_contact = lambda *args, **kwargs: expected
    contact = env._hand_contact(
        entry.pipeline_state, jnp.zeros(12), entry.info
    )
    assert contact is expected


def test_h1_sequence_initial_risk_uses_instantaneous_not_previous_interval_envelope():
    env, state = _h1_terminal_array_fixture()
    env._box_contact_force = lambda ps: jnp.float32(0.)
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(0.), "nonhand": jnp.float32(0.)}
    # Safety sampler reports the current state, not the historical tape below.
    env._physics_safety_sample = lambda ps: {
        "physics_safety_margins": jnp.asarray([-.5, -1., -1., -1.])}
    state = state.replace(info={**state.info, "physics_samples_valid": jnp.bool_(True),
        "physics_safety_margins": jnp.ones((1, 4)) * 99.})
    _, safe = env.sequence_score_risk(state, jnp.zeros((2, 2)))
    np.testing.assert_array_equal(safe[:3], 0.)
    env._physics_safety_sample = lambda ps: {
        "physics_safety_margins": jnp.asarray([.1, .2, .3, -.1])}
    _, unsafe = env.sequence_score_risk(state, jnp.zeros((2, 2)))
    np.testing.assert_allclose(unsafe[:3], [1., 1., .3])


def test_h1_unload_invalid_entry_scores_zero_force_and_valid_entry_can_execute():
    env, state = _h1_unload_context_array_fixture()
    env._sequence_score_risk = lambda *args, **kw: (jnp.float32(-7.), jnp.ones(4))
    malformed = state.replace(info={**state.info, "mga_execution_request": jnp.int32(1)})
    score, risk = jax.jit(env.emergency_sequence_score_risk)(malformed, jnp.zeros((5, 12)))
    assert float(score) == -7. and not bool(env.sequence_risk_is_safe(risk))
    assert bool(env._prepare_mga_execution_state(malformed, 1).info[
        "mga_emergency_zero_force"
    ])
    entry = env._prepare_mga_execution_state(state, 1)
    score, risk = env.emergency_sequence_score_risk(entry, jnp.zeros((5, 12)))
    assert float(score) == -7. and not bool(env.sequence_risk_is_safe(risk))
    # This no-physics stub isolates the host boundary: valid geometry is
    # executable mitigation even if its risk was not certified safe.
    env.step = lambda s, action: s.replace(reward=jnp.float32(-7.))
    executed = env.mga_execution_context["step"](entry, jnp.zeros(12), 1)
    assert float(executed.reward) == -7.
    for matrix in (-jnp.eye(3), jnp.eye(3).at[0, 0].set(jnp.nan)):
        corrupt = entry.replace(info={**entry.info, "mga_unload_stiffness_matrix": matrix})
        assert not bool(env._mga_unload_entry_is_ready(corrupt))


def test_h1_invalid_unload_geometry_uses_zero_cartesian_arm_force():
    env, state = _h1_unload_context_array_fixture()
    hand = {
        "p_c": jnp.asarray([1., 2., 3.]),
        "p_hand": jnp.asarray([1.5, 2.5, 3.5]),
        "jacp": jnp.zeros((76, 3)),
        "wrench": jnp.asarray([4., 5., 6.]),
        "f_n": jnp.float32(7.),
        "f_t": jnp.asarray([0., 5., 6.]),
    }
    ordinary = {
        **hand,
        "left": {**hand, "p_hand": -hand["p_hand"]},
        "n_c": jnp.asarray([-1., 0., 0.]),
        "p_surface": jnp.ones(3),
        "F_n": jnp.float32(20.),
        "F_n_cmd": jnp.float32(20.),
        "F_eff": jnp.float32(20.),
        "force_scale": jnp.float32(1.),
        "approach_alpha": jnp.float32(1.),
        "arm_task_scale": jnp.float32(0.),
        "arm_control_scale": jnp.float32(1.),
        "arm_posture_position_scale": jnp.float32(1.),
        "support_load": jnp.float32(12.),
    }
    env._normal_hand_contact = lambda *args, **kwargs: ordinary
    malformed = state.replace(info={
        **state.info,
        "mga_execution_request": jnp.int32(1),
    })
    contact = env._hand_contact(
        malformed.pipeline_state, jnp.zeros(12), malformed.info
    )
    np.testing.assert_array_equal(contact["wrench"], 0.)
    np.testing.assert_array_equal(contact["left"]["wrench"], 0.)
    assert float(contact["F_n"]) == 0.
    assert float(contact["F_eff"]) == 0.
    assert float(contact["support_load"]) == 0.
    assert float(contact["arm_task_scale"]) == 1.
    assert float(contact["arm_control_scale"]) == 1.
    assert float(contact["arm_posture_position_scale"]) == 0.


@pytest.mark.parametrize("has_support_hook", [False, True])
def test_h1_unload_contact_preserves_legacy_and_measured_support_pytrees(has_support_hook):
    env, state = _h1_unload_context_array_fixture()
    env._rhand_body, env._lhand_body = 10, 11
    env._box_contact_force = lambda ps: jnp.float32(12.)
    env._one_hand = lambda *args: {"wrench": jnp.zeros(3)}
    ordinary = {
        "wrench": jnp.zeros(3), "left": {"wrench": jnp.zeros(3)},
        "n_c": jnp.zeros(3), "p_surface": jnp.zeros(3),
        "F_n": jnp.float32(20.), "F_n_cmd": jnp.float32(20.),
        "F_eff": jnp.float32(20.), "force_scale": jnp.float32(1.),
        "approach_alpha": jnp.float32(1.),
    }
    if has_support_hook:
        ordinary["support_load"] = jnp.float32(7.)
    env._normal_hand_contact = lambda *args, **kwargs: ordinary
    entry = env._prepare_mga_execution_state(state, 1)
    contact = jax.jit(lambda mode: env._hand_contact(
        state.pipeline_state, jnp.zeros(12),
        {**entry.info, "mga_execution_request": mode}))
    normal, unload = contact(jnp.int32(0)), contact(jnp.int32(1))
    assert jax.tree_util.tree_structure(normal) == jax.tree_util.tree_structure(unload)
    assert set(normal) == set(ordinary) == set(unload)
    for key in ordinary:
        jax.tree_util.tree_map(np.testing.assert_array_equal, normal[key], ordinary[key])
    if has_support_hook:
        assert float(unload["support_load"]) == 12.
    else:
        assert "support_load" not in unload


def test_h1_execution_contact_uses_host_branch_for_concrete_mode(monkeypatch):
    env, state = _h1_unload_context_array_fixture()
    env._rhand_body, env._lhand_body = 10, 11
    env._box_contact_force = lambda ps: jnp.float32(12.)
    env._one_hand = lambda *args: {"wrench": jnp.zeros(3)}
    ordinary = {
        "wrench": jnp.zeros(3), "left": {"wrench": jnp.zeros(3)},
        "n_c": jnp.zeros(3), "p_surface": jnp.zeros(3),
        "F_n": jnp.float32(20.), "F_n_cmd": jnp.float32(20.),
        "F_eff": jnp.float32(20.), "force_scale": jnp.float32(1.),
        "approach_alpha": jnp.float32(1.),
    }
    env._normal_hand_contact = lambda *args, **kwargs: ordinary
    entry = env._prepare_mga_execution_state(state, 1)

    def unexpected_cond(*args, **kwargs):
        raise AssertionError("host-concrete execution mode must not compile lax.cond")

    monkeypatch.setattr(jax.lax, "cond", unexpected_cond)
    normal = env._hand_contact(state.pipeline_state, jnp.zeros(12), state.info)
    assert normal is ordinary

    unload = env._hand_contact(state.pipeline_state, jnp.zeros(12), entry.info)
    assert float(unload["F_n"]) == 0.
    assert float(unload["F_eff"]) <= 0.


def _h1_unload_terminal_array_fixture():
    """Absorbing context on real step/finish hooks, with array-only physics."""
    from types import SimpleNamespace

    env, state = _h1_unload_context_array_fixture()
    env._n_frames, env._debug = 5, False
    env._walk_requires_locomotion = False
    env._control = lambda ps, action, info: jnp.zeros(1)
    env._pipeline = SimpleNamespace(step=lambda sys, ps, tau, debug:
        ps._replace(geom_xpos=ps.geom_xpos + 1.))
    env._update_force_integral = lambda ps, action, info: jnp.float32(-30.)
    env._get_obs = lambda ps, info: jnp.asarray([info["force_int"], info["step"]], jnp.float32)
    env._reward_done = lambda ps, action, info: (jnp.float32(-2.), jnp.float32(0.))
    env._has_fallen = lambda ps: jnp.bool_(False)
    env._task_reached = lambda ps, info: jnp.bool_(False)
    env._contact_acquisition_detected = lambda *args: jnp.bool_(False)
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(85.), "nonhand": jnp.float32(0.)}
    env._box_contact_force = lambda ps: env._box_contact_forces(ps)["hand"]
    env._manifold = lambda ps, action, info: (jnp.zeros(0), jnp.zeros(3))
    env._safety_margins = lambda ps, forces, balance: jnp.asarray([
        (forces["hand"] - env._bcfg.f_max) / env._bcfg.f_max, -1., 0., -1.])
    env._physics_safety_sample = lambda ps: {
        "physics_hand_force": env._box_contact_force(ps),
        "physics_nonhand_force": jnp.float32(0.),
        "physics_safety_margins": env._safety_margins(ps, env._box_contact_forces(ps), 0.),
    }
    env.requested_force_reference = lambda ps, info: jnp.float32(0.)
    info = {**state.info, "contact_acquired": jnp.float32(1.), "contact_step": jnp.int32(1),
            "task_success": jnp.float32(1.), "task_fallen": jnp.float32(0.),
            "success_padding": jnp.bool_(False), "force_int": jnp.float32(7.),
            "nested": {"value": jnp.asarray([2., 3.])},
            **env._empty_physics_safety_samples()}
    return env, state.replace(info=info, obs=env._get_obs(state.pipeline_state, info))


@pytest.mark.parametrize("committed,prepared,pending", [
    (0, False, -1), (0, False, 1), (0, True, 0), (1, False, -1), (1, True, -1),
])
def test_h1_unload_success_padding_preserves_context_and_positive_integral(committed, prepared, pending):
    env, state = _h1_unload_terminal_array_fixture()
    if prepared:
        entry = env._mga_unload_entry(state)
        state = state.replace(info={**state.info, **entry})
    state = state.replace(info={**state.info, "mga_execution_mode": jnp.int32(committed),
                               "mga_execution_request": jnp.int32(pending)})
    context = env.mga_execution_context
    for mode in (0, 1):
        pending_state = context["prepare_state"](state, mode)
        jax.tree_util.tree_map(np.testing.assert_array_equal, pending_state, state)
        interval = env._mga_interval_state(pending_state)
        jax.tree_util.tree_map(np.testing.assert_array_equal, interval, state)
        # Invalid UNLOAD is irrelevant after actual success: no new action
        # executes. The original loop may still simulate and discard a suffix.
        padding = context["step"](state, jnp.ones(12), mode)
        direct = jax.jit(env.step)(state, jnp.ones(12))
        for observed in (padding, direct):
            jax.tree_util.tree_map(np.testing.assert_array_equal,
                                  observed.pipeline_state, state.pipeline_state)
            assert bool(observed.done) and bool(observed.info["success_padding"])
            assert not bool(observed.info["physics_samples_valid"])
            assert float(observed.info["force_int"]) == 7.
            assert int(observed.info["mga_execution_mode"]) == committed
            assert int(observed.info["mga_execution_request"]) == -1
            assert int(observed.info["step"]) == int(state.info["step"]) + 1
            np.testing.assert_array_equal(observed.info["physics_hand_force"], 0.)
            np.testing.assert_array_equal(observed.info["physics_safety_margins"], 0.)
            for key in env._mga_unload_entry_keys():
                np.testing.assert_array_equal(observed.info[key], state.info[key])
            jax.tree_util.tree_map(np.testing.assert_array_equal,
                                  observed.info["nested"], state.info["nested"])
            np.testing.assert_array_equal(observed.info["prev_action"], state.info["prev_action"])


@pytest.mark.parametrize("force", [10., 85.])
def test_h1_unload_success_padding_scores_actual_frozen_force_without_entry(force):
    env, state = _h1_unload_terminal_array_fixture()
    env._box_contact_forces = lambda ps: {"hand": jnp.float32(force), "nonhand": jnp.float32(0.)}
    assert not bool(env._mga_unload_entry_is_ready(state))
    actions = jnp.zeros((3, 12))
    score, risk = jax.jit(env.emergency_sequence_score_risk)(state, actions)
    normal_score, normal_risk = env.sequence_score_risk(state, actions[:1])
    assert np.isfinite(score) and np.isfinite(risk).all()
    np.testing.assert_array_equal(score, normal_score)
    np.testing.assert_array_equal(risk, normal_risk)
    assert bool(env.sequence_risk_is_safe(risk)) == (force <= env._bcfg.f_max)
    assert float(risk[0]) == float(force > env._bcfg.f_max)
    assert float(risk[3]) == pytest.approx(force / env._bcfg.f_target)
