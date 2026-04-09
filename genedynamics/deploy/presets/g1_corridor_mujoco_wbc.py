"""G1 corridor + WBC controller running in MuJoCo.

This is the WBC test-bed preset.  The controller solves correctly on the
standing integration test; closed-loop corridor following is still under
tuning.  See ``controllers/wbc/README.md`` for status and known issues.

Stack:
* IO:         ``MujocoRobotIO`` (CPU NumPy MuJoCo, 500 Hz physics)
* Follower:   ``HumanoidContactScheduler`` + ``HumanoidFootstepPlanner``
              + ``HumanoidTaskBuilder`` + ``HumanoidUpperBodyMapper``
* Controller: ``HumanoidWBCController`` (QP inverse dynamics)
* Safety:     composite (joint + torque limits)
* Observers:  logger + recorder

Inherit and override one inner class to tune a single aspect without
copying the whole preset:

.. code-block:: python

    class MyWBCPreset(G1CorridorMujocoWBCPreset):
        class controller(G1CorridorMujocoWBCPreset.controller):
            class wbc(G1CorridorMujocoWBCPreset.controller.wbc):
                class weights(ComponentConfig):
                    pelvis = 12.0   # raise pelvis lateral tracking weight
"""

from __future__ import annotations

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

__all__ = ["G1CorridorMujocoWBCPreset"]


class G1CorridorMujocoWBCPreset(DeployConfig):
    """Default G1 narrow-corridor preset (sim, WBC controller)."""

    control_hz: float = 50.0
    sim_dt: float = 1.0 / 500.0
    max_steps: int = 5_000

    class runtime(ComponentConfig):
        name = "numpy"

    class robot(ComponentConfig):
        robot_type = "humanoid"
        model_id = "g1"

    class io(ComponentConfig):
        registry_key = "io.mujoco"
        sim_dt = 1.0 / 500.0
        keyframe_name = "stand"

    class controller(ComponentConfig):
        registry_key = "controller.wbc"

        # --- WBC task weights (see LimitsConfig, TaskWeightsConfig) --------
        # These are the defaults from WBCConfig; copy here to make tuning
        # explicit and diff-able across preset variants.
        class weights(ComponentConfig):
            contact = 80.0
            com = 8.0
            pelvis = 7.0
            torso = 6.0
            swing_foot = 8.0
            arm_posture = 1.2
            waist_posture = 1.6

        # --- WBC solver options -------------------------------------------
        class solver(ComponentConfig):
            use_osqp = True
            osqp_maxiter = 4000
            osqp_polish = True
            use_slsqp = True
            slsqp_maxiter = 120

        # --- Follower sub-components --------------------------------------
        # These are passed through to the HumanoidMujocoPipeline constructor
        # (and eventually to the standalone WBC follower once Phase 10.5
        # replaces HumanoidMujocoPipeline with a direct component stack).
        source_dt: float = 0.25
        control_dt: float = 0.02

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.02},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.90},
        ]

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/g1_corridor/wbc"},
        {"registry_key": "observer.recorder", "out_dir": "results/g1_corridor/wbc"},
    ]
