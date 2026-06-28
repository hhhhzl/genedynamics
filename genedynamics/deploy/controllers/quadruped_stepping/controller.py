"""QuadrupedSteppingController — Controller-protocol wrapper for the
minimal stepping-walk follower (v2).

Architecture note
-----------------
:class:`SteppingWalkFollowerMinimal` is a *self-contained simulator*: it owns
its own MuJoCo model and data, runs its own gait state machine, and produces
a full trajectory in one ``follow_plan()`` call.  It therefore does **not**
fit the step-by-step ``act(state, intent) → ControlCommand`` pattern used by
WBC or sport-mode controllers, which operate on externally owned IO.

The wrapper class exposed here:

* Provides ``spec``, ``runtime``, ``produces`` so it satisfies the structural
  :class:`~genedynamics.deploy.interfaces.controller.Controller` protocol.
* Provides ``reset()`` for re-initialising the internal walker between
  episodes.
* Provides ``rollout(plan_states, **kwargs) → dict`` as the **primary API**
  (returned dict has keys ``qpos``, ``qvel``, ``ctrl``, ``base_rpy``,
  ``contact_legs``, ``summary``, ``terminated``, etc.).
* ``act()`` raises ``NotImplementedError`` with a clear message — it is
  never called by the stepping-stones scripts, which drive the controller
  through ``rollout()`` directly.

To use from a script::

    from genedynamics.deploy.controllers.quadruped_stepping import (
        QuadrupedSteppingController,
        load_stepping_plan_from_seed_dir,
    )
    ctrl = QuadrupedSteppingController(cfg=MinimalFollowerConfig(gait="walk"))
    plan = load_stepping_plan_from_seed_dir(seed_dir)
    result = ctrl.rollout(plan["states"])
    qpos, qvel, ctrl_hist = result["qpos"], result["qvel"], result["ctrl"]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from genedynamics.deploy.controllers.quadruped_stepping.stepping_walker import (
    MinimalFollowerConfig,
    SteppingWalkFollowerMinimal,
)
from genedynamics.deploy.controllers.quadruped_stepping.governor import (
    GovernorConfig,
    SteppingReferenceGovernor,
)

__all__ = ["QuadrupedSteppingController"]


class QuadrupedSteppingController:
    """Batch controller wrapping :class:`SteppingWalkFollowerMinimal`.

    Parameters
    ----------
    cfg:
        Gait/geometry configuration.  Defaults to :class:`MinimalFollowerConfig`
        with ``gait="walk"`` and ``sim_dt=0.01``.
    stepping_scene:
        Optional dict produced by
        :func:`~genedynamics.tasks.stepping_stones.stepping_scene_to_dict`.
        When provided the walker embeds stepping-stones geometry into the
        MuJoCo XML before loading.
    model_xml_path:
        Optional path to a Go2 scene XML.  When *None* the walker auto-detects
        via ``MUJOCO_MENAGERIE_PATH``.
    """

    # Controller-protocol attributes
    spec: Any = None          # Go2 spec — not needed; walker uses its own model
    runtime: str = "numpy"
    produces: tuple[str, ...] = ("torque",)

    def __init__(
        self,
        cfg: Optional[MinimalFollowerConfig] = None,
        stepping_scene: Optional[Dict[str, Any]] = None,
        model_xml_path: Optional[str] = None,
    ) -> None:
        self._cfg = cfg or MinimalFollowerConfig()
        self._stepping_scene = stepping_scene
        self._model_xml_path = model_xml_path
        self._walker: Optional[SteppingWalkFollowerMinimal] = None

    # ------------------------------------------------------------------
    # Controller protocol stubs
    # ------------------------------------------------------------------

    def reset(self, io: Any = None) -> None:
        """Re-initialise the internal walker (creates a fresh MuJoCo sim)."""
        self._walker = SteppingWalkFollowerMinimal(
            model_xml_path=self._model_xml_path,
            cfg=self._cfg,
            stepping_scene=self._stepping_scene,
        )

    def act(self, state: Any, intent: Any) -> Any:
        """Not supported — use :meth:`rollout` instead.

        :class:`QuadrupedSteppingController` owns its own internal MuJoCo
        simulation; the step-by-step ``act()`` pattern requires an externally
        owned IO to be the source of truth, which creates a conflicting sim.
        """
        raise NotImplementedError(
            "QuadrupedSteppingController is batch-only. "
            "Call rollout(plan_states) to run a full episode."
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def rollout(
        self,
        plan_states: Any,
        *,
        max_segments: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a full stepping-walk episode from ``plan_states``.

        Parameters
        ----------
        plan_states:
            ``(N, D)`` float array of planner states as produced by
            :func:`load_stepping_plan_from_seed_dir`.
        max_segments:
            Cap on the number of gait intervals to execute.  *None* = run all.

        Returns
        -------
        dict
            Keys: ``qpos`` (T, nq), ``qvel`` (T, nv), ``ctrl`` (T, nu),
            ``base_rpy`` (T, 3), ``contact_legs`` (list[list[str]]),
            ``swing_ref`` (dict), ``interval_stats`` (list), ``summary``
            (dict), ``terminated`` (bool), ``termination_reason`` (str|None).
        """
        if self._walker is None:
            self.reset()
        assert self._walker is not None
        return self._walker.follow_plan(plan_states, max_segments=max_segments)

    # ------------------------------------------------------------------
    # Governed API (stepping reference governor → walker hook)
    # ------------------------------------------------------------------

    def _governor_config(self, governor_cfg: Optional[GovernorConfig]) -> GovernorConfig:
        if governor_cfg is not None:
            return governor_cfg
        c = self._cfg
        return GovernorConfig(
            gait=str(c.gait),
            x_f_nominal=float(c.x_f_nominal),
            x_r_nominal=float(c.x_r_nominal),
            y_L_nominal=float(c.y_L_nominal),
            y_R_nominal=float(c.y_R_nominal),
            step_width=float(c.step_width),
            half_pair_length=float(c.leg_half_length),
            centerline_y=float(c.centerline_y),
            min_foot_z=float(c.min_foot_z),
        )

    def build_gait_reference(
        self,
        candidate_states: Sequence[Any],
        *,
        planner_best_idx: int = 0,
        candidate_costs: Optional[Sequence[float]] = None,
        governor_cfg: Optional[GovernorConfig] = None,
    ):
        """Govern a planner result and return ``(gait_reference, ranking_reports)``.

        Does **not** touch the simulator; useful for offline gate/diagnostics checks.
        """
        gov = SteppingReferenceGovernor(
            cfg=self._governor_config(governor_cfg),
            stepping_scene=self._stepping_scene,
        )
        return gov.govern(
            candidate_states,
            planner_best_idx=int(planner_best_idx),
            candidate_costs=candidate_costs,
        )

    def rollout_governed(
        self,
        candidate_states: Sequence[Any],
        *,
        planner_best_idx: int = 0,
        candidate_costs: Optional[Sequence[float]] = None,
        governor_cfg: Optional[GovernorConfig] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Govern a planner result, then execute the governed gait on the walker.

        ``candidate_states`` is the list of planner candidate trajectories (or a single
        ``(N, D)`` array, which is treated as a one-candidate list).  Returns the
        :meth:`SteppingWalkFollowerMinimal.follow_gait_reference` result, augmented with
        ``governor_ranking`` (the per-candidate ranking reports).
        """
        cands = candidate_states
        # Only probe shape when an ndarray is passed; never np.asarray the whole list
        # (candidate trajectories can be ragged / variable-length and would raise).
        if isinstance(candidate_states, np.ndarray):
            if candidate_states.ndim == 2:      # a single (N, D) trajectory
                cands = [candidate_states.astype(np.float32)]
                planner_best_idx = 0
            elif candidate_states.ndim == 3:    # stacked candidates (K, N, D)
                cands = list(candidate_states.astype(np.float32))
        gait_ref, reports = self.build_gait_reference(
            cands,
            planner_best_idx=planner_best_idx,
            candidate_costs=candidate_costs,
            governor_cfg=governor_cfg,
        )
        if self._walker is None:
            self.reset()
        assert self._walker is not None
        result = self._walker.follow_gait_reference(gait_ref, max_steps=max_steps)
        result["governor_ranking"] = reports
        return result
