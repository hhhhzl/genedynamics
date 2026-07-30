"""Receding-horizon execution bridge for MBD-family solvers.

The genedynamics ``Solver.solve(x0, horizon)`` is a *single-shot, full-horizon*
planner: it returns one open-loop trajectory. DIAL-MPC's defining behaviour is
the opposite -- a receding-horizon controller that, at every real control step:

    1. re-plans from the *current* state, **warm-started** from the previous
       (shifted) solution, using only a *few* diffusion steps;
    2. executes only the first control ``u_0``;
    3. shifts the remaining plan and repeats.

This module provides ONE solver-agnostic bridge,
:class:`RecedingHorizonController`, that turns any planner exposing the small
:class:`WarmStartPlanner` capability into a DIAL-MPC-like real-time controller.
The bridge owns no algorithm math -- the per-step reverse update lives entirely
in the wrapped planner (``dial``, ``mbd``, ``2go`, ...). Composing a planner's
reverse-update with this bridge is exactly DIAL-MPC:

    DIAL-MPC  ==  RecedingHorizonController( <control-space MBD planner> )

The two behaviours that make a plain long-horizon solver "DIAL-like" -- and that
``experiments/common/d3il_mpc.py`` (chunked replan, no warm-start) does NOT do --
are **warm-starting the plan variable** across steps and **annealing the
diffusion step count** (many on the first step, few thereafter). Both are driven
here and delegated to the planner via :class:`WarmStartPlanner`.

Note on the diffusion schedule: different MBD solvers anneal differently. DIAL
uses a *geometric per-node noise scale* decoupled from the step count, so "few
steps" is just a shorter factor prefix. DDPM-chain solvers (mbd/2go) index a
cumulative ``alpha_bar`` chain, so "few steps" is a strided sub-sample of that
chain. The bridge therefore does not build the schedule itself; it asks the
planner for one of length ``n_diffuse`` via ``make_schedule`` (default:
``planner.make_schedule``) and hands it back to ``replan``. Each planner
interprets the schedule in its own annealing convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class WarmStartPlanner(Protocol):
    """Capability a solver must expose to be driven by the bridge.

    Intentionally tiny so it can be implemented natively by a new solver (the
    ``dial`` backend) OR by a non-invasive external adapter/subclass over an
    existing solver (mbd/2go) WITHOUT editing that solver's source.

    The ``plan_var`` is the solver's own control representation -- DIAL's spline
    *nodes* ``(Hnode+1, nu)`` or a dense control ``(Hsample+1, nu)``. The bridge
    treats it as an opaque pytree/array and only moves it between calls.
    """

    def init_plan_var(self) -> Any:
        """Return the initial (cold-start) plan variable, e.g. zeros."""
        ...

    def replan(self, state: Any, warm_start: Any, schedule: Any, rng: Any, t0: Any = 0.0) -> Any:
        """Run the reverse update from ``warm_start`` for ``len(schedule)`` steps.

        ``t0`` is the receding-horizon real-step offset, for time-dependent
        rewards (e.g. gait phase); planners that ignore time accept it via a
        default. Returns the refined plan variable (same structure as
        ``warm_start``). This is the ONLY place the per-solver diffusion math runs.
        """
        ...

    def first_action(self, plan_var: Any) -> Any:
        """Extract the first executable control ``u_0`` from ``plan_var``
        (DIAL: ``node2u(nodes)[0]``; dense solvers: ``plan_var[0]``)."""
        ...

    def shift(self, plan_var: Any) -> Any:
        """Shift the plan one control step forward for the next warm start
        (DIAL: roll dense ``u`` by -1, zero the tail, refit nodes)."""
        ...

    def make_schedule(self, n_diffuse: int) -> Any:
        """Return a per-step anneal schedule of length ``n_diffuse`` in the
        planner's own convention (factor prefix / strided index subset)."""
        ...


@dataclass
class RecedingHorizonResult:
    """Closed-loop rollout produced by the bridge."""

    states: List[Any]
    actions: List[Any]
    plan_vars: List[Any] = field(default_factory=list)
    infos: List[Any] = field(default_factory=list)

    @property
    def horizon(self) -> int:
        return len(self.actions)


def _default_split_rng(rng: Any):
    """Split an rng key, supporting jax PRNGKeys and falling back to a passthrough.

    Lazily imports jax so this module has no hard jax dependency and can be
    unit-tested with a custom ``split_rng`` (e.g. a numpy counter)."""
    try:
        import jax

        return tuple(jax.random.split(rng, 2))
    except Exception:
        # Stateless passthrough: callers wanting reproducible numpy rng should
        # pass their own ``split_rng``.
        return rng, rng


class RecedingHorizonController:
    """Solver-agnostic DIAL-MPC-style receding-horizon driver (the "bridge").

    Args:
        planner: any object satisfying :class:`WarmStartPlanner`.
        step_fn: real dynamics advance ``(state, action) -> next_state``. For
            DIAL parity this is the SAME env-step used inside the planner's
            rollout (e.g. ``dynamics.step`` / ``DynamicsToEnvAdapter.step``).
        n_steps: number of real control steps to execute (episode length, NOT
            the planning horizon).
        n_diffuse_init: diffusion steps on the FIRST replan (large; warms up the
            cold zero plan). Mirrors DIAL's ``Ndiffuse_init``.
        n_diffuse: diffusion steps on every subsequent replan (small; refines a
            warm-started plan). Mirrors DIAL's ``Ndiffuse``.
        make_schedule: optional override for ``planner.make_schedule``.
        split_rng: optional override for rng splitting (default: jax-aware).
        collect_plan_vars: store each step's plan var (diagnostics / warm-start
            inspection); off by default to keep memory flat.
        collect_states: store every full dynamics state. Disable for long MJX
            experiments whose evaluators replay the executed actions; the
            result then keeps only initial and final state.
        synchronize_steps: block after each real step so asynchronous JAX
            execution cannot queue an episode's worth of device buffers.
    """

    def __init__(
        self,
        planner: WarmStartPlanner,
        step_fn: Callable[[Any, Any], Any],
        *,
        n_steps: int,
        n_diffuse_init: int,
        n_diffuse: int,
        make_schedule: Optional[Callable[[int], Any]] = None,
        split_rng: Optional[Callable[[Any], Any]] = None,
        collect_plan_vars: bool = False,
        collect_states: bool = True,
        synchronize_steps: bool = False,
    ) -> None:
        self.planner = planner
        self.step_fn = step_fn
        self.n_steps = int(n_steps)
        self.n_diffuse_init = int(n_diffuse_init)
        self.n_diffuse = int(n_diffuse)
        self._make_schedule = make_schedule or planner.make_schedule
        self._split_rng = split_rng or _default_split_rng
        self.collect_plan_vars = bool(collect_plan_vars)
        self.collect_states = bool(collect_states)
        self.synchronize_steps = bool(synchronize_steps)

    def n_diffuse_at(self, t: int) -> int:
        """Diffusion-step count for real step ``t`` (init on t==0, else steady)."""
        return self.n_diffuse_init if t == 0 else self.n_diffuse

    def run(self, x0: Any, rng: Any) -> RecedingHorizonResult:
        """Execute the closed-loop receding-horizon rollout."""
        state = x0
        plan_var = self.planner.init_plan_var()
        states: List[Any] = [state]
        actions: List[Any] = []
        plan_vars: List[Any] = []
        infos: List[Any] = []

        for t in range(self.n_steps):
            schedule = self._make_schedule(self.n_diffuse_at(t))
            rng, sub = self._split_rng(rng)

            # 1) re-plan from current state, warm-started, few steps.
            # Pass the real-step index t as t0 so time-dependent rewards (e.g.
            # gait phase) advance with execution; planners that ignore time take
            # t0 via **kwargs / a default and are unaffected.
            plan_var = self.planner.replan(state, plan_var, schedule, sub, t0=t)

            # 2) execute only the first control on the REAL dynamics
            u0 = self.planner.first_action(plan_var)
            state = self.step_fn(state, u0)
            if self.synchronize_steps:
                try:
                    import jax
                    state, plan_var, u0 = jax.block_until_ready((state, plan_var, u0))
                except (ImportError, TypeError):
                    pass

            actions.append(u0)
            if self.collect_states:
                states.append(state)
            if self.collect_plan_vars:
                plan_vars.append(plan_var)

            # 3) shift the remaining plan to warm-start the next step
            plan_var = self.planner.shift(plan_var)

        if not self.collect_states and self.n_steps:
            states.append(state)
        return RecedingHorizonResult(
            states=states, actions=actions, plan_vars=plan_vars, infos=infos
        )


class SolverPlannerAdapter:
    """Drive ANY genedynamics solver through the bridge -- WITHOUT editing it.

    Wraps a solver exposing ``solve(x0, horizon=..., rng_key=...) -> Trajectory``
    (MBD / CFS-MBD / 2GO / MPPI / CEM / ...) in the :class:`WarmStartPlanner`
    protocol. This is the general answer to "how do I run <solver> + bridge":
    the bridge is solver-agnostic, only DIAL implemented the protocol natively,
    and this adapter supplies it for everyone else.

    Default behaviour is RE-SOLVE: each ``replan`` calls ``solver.solve`` from the
    current state, so the solver runs its OWN full diffusion (its own ``Ndiffuse``,
    schedule, geometry shaping, etc.) -- closed-loop MPC, same semantics as
    ``experiments/common/d3il_mpc.run_mpc_episode`` but behind the unified bridge.
    The bridge's ``n_diffuse``/``n_diffuse_init`` are therefore unused here (each
    replan is a full solve); ``plan_var`` is the dense action sequence ``(H, nu)``.

    Warm start (optional): if the solver's ``solve`` accepts an initial-plan
    kwarg, pass its name as ``warm_start_kwarg`` and the adapter feeds the shifted
    previous plan -- then the bridge's warm-start (small per-step diffusion) gains
    real value. ``set_initial_state`` is called on the solver's dynamics when
    present (some flat planners linearise/seed around x0 there).

    The solver keeps its native env+reward/energy: for 2GO that is the constraint
    SDF manifold on the corridor / stepping-stones task -- exactly where its
    geometry shaping matters.
    """

    def __init__(
        self,
        solver: Any,
        *,
        horizon: Optional[int] = None,
        rng_key_name: str = "rng_key",
        warm_start_kwarg: Optional[str] = None,
    ) -> None:
        self.solver = solver
        self.horizon = int(horizon if horizon is not None else getattr(solver, "horizon", 20))
        self.rng_key_name = rng_key_name
        self.warm_start_kwarg = warm_start_kwarg

    @staticmethod
    def _as_array(actions: Any):
        import numpy as np
        if hasattr(actions, "shape"):           # already an array
            return actions
        return np.asarray(list(actions), dtype=np.float32)

    def init_plan_var(self) -> Any:
        return None                              # cold start: first replan solves from scratch

    def replan(self, state: Any, warm_start: Any, schedule: Any, rng: Any, t0: Any = 0.0) -> Any:
        dyn = getattr(self.solver, "dynamics", None)
        if dyn is not None and hasattr(dyn, "set_initial_state"):
            try:
                dyn.set_initial_state(state)
            except Exception:
                pass
        kw: dict = {"horizon": self.horizon}
        if rng is not None:
            kw[self.rng_key_name] = rng
        if self.warm_start_kwarg is not None and warm_start is not None:
            kw[self.warm_start_kwarg] = warm_start
        traj = self.solver.solve(state, **kw)
        actions = getattr(traj, "actions", traj)
        return self._as_array(actions)

    def first_action(self, plan_var: Any) -> Any:
        return plan_var[0]

    def shift(self, plan_var: Any) -> Any:
        if plan_var is None:
            return None
        import numpy as np
        a = np.asarray(plan_var)
        return np.concatenate([a[1:], a[-1:]], axis=0)

    def make_schedule(self, n_diffuse: int) -> Any:
        # The wrapped solver controls its own diffusion length; the bridge's
        # n_diffuse is informational here. Returned for protocol completeness.
        return list(range(int(n_diffuse)))
