# `genedynamics/deploy/` Refactor Plan

> **Goal:** Rebuild `deploy/` into a clean orchestration layer that (a) reuses everything `genedynamics/` already provides, (b) supports multiple **runtime backends** (JAX / Torch / NumPy / Rust) AND multiple **execution backends** (MuJoCo / MJX / Brax / Isaac / real hardware / ROS), (c) supports multiple controllers (Sport-mode / RL / WBC / PD / MPC), and (d) covers everything `spark` does while scaling to higher-DoF systems and more tasks.

---

## 0. Naming: disambiguate "backend"

`backend` is overloaded. We use three distinct terms throughout this doc and the refactor:

| Term | Meaning | Lives in | Examples |
|---|---|---|---|
| **RuntimeBackend** | Compute framework: tensor ops, JIT, autodiff | `core/backends/runtime/` (already exists) | `jax`, `torch`, `numpy`, future: `rust`, `triton` |
| **PhysicsBackend** | Simulation kernel | `core/backends/physics.py` (already exists) + `envs/*` | `mujoco`, `mjx`, `brax`, `isaac`, `analytic` |
| **RobotIO** | Hardware/sim IO adapter (read state, write control) | **`deploy/io/`** (new) | `MujocoRobotIO`, `UnitreeG1RealIO`, `Go2RealIO`, `RosRobotIO` |

`RobotIO` is what `spark` calls "Agent". We rename to avoid the loaded RL term and to make the read/write contract explicit.

---

## 1. Existing infrastructure to REUSE (do not duplicate)

The following already exist in `genedynamics/`. The new `deploy/` must depend on them — never reimplement.

| Concept | Existing module | What it provides |
|---|---|---|
| Runtime backends (JAX/Torch/NumPy) | `core/backends/runtime/` + `core/backends/base.py` | `Backend` protocol; `BackendManager` |
| Multi-backend registry pattern | `core/registry/` (`base.py`, `backends.py`, `action_filters.py`, `projections.py`, `environments.py`, `solvers.py`) | `BaseRegistry` w/ entry-point lazy loading; **two-level** `{type: {backend: impl}}` for multi-backend components |
| Dynamics model abstraction | `core/dynamics/base.py` | `DynamicsModel.step / rollout / rollout_batch` |
| Solver protocol | `core/solvers/base.py` | `Solver(dynamics, energy, backend).solve(x0, horizon) → Trajectory` |
| Energy functional | `core/energy/base.py` | `EnergyFunctional` (task + constraint + entropy) |
| Constraint pipeline | `core/constraints/{terms,convexify,operators,schedulers,solvers}/` | Full barrier/CBF/CFS/QP stack — **deploy uses this for safety filters** |
| Types | `core/types.py` | `State`, `Action`, `Trajectory` (intentionally `Any` for backend flexibility) |
| Position / success extraction | `core/task_spec.py` | `TaskSpec.extract_position / success_criterion` |
| Env protocol | `envs/base_env.py` | `BaseEnv.reset / step / transition / jax_transition` + `physics_backend`, `render_backend` attributes |
| Env factory | `envs/factories.py` | `make_env(name, **kw)`; supports humanoid/quadruped/drone × {analytic, mjx, brax, mujoco, isaac} |
| Env adapters | `envs/adapters/{unified,gymnasium,brax}_adapter.py` | Auto-detect external sims |
| **Robot kinematics protocol** | `envs/robots/base.py` | `RobotModel`: n_dof, joint_names, FK, IK, jacobian, forward_dynamics |
| Robot registry (meta) | `robots/registry.py` | `(robot_type, model_id) → RobotEntry(nq, nv, act_dim, mjcf_path, env_factory_name)` |
| Task specs | `tasks/{humanoid,quadruped,uav3d}/spec.py` | per-task `nq, nv, act_dim`, control limits, position layout |
| Stepping-stones scene | `tasks/stepping_stones/` | `SteppingStonesScene`, kinematics |

**Implication:** the refactored `deploy/` must shrink dramatically. The current `factory.py` `make_*_env / make_*_planner` is a duplicate of `envs/factories.py` + `core/registry/` and should be deleted.

---

## 2. What `deploy/` SHOULD own (and only this)

`deploy/` is the **execution / orchestration** layer. Its job is:

1. **Compose** an env / robot / controller / safety / IO bundle from a config
2. **Run** a closed loop at the right frequency, in sim or on hardware
3. **Bridge** between the planner's output (a `Trajectory`) and the controller's input
4. **Adapt** to hardware specifics (SDK calls, ROS topics, sensor fusion)
5. **Telemetry**: log, render, replay, shadow

It does **not** own: robots, tasks, environments, dynamics, solvers, energies, runtime backends, registries, kinematics. Those live where they already live.

---

## 3. Where the G1 model spec should live (per your point 2)

**Move out of `deploy/`:**

| Current location | New location | Reason |
|---|---|---|
| `deploy/followers/humanoid/models/g1_model.py` | `genedynamics/robots/g1/spec.py` | Robot metadata is not deploy-specific |
| `deploy/followers/humanoid/models/__init__.py` | `genedynamics/robots/g1/__init__.py` | — |
| (currently absent) G1 kinematics | `genedynamics/envs/robots/g1.py` (new, implements `RobotModel`) | Matches existing `drone.py`, `manipulator.py` |
| (currently absent) G1 MJCF asset path resolver | `genedynamics/robots/g1/assets.py` | Asset resolution belongs with the robot |
| `robots/registry.py` (existing) gains a `register_humanoid("g1", ...)` call | `robots/g1/__init__.py` runs registration on import | Single source of robot truth |

Same pattern for future H1 / H1.2 / Go2 / etc. After this move, **`deploy/` contains zero robot-specific files**.

```
genedynamics/
├── robots/
│   ├── registry.py            # already exists
│   ├── g1/
│   │   ├── __init__.py        # registers G1 in registry on import
│   │   ├── spec.py            # joint names, limits, Kp/Kd, sites, dof maps
│   │   └── assets.py          # MJCF path resolver
│   ├── h1/                    # future
│   └── go2/                   # future (currently in deploy/backends/unitree_go2.py — move IO out)
└── envs/
    └── robots/
        ├── base.py            # RobotModel protocol (already exists)
        ├── drone.py           # already exists
        ├── manipulator.py     # already exists
        └── g1.py              # NEW: G1 kinematics implementing RobotModel
```

---

## 4. Target `deploy/` layout

```
genedynamics/deploy/
├── __init__.py
├── cli.py                          # thin: parse args → load config → run()
├── pipeline.py                     # main loop (sim and real share this)
│
├── config/
│   ├── __init__.py
│   ├── schema.py                   # nested-class config base + helpers
│   └── presets/
│       ├── g1_corridor_mujoco.py
│       ├── g1_corridor_real.py
│       ├── g1_corridor_mjx.py
│       └── go2_flat_mujoco.py
│
├── interfaces/                     # protocols only — no implementations
│   ├── __init__.py
│   ├── robot_io.py                 # RobotIO protocol (sim/real symmetric)
│   ├── controller.py               # Controller protocol
│   ├── follower.py                 # TrajectoryFollower protocol
│   ├── safety.py                   # SafetyFilter protocol
│   ├── observers.py                # Telemetry/Render/Logger protocol
│   └── messages.py                 # TypedDict / dataclass: RobotState, ControlCmd, Intent
│
├── io/                             # ← formerly "backends" (HW IO)
│   ├── __init__.py
│   ├── base.py                     # BaseRobotIO with shared lifecycle
│   ├── mujoco_io.py                # uses existing envs/humanoid_base_physics.py
│   ├── mjx_io.py                   # uses envs/humanoid_mjx.py
│   ├── brax_io.py                  # uses envs/humanoid_brax.py
│   ├── isaac_io.py                 # via envs/drone_full_3d_isaac.py pattern
│   ├── stub_io.py
│   ├── unitree_g1_io.py            # ← real G1 SDK, replaces stub
│   ├── unitree_go2_io.py           # ← moved from backends/unitree_go2.py
│   └── ros2_io.py                  # ← future: ROS2 topics
│
├── followers/                      # trajectory → high-level intent
│   ├── __init__.py
│   ├── base.py                     # TrajectoryFollower protocol + base impl
│   ├── plan_schema.py              # ← from common/plan_schema.py (14D corridor)
│   ├── plan_adapter.py             # ← from common/plan_adapter.py
│   ├── traversal_intent.py         # ← from common/traversal_intent.py (good, keep)
│   ├── corridor_follower.py        # 14D corridor trajectory → TraversalIntent stream
│   ├── waypoint_follower.py        # generic waypoint follower
│   └── replay_follower.py          # play recorded intent
│
├── controllers/                    # intent → control command (this is "Policy" in spark)
│   ├── __init__.py
│   ├── base.py                     # Controller protocol
│   ├── sport_mode/
│   │   ├── __init__.py
│   │   └── unitree_loco.py         # wraps Unitree LocoClient (option 1)
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── base.py                 # loads (obs_fn, action_fn, ckpt) policies
│   │   ├── unitree_rl_gym.py       # option 2: G1 walking policy from unitree_rl_gym
│   │   └── isaaclab_policy.py      # future
│   ├── wbc/                        # ← option 3, current 1273-line solver split
│   │   ├── __init__.py
│   │   ├── controller.py           # ~150 lines: Controller protocol impl
│   │   ├── task_stack.py           # COM / pelvis / torso / swing-foot / posture tasks
│   │   ├── contact_blocks.py       # support contact equality blocks
│   │   ├── friction_cone.py        # inequality constraints
│   │   ├── qp_builder.py           # build (H, g, A, b, lb, ub) for QP
│   │   ├── qp_solver_adapter.py    # delegates to core/constraints/solvers/{osqp,cvxopt}
│   │   └── config.py               # GROUPED config (task_weights / regularization / limits / contact)
│   ├── pd/
│   │   └── joint_pd.py             # joint-space PD baseline
│   └── mpc/                        # future, can wrap core/solvers
│       └── linear_mpc.py
│
├── safety/                         # SafetyFilter implementations
│   ├── __init__.py
│   ├── base.py                     # SafetyFilter protocol
│   ├── joint_limit.py              # clip to robot.spec limits
│   ├── torque_limit.py
│   ├── self_collision.py           # uses envs/robots/g1.py kinematics
│   └── cbf_filter.py               # ← bridges to core/constraints/operators/qp + convexify/cbf
│
├── tasks/                          # deploy-side task adapters (NOT redefine task)
│   ├── __init__.py                 # imports from genedynamics/tasks/*/spec.py
│   ├── base.py                     # ExecutionTask protocol: reset(io), step(state) → done, info
│   ├── corridor_follow.py          # consumes 14D plan + envs/humanoid_corridor_2d
│   ├── flat_locomotion.py
│   └── teleop.py
│
├── modes/                          # high-level "what to run"
│   ├── __init__.py
│   ├── base.py                     # ExecutionMode protocol (already good)
│   ├── sim.py                      # closed-loop sim
│   ├── real.py                     # closed-loop hardware
│   ├── shadow.py                   # real IO + offline controller, no actuation
│   └── replay.py                   # offline rollout from logged data
│
├── localization/                   # ← keep as-is, already clean
│   ├── base_plugin.py
│   ├── mock_plugin.py
│   ├── ros2_odometry_plugin.py
│   └── vicon_shm_plugin.py
│
├── observers/                      # cross-cutting telemetry
│   ├── __init__.py
│   ├── base.py
│   ├── logger.py                   # structured logs (parquet/jsonl)
│   ├── recorder.py                 # state/action timeseries
│   ├── mujoco_renderer.py          # ← from viz/mujoco_render.py
│   ├── web_viz.py                  # ← from viz/web_viz.py
│   └── ros2_publisher.py           # future
│
└── runtime/                        # rate, threads, timing
    ├── rate_limiter.py
    ├── decimation.py               # control_rate / sim_rate decoupling
    └── clock.py
```

### Files / dirs to DELETE

| Path | Reason |
|---|---|
| `deploy/factory.py` | Replaced by `core/registry/` + nested-class config |
| `deploy/sim_plan/humanoid_corridor/` | Duplicate of followers/humanoid; merge into `controllers/wbc/` |
| `deploy/followers/humanoid/mujoco/ik_solver_legacy.py` | Dead legacy shim |
| `deploy/followers/humanoid/mujoco/joint_tracker_legacy.py` | Dead legacy shim |
| `deploy/followers/humanoid/models/` | Moved to `genedynamics/robots/g1/` |
| `deploy/followers/humanoid/mujoco/pipeline.py` | Logic split between `pipeline.py`, `io/mujoco_io.py`, `controllers/wbc/` |
| `deploy/followers/quadruped/`, `deploy/followers/mobile/` | Empty stubs; recreate when needed under controllers |
| `deploy/controllers/` (existing empty dir) | Replaced by new `controllers/` tree |
| `deploy/backends/` (current) | Renamed to `deploy/io/`, contents migrated |
| `deploy/quadruped_planner.py` | Belongs under `controllers/` if kept; otherwise delete |
| `deploy/profiles/` | Replaced by nested-class config presets |
| `deploy/viz/` | Renamed to `deploy/observers/` |
| `deploy/task_config.py` | Folded into per-task config preset |

---

## 5. Core interfaces

All in `deploy/interfaces/`. Protocols only — implementations elsewhere.

### 5.1 `messages.py` — typed payloads

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RobotState:
    t: float
    qpos: np.ndarray            # generalized coordinates (incl. floating base if any)
    qvel: np.ndarray
    base_pose: Optional[np.ndarray] = None     # 7-vec [x,y,z,qw,qx,qy,qz]
    base_twist: Optional[np.ndarray] = None    # 6-vec
    contact: Optional[dict] = None             # {site_name: 6-vec wrench}
    imu: Optional[dict] = None                 # {acc, gyro, ori}
    extras: Optional[dict] = None              # backend-specific passthrough

@dataclass
class ControlCommand:
    kind: str                   # "joint_pos" | "joint_torque" | "loco" | "mixed"
    joint_pos: Optional[np.ndarray] = None
    joint_vel: Optional[np.ndarray] = None
    joint_torque: Optional[np.ndarray] = None
    kp: Optional[np.ndarray] = None
    kd: Optional[np.ndarray] = None
    loco_cmd: Optional["LocoCommand"] = None   # for sport-mode
    extras: Optional[dict] = None

@dataclass
class Intent:                   # output of TrajectoryFollower
    t: float
    base_pos_xy: np.ndarray     # (2,)
    base_yaw: float
    base_height: float
    base_lin_vel: np.ndarray    # (2,)
    base_yaw_rate: float
    torso_yaw: Optional[float] = None
    arm_targets: Optional[dict] = None
    contact_phase: Optional[str] = None
    extras: Optional[dict] = None
```

These are dataclasses — **not** dicts (your point 5: better than spark). Backwards compatible with the existing `TraversalIntent` (which becomes a builder for `Intent`).

### 5.2 `robot_io.py` — sim/real symmetric IO

```python
class RobotIO(Protocol):
    spec: "RobotSpec"               # from genedynamics/robots/<id>/spec.py
    physics_backend: Optional["PhysicsBackend"]   # None for real

    def reset(self) -> RobotState: ...
    def get_state(self) -> RobotState: ...
    def send_control(self, cmd: ControlCommand) -> None: ...
    def step(self, dt: float) -> None: ...        # advance sim; no-op on real
    def close(self) -> None: ...
```

`MujocoRobotIO`, `MjxRobotIO`, `BraxRobotIO`, `IsaacRobotIO` all wrap an existing `envs/*` env. `UnitreeG1RealIO` wraps the SDK. Pipeline never branches on sim/real.

### 5.3 `follower.py`, `controller.py`, `safety.py`

```python
class TrajectoryFollower(Protocol):
    def reset(self, plan) -> None: ...
    def step(self, t: float, state: RobotState) -> Intent: ...

class Controller(Protocol):
    runtime: "Backend"          # core/backends/runtime — jax/torch/numpy/rust
    spec: "RobotSpec"
    def reset(self, io: RobotIO) -> None: ...
    def act(self, state: RobotState, intent: Intent) -> ControlCommand: ...

class SafetyFilter(Protocol):
    def filter(self, state: RobotState, cmd: ControlCommand) -> ControlCommand: ...
```

Notice: **`Controller.runtime` is a runtime backend**. A WBC controller can be `numpy` (osqp); an RL controller can be `torch` or `jax`; a future Rust solver can be `rust`. This makes runtime-backend pluggability first-class — directly answering your point 1.

### 5.4 `observers.py`

```python
class Observer(Protocol):
    def on_step(self, t: float, state: RobotState, intent: Intent,
                cmd: ControlCommand, info: dict) -> None: ...
    def on_episode_end(self, summary: dict) -> None: ...
```

Multiple observers can attach to a pipeline (logger + renderer + ros publisher) without controller knowing.

---

## 6. The single `pipeline.py`

```python
def run(cfg: DeployConfig):
    runtime  = registry.get_runtime_backend(cfg.runtime.name)        # core/backends/runtime
    robot    = registry.get_robot(cfg.robot.id)                      # genedynamics/robots
    io       = initialize(cfg.io,         robot=robot, runtime=runtime)
    follower = initialize(cfg.follower,   robot=robot)
    ctrl     = initialize(cfg.controller, robot=robot, runtime=runtime)
    safety   = initialize(cfg.safety,     robot=robot, runtime=runtime)
    task     = initialize(cfg.task,       robot=robot)
    obs      = [initialize(o) for o in cfg.observers]
    rate     = RateLimiter(cfg.control_hz)

    plan = task.load_plan()                          # e.g. 14D corridor json
    follower.reset(plan)
    state = io.reset()
    ctrl.reset(io); safety_state = safety_init(safety, robot)

    for step in range(cfg.max_steps):
        rate.tick()
        intent = follower.step(state.t, state)
        cmd    = ctrl.act(state, intent)
        cmd    = safety.filter(state, cmd)
        io.send_control(cmd)
        io.step(cfg.sim_dt)                           # no-op on real
        state  = io.get_state()
        info   = task.step(state, intent, cmd)
        for o in obs: o.on_step(state.t, state, intent, cmd, info)
        if info.get("done"): break

    for o in obs: o.on_episode_end(task.summary())
```

This loop runs **identically** on sim and real. Sim/real difference is one config field: `cfg.io.class_name = "MujocoRobotIO" | "UnitreeG1RealIO"`. Same for runtime backend, controller, follower.

---

## 7. Config system (nested classes, registry-driven)

Reuse `core/registry/`. Add a thin nested-class helper (≤30 lines) similar to spark's `init_member_classes`, but resolving via `core/registry/` instead of module-name reflection. Concrete preset:

```python
# deploy/config/presets/g1_corridor_mujoco.py
from genedynamics.deploy.config.schema import DeployConfig

class G1CorridorMujocoConfig(DeployConfig):
    control_hz = 200
    sim_dt = 1/1000
    max_steps = 20_000

    class runtime:
        name = "numpy"                        # core/backends/runtime

    class robot:
        id = "g1"                             # → genedynamics/robots/g1

    class io:
        registry_key = "io.mujoco"
        mjcf = "${robot.assets.mjcf}"
        physics_backend = "mujoco"

    class follower:
        registry_key = "follower.corridor"
        plan_path = "${task.plan_path}"
        plan_dt_target = 0.005

    class controller:
        registry_key = "controller.wbc"
        class config:
            class task_weights:
                com = 100.0
                pelvis_orient = 50.0
                swing_foot_pos = 200.0
                # ...
            class regularization:
                ddq = 1e-4
                lambda_ = 1e-6
            class limits:
                use_torque_bound = True
                use_friction_cone = True
            class contact:
                support_eq_hard = False        # ← was True; soft is more robust
                friction_mu = 0.7

    class safety:
        registry_key = "safety.composite"
        filters = ["safety.joint_limit", "safety.torque_limit"]

    class task:
        registry_key = "task.corridor_follow"
        plan_path = "results/humanoid/corridor_2d/.../trajectory.json"

    observers = [
        {"registry_key": "observer.logger",  "out_dir": "logs/"},
        {"registry_key": "observer.mujoco_renderer", "fps": 60},
    ]
```

To switch to real G1: change `io.registry_key` to `"io.unitree_g1"` and `physics_backend = None`. To switch to RL: change `controller.registry_key` to `"controller.rl.unitree_rl_gym"` and `runtime.name = "torch"`. To switch to sport-mode: `"controller.sport_mode.unitree_loco"`. The pipeline file does not change.

---

## 8. Mapping each spark feature → our home

| spark concept | spark file | our equivalent | improvement |
|---|---|---|---|
| `BasePipeline` loop | `spark_pipeline/base/base_pipeline.py` | `deploy/pipeline.py` | Single loop for sim+real (spark has one too, ✓) |
| `BaseAgent` | `spark_agent/base/base_agent.py` | `deploy/interfaces/robot_io.py` (`RobotIO`) | Renamed; explicit read/write contract; typed `RobotState` instead of dict |
| `MujocoAgent` | `spark_agent/mujoco/.../mujoco_agent.py` | `deploy/io/mujoco_io.py` | Reuses `envs/humanoid_base_physics.py` instead of duplicating |
| `G1RealAgent` | `spark_agent/.../g1_real_agent.py` | `deploy/io/unitree_g1_io.py` | Currently the gap (spark also incomplete); make it real |
| `BasePolicy` / `ControlPolicy` | `spark_policy/base/base_policy.py` | `deploy/interfaces/controller.py` (`Controller`) | Adds `runtime` field → first-class JAX/Torch/Rust pluggability |
| `G1WBCPolicy` | `spark_policy/.../g1_wbc_policy.py` | `deploy/controllers/wbc/` (split) | Split out of 1273-line god file; grouped config |
| `UserRLPolicy` | `spark_policy/.../user_rl_policy.py` | `deploy/controllers/rl/` | Multiple RL backends (unitree_rl_gym, IsaacLab, custom) |
| `BaseSafeController` | `spark_safe_algo/base/...` | `deploy/safety/` + `core/constraints/` | Spark's "safe set" rolls own QP; we delegate to existing `core/constraints/{convexify/cbf, operators/qp}` — battle-tested, multi-backend |
| `BaseTask` | `spark_task/base/base_task.py` | `deploy/tasks/base.py` (delegates layout to `genedynamics/tasks/*/spec.py`) | No duplication |
| `BaseRobotConfig` (DoF maps) | `spark_robot/base/base_robot_config.py` | `genedynamics/robots/<id>/spec.py` | Generated from MJCF metadata, not 4 hand-maintained enums |
| `BaseKinematics` | `spark_robot/base/base_kinematics.py` | `genedynamics/envs/robots/<id>.py` (implements `RobotModel`) | Already exists for drone/manipulator |
| Nested config + `init_member_classes` | `spark_utils/helpers.py` | `deploy/config/schema.py` + `core/registry/` | Registry-driven, not module-reflection-driven; less brittle |
| `_post_control_processing` hook | scattered | `deploy/observers/` (pluggable) | Multiple observers, decoupled |
| Render in agent | tangled in `MujocoAgent` | `deploy/observers/mujoco_renderer.py` | Renderer is an Observer, not coupled to IO |
| dict-based info flow | everywhere in spark | dataclass `RobotState`/`Intent`/`ControlCommand` | Type-safe; IDE autocomplete; no KeyError |
| Examples / launchers | `spark_pipeline/example/` | `deploy/config/presets/` + `cli.py` | Same idea, less boilerplate |
| Device selection (cuda/cpu) | per-agent flag | `cfg.runtime.device` via `core/backends` | Single source |

**Things spark has that we already beat:**

- **Multi-runtime tensor ops**: spark hard-codes torch in policies and numpy in WBC. We have `core/backends/runtime/{jax,torch,numpy}_backend.py` so a single controller code path can pick its runtime.
- **Multi-physics**: spark only has MuJoCo. We have `mujoco / mjx / brax / isaac` already in `envs/` — `RobotIO` auto-pluggable.
- **Constraint stack**: spark has a hand-rolled CBF/QP. We have `core/constraints/{terms,convexify,operators,schedulers,solvers}` — far more capable.
- **Higher-DoF**: spark caps at G1; nothing in our design assumes a max DoF (`RobotState.qpos` is variable-length, controllers query `spec.n_dof`).
- **Higher-dim tasks**: trajectory schemas are explicit (`plan_schema.py`); extending 14D → 16D → N-D is a one-file change.

**Things spark has that we should also build:**

- The clean nested-class config pattern (point 7 above)
- Sim-vs-real symmetric Agent contract (our `RobotIO`)
- A real-hardware Agent that actually works (their `G1RealAgent` is aspirational; ours must not be)

---

## 9. Migration phases

Each phase ends in a runnable state. Phase ordering minimizes time before you have a **working follower in MuJoCo**, even if WBC is still broken.

### Phase 0 — Cleanup (no behavior change)

- Delete `*_legacy.py`, empty dirs, the duplicated `sim_plan/humanoid_corridor/`
- Move `viz/` → `observers/` (rename only)
- **Exit criterion:** existing `genedynamics-deploy --mode sim` still runs

### Phase 1 — Move robot spec out of deploy

- Create `genedynamics/robots/g1/{__init__.py, spec.py, assets.py}`
- Move all G1 hardcoding (joint tuples, sites, limits, Kp/Kd, MJCF resolution) here
- Generate `RobotSpec` from MJCF metadata where possible (joint names → reflection from model)
- Register G1 in `genedynamics/robots/registry.py` on import
- Create `genedynamics/envs/robots/g1.py` implementing the existing `RobotModel` protocol
- Update `wbc_solver.py` to consume `RobotSpec` instead of `G1ModelSpec`
- **Exit criterion:** zero references to G1 inside `deploy/`; all robot facts come from `genedynamics/robots/g1/`

### Phase 2 — Define interfaces

- Create `deploy/interfaces/{messages.py, robot_io.py, controller.py, follower.py, safety.py, observers.py}`
- These are pure protocols + dataclasses; no logic
- **Exit criterion:** `from genedynamics.deploy.interfaces import RobotIO, Controller, ...` works

### Phase 3 — Extract `MujocoRobotIO` AND `MjxRobotIO`

- Create `deploy/io/base.py` (`BaseRobotIO`: shared lifecycle, control-cmd dispatch, metadata)
- Create `deploy/io/mujoco_io.py` wrapping `envs/humanoid_base_physics.py` (numpy `RobotState`)
- Create `deploy/io/mjx_io.py` wrapping `envs/humanoid_mjx.py` (jax `RobotState`) — **this is the default for new presets**
- Pipeline asserts `Controller.runtime` is compatible with the chosen IO at startup
- Refactor `HumanoidMujocoPipeline.rollout_plan` to call `io.get_state / io.send_control / io.step` — no direct `self.data`
- **Exit criterion:** existing WBC sim runs end-to-end through both `MujocoRobotIO` and `MjxRobotIO`

### Phase 4 — Extract `Controller` and split WBC

- Create `deploy/controllers/wbc/{controller.py, task_stack.py, contact_blocks.py, friction_cone.py, qp_builder.py, qp_solver_adapter.py, config.py}`
- Each file ≤ 300 lines; clear inputs/outputs; unit-testable
- WBC config split into: `task_weights`, `regularization`, `limits`, `contact` groups
- `qp_solver_adapter.py` delegates to `core/constraints/solvers/{osqp,cvxopt}`
- Add unit tests: standing balance, single-leg support, transition phase
- **Exit criterion:** old WBC tests pass; you can now meaningfully iterate on WBC

### Phase 5 — Build `SportModeController` + `MockLocoClient` (your option 1)

- `deploy/controllers/sport_mode/loco_client_base.py` — `LocoClient` protocol (`set_velocity`, `balance_stand`, `damp`, ...)
- `deploy/controllers/sport_mode/unitree_loco.py` — `SportModeController` (Controller protocol impl) consuming any `LocoClient`
- `deploy/controllers/sport_mode/mock_loco_client.py` — **template walking** for sim:
  - 3D-LIPM step planning from `(vx, vy, yaw_rate)`
  - Bezier swing trajectory in pelvis frame
  - Closed-form leg IK against `genedynamics/envs/robots/g1.py` (built in Phase 1)
  - Outputs `ControlCommand(kind="joint_pos", kp=..., kd=...)`
- Create preset `g1_corridor_mjx.py` (default) and `g1_corridor_mujoco.py` (CPU fallback)
- **Exit criterion:** **G1 walks the corridor in both MJX and MuJoCo using sport-mode + MockLocoClient**, regardless of WBC state. This is the project unblock — you now have a baseline controller, dual physics backends working, and the full sim pipeline validated.

### Phase 6 — Build `RLController` (your option 2)

- `deploy/controllers/rl/unitree_rl_gym.py` with `runtime = "torch"`
- Adapter: `Intent` → policy obs vector; policy action → `ControlCommand(kind="joint_pos")`
- **Exit criterion:** swap one config line, RL policy walks the corridor

### Phase 7 — Real `UnitreeG1RobotIO` + `RealLocoClient`

- `deploy/io/unitree_g1_io.py`: implements `RobotIO` against G1 SDK
  - `get_state`: read joint encoders + IMU + (optional) external localization plugin
  - `send_control`: dispatch on `cmd.kind`: joint_pos / loco / mixed
  - `step`: no-op
- `deploy/controllers/sport_mode/real_loco_client.py`: wraps Unitree SDK `LocoClient` — **drop-in replacement** for `MockLocoClient` from Phase 5
- Wire localization plugins (already in `deploy/localization/`)
- **Exit criterion:** Phase 5 sport-mode preset runs on real G1 by changing **two** config keys: `io.class_name` → `UnitreeG1RobotIO`, `controller.loco_client.class_name` → `RealLocoClient`. No code change.

### Phase 8 — Safety + Observers

- `deploy/safety/{joint_limit, torque_limit, self_collision, cbf_filter}.py`
- `cbf_filter.py` bridges to `core/constraints/convexify/cbf` + `operators/qp`
- Observers: logger, recorder, mujoco_renderer, web_viz, ros2_publisher (later)
- **Exit criterion:** can run sim with safety on, can record + replay episodes

### Phase 9 — Config system + registry wiring

- `deploy/config/schema.py` with nested-class base + `initialize_class` resolving via `core/registry/`
- Migrate `cli.py` to load preset modules, delete `factory.py`
- Register all `io.*`, `controller.*`, `follower.*`, `safety.*`, `task.*`, `observer.*` keys
- **Exit criterion:** all entry points go through registry; `deploy/factory.py` deleted

### Phase 10 — Docs + tests + cleanup

- Per-component README in each subdir
- Pipeline-level integration test: sim + each controller × each robot
- Delete remaining legacy paths
- **Exit criterion:** `deploy/` LOC < 50% of current; new contributors can add a controller in <100 lines

---

## 10. Acceptance checklist

This refactor is "done" when **all** of the following hold:

- [ ] `deploy/` contains zero G1-specific code; `grep -r "g1\|G1" deploy/` returns only string keys / docstrings
- [ ] Adding H1 = create `genedynamics/robots/h1/`; **zero changes** in `deploy/`
- [ ] Adding a new controller = one file under `deploy/controllers/`; one registry entry; one config-line swap
- [ ] Adding a new physics backend = one file under `deploy/io/`; ditto
- [ ] Adding a new runtime backend = `core/backends/runtime/<x>_backend.py`; controllers using `runtime` get it free
- [ ] `pipeline.py` is < 200 lines and identical for sim and real
- [ ] No file in `deploy/` exceeds 400 lines
- [ ] WBC controller is split into ≥ 4 files, each unit-testable in isolation
- [ ] WBC config has ≤ 4 grouped sub-classes, no flat 92-param dataclass
- [ ] `RobotIO`, `Controller`, `SafetyFilter`, `TrajectoryFollower`, `Observer` are all `Protocol`s in `deploy/interfaces/`
- [ ] `RobotState`, `ControlCommand`, `Intent` are dataclasses, not dicts
- [ ] No duplication of `envs/factories.py`, `core/registry/`, `core/constraints/`, `envs/robots/`, `tasks/*/spec.py`, `robots/registry.py`
- [ ] Sport-mode, RL, and WBC controllers all run the corridor task in MuJoCo from the same `pipeline.py`
- [ ] At least sport-mode controller runs on real G1 hardware
- [ ] All four execution modes (sim/real/shadow/replay) reuse the same loop
- [ ] Every spark feature in §8 is matched or superseded

---

## 11. Resolved decisions

### 11.1 Physics backends — both, default MJX (GPU)

Both `mjx` and `mujoco` (numpy) are first-class. **Default = MJX** so we get GPU + JAX autodiff for free; `mujoco` numpy is the fallback for debugging, profiling, and CPU-only environments.

Implications for `deploy/io/`:

```
deploy/io/
├── base.py                 # BaseRobotIO (shared lifecycle)
├── mujoco_io.py            # NumPy MuJoCo, wraps envs/humanoid_base_physics.py
├── mjx_io.py               # JAX MJX, wraps envs/humanoid_mjx.py        ← DEFAULT
├── brax_io.py
└── ...
```

- **Two separate classes**, not one with a flag. `RobotState.qpos` is `np.ndarray` for `MujocoRobotIO` and `jax.Array` for `MjxRobotIO`. Controllers consume `RobotState` through `runtime.to_numpy(state.qpos)` if they want host arrays.
- **`Controller.runtime` must match `RobotIO`'s array type** (MJX IO → JAX runtime; numpy IO → numpy/torch runtime). Pipeline asserts this at startup with a clear error.
- **WBC controller** runs `runtime="numpy"` (osqp on host) regardless of IO. When IO is MJX, the pipeline pulls state to host via `jax.device_get` once per step, sends torque cmd back to device. The cost is one small host↔device copy per control step — acceptable for 200 Hz.
- **RL controller** can run `runtime="jax"` end-to-end with MJX IO (zero copies, full GPU pipeline) or `runtime="torch"` with either IO.
- **Default preset** `g1_corridor_mjx.py` is created in Phase 5 alongside the existing mujoco preset; both must pass integration tests.

This raises Phase 3's exit criterion: **both `MujocoRobotIO` and `MjxRobotIO` must implement `RobotIO`** before Phase 4. They share `BaseRobotIO` for control-cmd dispatch and metadata, differ in state extraction.

### 11.2 Sport-mode in sim — yes, build `MockLocoClient`

`deploy/controllers/sport_mode/mock_loco_client.py` ships with the sport-mode controller so it works in MuJoCo / MJX too. This makes sport-mode a real sim baseline, not just a real-hw shortcut.

Design:

```
controllers/sport_mode/
├── __init__.py
├── unitree_loco.py         # SportModeController (Controller protocol impl)
│                           #   wraps a "LocoClient" interface
├── loco_client_base.py     # LocoClient protocol: SetVelocity, BalanceStand, ...
├── real_loco_client.py     # wraps Unitree SDK LocoClient (real G1 only)
└── mock_loco_client.py     # template-walking implementation for sim
```

`MockLocoClient` translates `LocoCommand(vx, vy, yaw_rate, body_height)` → joint position targets via:

1. **Inverse-pendulum step planning**: given `(vx, vy, yaw_rate)`, compute desired step length / width / period from a 3D-LIPM heuristic
2. **Bezier swing trajectory**: parameterize swing foot in pelvis frame
3. **Stance leg IK**: solve simple closed-form leg IK against the existing `genedynamics/envs/robots/g1.py` (the G1 RobotModel built in Phase 1)
4. **Output**: `ControlCommand(kind="joint_pos", joint_pos=q_target, kp=cfg.kp, kd=cfg.kd)`

This makes Phase 5 the milestone where you have **a working G1 corridor follower in MuJoCo and MJX** — independent of WBC. Real-G1 sport-mode in Phase 7 then literally just swaps `MockLocoClient` for `RealLocoClient`.

`MockLocoClient` is intentionally **not** a high-quality walking controller — it's a deterministic baseline that lets you validate the IO ↔ Follower ↔ Controller plumbing without depending on WBC convergence. Sport mode on the real robot is far better and is what you'd actually deploy with this controller.

### 11.3 Open questions still pending

- **Localization on real G1**: use existing `LocalizationPlugin` system, or pull base pose directly from G1 SDK? Defer until Phase 7.
- **WBC QP solver**: delegate to `core/constraints/solvers/osqp_solver.py` (preferred for consistency) or keep direct `osqp` for speed? Bench during Phase 4.
- **Trajectory schema extension** (14D → 16D): confirm `plan_schema.py` is single source of truth so extension is one diff.
- **`genedynamics/robots/g1/` is created fresh** in Phase 1 (currently only `robots/registry.py` + README exist).

---

## 12. Diff of responsibilities (current → target)

| Concern | Current location(s) | Target location |
|---|---|---|
| 14D corridor schema | `deploy/followers/common/plan_schema.py` | `deploy/followers/plan_schema.py` (kept) |
| 14D plan adapter | `deploy/followers/common/plan_adapter.py` | `deploy/followers/plan_adapter.py` (kept) |
| Robot-agnostic intent | `deploy/followers/common/traversal_intent.py` | `deploy/followers/traversal_intent.py` (kept; type → `Intent`) |
| Contact phase scheduler | `deploy/followers/humanoid/contact_scheduler.py` | `deploy/controllers/wbc/contact_blocks.py` (consumed by WBC; no separate scheduler dir) OR move into `corridor_follower.py` if used by other controllers |
| Footstep planner | `deploy/followers/humanoid/footstep_planner.py` | `deploy/followers/corridor_follower.py` |
| Upper-body mapper | `deploy/followers/humanoid/upper_body_mapper.py` | `deploy/followers/corridor_follower.py` |
| Task builder | `deploy/followers/humanoid/task_builder.py` | `deploy/controllers/wbc/task_stack.py` (input is `Intent`, not bespoke) |
| WBC solver (1273 lines) | `deploy/followers/humanoid/mujoco/wbc_solver.py` | `deploy/controllers/wbc/{controller, task_stack, contact_blocks, friction_cone, qp_builder, qp_solver_adapter, config}.py` |
| MuJoCo loop | `deploy/followers/humanoid/mujoco/pipeline.py` | `deploy/io/mujoco_io.py` (IO) + `deploy/pipeline.py` (loop) |
| G1 model spec | `deploy/followers/humanoid/models/g1_model.py` | `genedynamics/robots/g1/spec.py` |
| G1 kinematics | (absent) | `genedynamics/envs/robots/g1.py` |
| Real G1 SDK | `deploy/followers/humanoid/unitree/g1_backend.py` (stub) | `deploy/io/unitree_g1_io.py` (real) |
| Mixed control publisher | `deploy/followers/humanoid/unitree/mixed_control_publisher.py` (stub) | folded into `unitree_g1_io.py:send_control` |
| Loco adapter | `deploy/followers/humanoid/unitree/loco_adapter.py` (stub) | `deploy/controllers/sport_mode/unitree_loco.py` |
| Quadruped Go2 SDK | `deploy/backends/unitree_go2.py` | `deploy/io/unitree_go2_io.py` |
| Stub IO | `deploy/backends/stub.py` | `deploy/io/stub_io.py` |
| Localization adapter | `deploy/backends/localization_adapter.py` | `deploy/io/localization_io.py` (or merge into base IO) |
| Profiles | `deploy/profiles/{base, humanoid, quadruped, uav3d}.py` | `deploy/config/presets/*.py` |
| Factories | `deploy/factory.py` | DELETED — use `core/registry/` |
| Task config | `deploy/task_config.py` | per-preset under `deploy/config/presets/` |
| Modes | `deploy/modes/` | `deploy/modes/` (kept; minor cleanup) |
| Localization plugins | `deploy/localization/` | `deploy/localization/` (kept) |
| Visualization | `deploy/viz/` | `deploy/observers/` |
| `sim_plan/humanoid_corridor/` | `deploy/sim_plan/humanoid_corridor/` | DELETED — duplicate |

---

## 13. Why this beats spark

1. **First-class runtime backends**. Spark hardcodes torch in policies, numpy in WBC. Ours: every controller declares `runtime: Backend`, swappable JAX/Torch/NumPy/Rust at config time.
2. **First-class physics backends**. Spark = MuJoCo only. Ours: `MujocoRobotIO`, `MjxRobotIO`, `BraxRobotIO`, `IsaacRobotIO`, `RosRobotIO` — same `RobotIO` contract, same `pipeline.py`.
3. **Production-grade constraint stack**. Spark hand-rolls a small CBF QP. We bridge to `core/constraints/` (terms × convexify × operators × schedulers × solvers) — far more capable, tested, multi-backend.
4. **Typed messages**. Spark passes plain dicts; future bugs guaranteed. We use dataclasses with optional fields and typed unions.
5. **Single source of robot truth**. Spark maintains 4 hand-aligned IntEnums per robot. We generate from MJCF metadata into one `RobotSpec`.
6. **Multi-controller in one runtime**. Spark only ships one policy at a time; we register sport-mode + RL + WBC + PD + MPC simultaneously and select via config.
7. **No deploy/robots duplication**. Spark mixes robot models inside its agent layer; we keep robots in `genedynamics/robots/` so deploy stays orchestration-only.
8. **DoF/dimension scalability**. Spark assumes G1's 23-dof. Our `RobotState.qpos` is variable-length, controllers query `spec.n_dof`. The same WBC code handles G1 today and a 50-dof humanoid tomorrow.
9. **Higher-dim tasks**. 14D corridor today, N-D tomorrow. The trajectory schema is one file; extension is one diff.
10. **Sim ↔ real symmetry actually delivered**. Spark documents it but `G1RealAgent` is incomplete. We commit to making `UnitreeG1RobotIO` actually run on hardware as a Phase 7 exit criterion.

---

## 14. Third-party policy integration guide

This section is the cookbook for plugging a pre-trained RL policy (Unitree RL Gym, IsaacLab, LeggedGym, custom) into the deploy pipeline. The contract is small enough that integrating a new policy = writing one adapter file under `deploy/controllers/rl/<policy_name>.py` and adding one preset.

### 14.1 The contract

A third-party policy is integrated by implementing four pieces:

| Piece | Type | Purpose |
|---|---|---|
| **`PolicyArtifact`** | dataclass | Where the weights / training cfg live, plus how to load them |
| **`ObsBuilder`** | protocol impl | `RobotState + Intent → obs_vector` (training-format obs) |
| **`ActionMapper`** | protocol impl | `policy_output → ControlCommand` (joint targets, kp/kd) |
| **`RLController`** subclass | `Controller` protocol impl | Owns the loaded model + history buffer + decimation |

The `Controller` protocol from §5.3 is the only contract the rest of the pipeline sees. Sport-mode, WBC and RL all conform to it identically — switching policies is one config change.

### 14.2 Reference interfaces

```python
# deploy/controllers/rl/base.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np

from genedynamics.deploy.interfaces.controller import Controller
from genedynamics.deploy.interfaces.messages import RobotState, ControlCommand, Intent
from genedynamics.robots.base import RobotSpec


@dataclass
class PolicyArtifact:
    """Where a third-party policy lives on disk and how to load it."""
    ckpt_path: Path                # .pt / .onnx / .pkl / .jax-flax
    train_cfg_path: Path           # YAML/JSON: default_pose, action_scale, obs_scales, kp, kd, dof_names
    framework: str                 # "torch" | "onnx" | "jax" | "tflite"


class ObsBuilder(Protocol):
    """Builds the obs vector this policy was trained on.

    Implementations capture the EXACT obs layout from the original repo:
    field order, units, frame, normalization, history stacking.
    """
    obs_dim: int

    def reset(self) -> None: ...
    def build(self, state: RobotState, intent: Intent) -> np.ndarray: ...


class ActionMapper(Protocol):
    """Maps raw policy output to a ControlCommand for the robot.

    Most sim2real policies output `delta_q` to be added to a default pose
    and tracked by joint PD with fixed Kp/Kd. ActionMapper handles:
      - permutation from policy DoF order → robot spec DoF order
      - default-pose offset
      - action scaling
      - clipping
    """
    action_dim: int

    def map(self, action: np.ndarray) -> ControlCommand: ...
```

### 14.3 Reference `RLController`

```python
# deploy/controllers/rl/base.py (continued)

class RLController(Controller):
    """
    Generic RL controller base. Subclass to bind a specific (artifact, obs_builder, action_mapper).

    Handles:
      - decimation: policy may run at 50 Hz while control loop runs at 200 Hz
      - obs history rolling buffer
      - device transfer (host ↔ accelerator) once per policy step
      - last-action latching between decimated policy steps
    """

    def __init__(
        self,
        spec: RobotSpec,
        artifact: PolicyArtifact,
        obs_builder: ObsBuilder,
        action_mapper: ActionMapper,
        *,
        policy_hz: float,
        control_hz: float,
        runtime: str = "torch",
    ) -> None:
        self.spec = spec
        self.runtime = runtime
        self.artifact = artifact
        self.obs_builder = obs_builder
        self.action_mapper = action_mapper
        self._policy = self._load_policy(artifact)
        self._decimation = max(1, int(round(control_hz / policy_hz)))
        self._tick = 0
        self._last_cmd: ControlCommand | None = None

    def reset(self, io) -> None:
        self.obs_builder.reset()
        self._tick = 0
        self._last_cmd = None

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        if self._tick % self._decimation == 0 or self._last_cmd is None:
            obs = self.obs_builder.build(state, intent)
            action = self._infer(obs)
            self._last_cmd = self.action_mapper.map(action)
        self._tick += 1
        return self._last_cmd

    def _load_policy(self, artifact: PolicyArtifact):
        if artifact.framework == "torch":
            import torch
            return torch.jit.load(str(artifact.ckpt_path)).eval()
        if artifact.framework == "onnx":
            import onnxruntime as ort
            return ort.InferenceSession(str(artifact.ckpt_path))
        if artifact.framework == "jax":
            import pickle
            with open(artifact.ckpt_path, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"Unknown framework: {artifact.framework}")

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        if self.artifact.framework == "torch":
            import torch
            with torch.no_grad():
                return self._policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
        if self.artifact.framework == "onnx":
            return self._policy.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        if self.artifact.framework == "jax":
            import jax
            return np.asarray(jax.jit(self._policy)(obs))
        raise ValueError
```

### 14.4 Concrete adapter: `UnitreeRLGymG1Controller`

```python
# deploy/controllers/rl/unitree_rl_gym.py

import numpy as np
import yaml
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from genedynamics.deploy.controllers.rl.base import (
    ActionMapper, ObsBuilder, PolicyArtifact, RLController,
)
from genedynamics.deploy.interfaces.messages import ControlCommand, Intent, RobotState
from genedynamics.robots.g1 import G1RobotSpec


@dataclass
class UnitreeRLGymObsCfg:
    """Mirrors unitree_rl_gym/legged_gym/envs/g1/g1_config.py."""
    history_len: int                       # frame stack
    lin_vel_scale: float
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    default_dof_pos: np.ndarray            # (n_actuated,)
    policy_dof_names: tuple[str, ...]      # order in which the policy expects DoFs


class UnitreeRLGymObsBuilder(ObsBuilder):
    """Build obs in the exact format Unitree RL Gym trained on.

    Layout (one frame, in policy DoF order):
      [base_lin_vel*scale (3),
       base_ang_vel*scale (3),
       projected_gravity (3),
       cmd_vx, cmd_vy, cmd_yaw_rate,
       (dof_pos - default)*scale (n_actuated),
       dof_vel*scale (n_actuated),
       last_action (n_actuated)]
    Then concatenated `history_len` times.
    """
    def __init__(self, spec: G1RobotSpec, cfg: UnitreeRLGymObsCfg, perm_policy_to_robot: np.ndarray):
        self.spec = spec
        self.cfg = cfg
        self.perm = perm_policy_to_robot         # policy_idx[i] = robot_idx
        self.inv_perm = np.argsort(perm_policy_to_robot)
        self.n = spec.num_actuated
        self.single_dim = 3 + 3 + 3 + 3 + 3 * self.n
        self.obs_dim = self.single_dim * cfg.history_len
        self._history: deque[np.ndarray] = deque(maxlen=cfg.history_len)
        self._last_action = np.zeros(self.n, dtype=np.float32)

    def reset(self) -> None:
        self._history.clear()
        self._last_action[:] = 0.0

    def build(self, state: RobotState, intent: Intent) -> np.ndarray:
        # 1. extract actuated dof in robot order
        q_robot = state.qpos[self.spec.actuated_qpos_indices]
        qd_robot = state.qvel[self.spec.actuated_dof_indices]
        # 2. permute to policy order
        q_pol = q_robot[self.inv_perm]
        qd_pol = qd_robot[self.inv_perm]
        # 3. base velocities & projected gravity
        base_lin_vel = state.base_twist[:3] if state.base_twist is not None else np.zeros(3)
        base_ang_vel = state.base_twist[3:] if state.base_twist is not None else np.zeros(3)
        proj_g = _project_gravity(state.base_pose[3:7]) if state.base_pose is not None else np.array([0., 0., -1.])
        # 4. assemble single-frame obs
        cfg = self.cfg
        single = np.concatenate([
            base_lin_vel * cfg.lin_vel_scale,
            base_ang_vel * cfg.ang_vel_scale,
            proj_g,
            np.array([intent.base_lin_vel[0], intent.base_lin_vel[1], intent.base_yaw_rate], dtype=np.float32),
            (q_pol - cfg.default_dof_pos) * cfg.dof_pos_scale,
            qd_pol * cfg.dof_vel_scale,
            self._last_action,
        ]).astype(np.float32)
        # 5. history stack
        if not self._history:
            for _ in range(cfg.history_len):
                self._history.append(single)
        else:
            self._history.append(single)
        return np.concatenate(self._history, axis=0)


class UnitreeRLGymActionMapper(ActionMapper):
    """delta_q → joint_pos_target, permuted back to robot order, with PD gains."""
    def __init__(self, spec: G1RobotSpec, cfg: UnitreeRLGymObsCfg,
                 perm_policy_to_robot: np.ndarray, kp: np.ndarray, kd: np.ndarray):
        self.spec = spec
        self.cfg = cfg
        self.perm = perm_policy_to_robot
        self.kp = kp.astype(np.float32)
        self.kd = kd.astype(np.float32)
        self.action_dim = spec.num_actuated

    def map(self, action: np.ndarray) -> ControlCommand:
        # action is in policy DoF order
        q_target_pol = self.cfg.default_dof_pos + action * self.cfg.action_scale
        q_target_robot = q_target_pol[self.perm]                # → robot order
        return ControlCommand(
            kind="joint_pos",
            joint_pos=q_target_robot.astype(np.float32),
            kp=self.kp,
            kd=self.kd,
        )


class UnitreeRLGymG1Controller(RLController):
    """Adapter for policies trained with unitree_rl_gym (legged_gym fork)."""

    def __init__(self, spec: G1RobotSpec, ckpt_path: str | Path, train_cfg_path: str | Path, *,
                 policy_hz: float = 50.0, control_hz: float = 200.0):
        artifact = PolicyArtifact(
            ckpt_path=Path(ckpt_path),
            train_cfg_path=Path(train_cfg_path),
            framework="torch",
        )
        cfg, perm, kp, kd = self._parse_train_cfg(train_cfg_path, spec)
        obs_builder = UnitreeRLGymObsBuilder(spec, cfg, perm)
        action_mapper = UnitreeRLGymActionMapper(spec, cfg, perm, kp, kd)
        super().__init__(
            spec=spec,
            artifact=artifact,
            obs_builder=obs_builder,
            action_mapper=action_mapper,
            policy_hz=policy_hz,
            control_hz=control_hz,
            runtime="torch",
        )
        # last_action feedback wiring: ActionMapper → ObsBuilder
        self._wire_last_action_feedback()

    def _wire_last_action_feedback(self) -> None:
        original_map = self.action_mapper.map
        def wrapped(action: np.ndarray) -> ControlCommand:
            self.obs_builder._last_action = action.astype(np.float32)
            return original_map(action)
        self.action_mapper.map = wrapped  # type: ignore[method-assign]

    @staticmethod
    def _parse_train_cfg(path: Path, spec: G1RobotSpec):
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Concrete keys depend on the upstream repo; this is a template.
        policy_dof_names = tuple(raw["dof_names"])
        cfg = UnitreeRLGymObsCfg(
            history_len=int(raw.get("history_len", 1)),
            lin_vel_scale=float(raw["obs_scales"]["lin_vel"]),
            ang_vel_scale=float(raw["obs_scales"]["ang_vel"]),
            dof_pos_scale=float(raw["obs_scales"]["dof_pos"]),
            dof_vel_scale=float(raw["obs_scales"]["dof_vel"]),
            action_scale=float(raw["control"]["action_scale"]),
            default_dof_pos=np.asarray(raw["init_state"]["default_joint_angles"], dtype=np.float32),
            policy_dof_names=policy_dof_names,
        )
        # Build permutation: position i in policy order ↔ position perm[i] in robot order.
        robot_names = spec.actuated_joints
        perm = np.array([robot_names.index(n) for n in policy_dof_names], dtype=np.int64)
        kp = np.asarray(raw["control"]["stiffness"], dtype=np.float32)
        kd = np.asarray(raw["control"]["damping"], dtype=np.float32)
        return cfg, perm, kp, kd
```

### 14.5 Wiring it via config

```python
# deploy/config/presets/g1_corridor_rlgym.py
from genedynamics.deploy.config.presets.g1_corridor_mjx import G1CorridorMjxConfig

class G1CorridorRLGymConfig(G1CorridorMjxConfig):
    class controller:
        registry_key = "controller.rl.unitree_rl_gym_g1"
        ckpt_path = "third_party/unitree_rl_gym/logs/g1/exported/policies/policy_1.pt"
        train_cfg_path = "third_party/unitree_rl_gym/logs/g1/exported/policies/cfg.yaml"
        policy_hz = 50.0
```

That's the entire integration. `pipeline.py` doesn't change, `RobotIO` doesn't change, `Follower` doesn't change.

### 14.6 Integration checklist

When you receive a new third-party policy, run through this list **in order**:

- [ ] **Locate the obs definition** in upstream code. Read `compute_observations()` (or equivalent) line-by-line. Write down: field order, scaling, frame (world / base / pelvis), history length, normalization stats. Mismatch here is the #1 sim2real failure.
- [ ] **Locate the action definition**. Is it `delta_q` (residual) or absolute `q_target`? What's the action scale? Where does the default pose come from?
- [ ] **Locate the DoF order**. The policy's DoF order is almost never the same as your `RobotSpec.actuated_joints`. Build the permutation and unit-test it both ways.
- [ ] **Locate Kp/Kd**. They're in the training cfg, not in the spec — use the *training* gains, not the robot's default.
- [ ] **Locate decimation**. `control_hz` (PD loop) vs `policy_hz` (network). Set both in the controller, latch the action between policy steps.
- [ ] **Domain randomization residue**. Did training apply obs noise / action delay? Check the training cfg. Either leave it off (deploy is the "lucky env") or replicate explicitly.
- [ ] **Last-action feedback**. Most policies feed the previous action back into the obs. Wire `ObsBuilder._last_action ← ActionMapper.map()`.
- [ ] **Initial condition**. What pose does the policy expect at reset? Set `RobotIO.reset()` to spawn there.
- [ ] **Sanity test in MJX preset first**. If it walks in MJX with `MockLocoClient`-like behavior, the wiring is sane. Then deploy on real.
- [ ] **Latency test**. Measure `act()` p99 latency. Torch + GPU + small policy = ~1 ms. If above 5 ms, you can't run at 200 Hz control.
- [ ] **Determinism**. Set `torch.use_deterministic_algorithms(True)` for replay/debug.
- [ ] **Save the adapter + cfg + ckpt together**. Treat the trio as one artifact; version it.

### 14.7 Common pitfalls (read this before debugging "policy walks in sim but not on robot")

| Pitfall | Symptom | Fix |
|---|---|---|
| Wrong DoF permutation | Robot kicks one leg out, falls instantly | Unit-test `perm` and `inv_perm` round-trip |
| Wrong obs frame | Diverges slowly, oscillates | Recompute base velocities in base frame, not world |
| Wrong action scale | Robot tries to fly / barely moves | Read `cfg.control.action_scale` precisely; usually 0.25–0.5 |
| Wrong default pose | Robot drifts to weird posture | Use the training default, not the MJCF keyframe |
| Missing last_action in obs | Jittery, unstable | Wire feedback as in §14.4 |
| Wrong gravity convention | Falls forward or backward | Project gravity into base frame as `R_base_world.T @ [0,0,-1]` |
| Wrong Kp/Kd | Stiff/floppy | Use training gains, not robot spec gains |
| Double scaling | Tiny actions | `action * action_scale` is applied once, not at both ends |
| Sim ≠ deploy obs noise | Works in sim, not real | Identify training noise model, replicate in `ObsBuilder` |
| Latency | Falls right at policy step boundary | Reduce decimation, or quantize policy to ONNX |

### 14.8 Multi-framework support

`PolicyArtifact.framework` decouples the adapter from the inference framework. Adding ONNX / JAX / TFLite is one branch in `RLController._load_policy()` and `_infer()`. The runtime backend (`Controller.runtime`) follows the framework, so the rest of the pipeline can plan device transfers correctly:

| Framework | `runtime` | Best paired IO |
|---|---|---|
| torch | "torch" | MujocoRobotIO (CPU) or MjxRobotIO (one host↔device copy / step) |
| jax (flax / haiku) | "jax" | MjxRobotIO (zero-copy, full GPU pipeline) |
| onnx | "numpy" | MujocoRobotIO |
| tflite | "numpy" | UnitreeG1RobotIO (real, low-latency edge) |
