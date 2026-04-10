# Phase 16–18 Plan — Runtime & Backend Expansion

**Status:** proposed
**Owner:** deploy maintainer
**Depends on:** Phases 9–15 (registry runner, teleop + ROS2 stack) — all complete
**Last updated:** 2026-04-09

---

## Overview

Phases 16–18 close the "runtime & backend" gaps identified in the Phase 15
audit. They are **strictly additive** — no existing preset, script, or
controller is modified; every change is either a new file or an extension
point in `runner.py` / `config_schema.py`.

| Phase | Title                         | Goal                                                     | Blocking?       |
|-------|-------------------------------|----------------------------------------------------------|-----------------|
| 16    | Runtime compatibility & bridge | Validate IO↔controller runtime pairing and auto-convert  | Prerequisite    |
| 17    | BraxRobotIO                   | GPU sim backend for contact-rich training                 | Optional (needs 16) |
| 18    | IsaacLabRobotIO               | Torch-runtime IO for Isaac Lab                            | Optional (needs 16) |

Phase 16 is a **prerequisite** for 17 and 18 because both introduce new
(physics_backend, array_runtime) combinations and would silently crash at
`controller.act()` today.

---

## Motivation — why now, why these three

### Current pain points

1. **Silent runtime mismatch** — nothing in `runner.run_preset()` or
   `config_schema.build_components()` asserts that
   `io.array_runtime == controller.runtime`.  A user who wires
   `MjxRobotIO` (`jax`) with `HumanoidWBCController` (`numpy`) gets a
   `TypeError` somewhere deep in `osqp` after 100+ lines of debugging.

2. **Scattered conversion code** — `MjxRobotIO` calls `jax.device_get()`
   inline in `_read_state()` and `jnp.asarray()` inline in
   `_apply_command()`.  There is no central place to add
   `torch.from_numpy` / `tensor.detach().cpu().numpy()` when Phase 18
   lands.

3. **Backends that the docstrings promise but that don't exist** —
   `interfaces/robot_io.py` lists Brax and Isaac in the class docstring;
   neither file is present.  New contributors hit dead ends.

### Why not wait

* Teleoperation (Phase 12) pushed the deploy loop closer to real-time
  hardware use, and real robot fleets often mix runtimes (torch RL policy
  + numpy safety filter).  The mismatch bug is one teleop session away.
* Brax / Isaac Lab are needed for faster RL training than MJX offers on
  some GPU configurations.
* All three phases are small (< 600 LOC total) and self-contained.

### Out of scope

* Rewriting WBC in JAX or Torch (huge; unrelated to backend support).
* Replacing `osqp` with a differentiable QP solver.
* Adding a "rust" runtime tag (listed in interfaces for symmetry but no
  implementation planned).
* Any changes to `/spark`, which is still a data-only dependency (only
  `assets/g1_motion.pt` was extracted — Phase 11 outcome).

---

## Phase 16 — Runtime compatibility & array bridge

### Objective

Guarantee that any (IO, controller, safety) combination either

1. shares a compatible `array_runtime` tag, **or**
2. is wrapped in an automatic conversion bridge at startup.

Failures must surface during `build_components()` with a human-readable
message, not during the first `act()` call.

### Architecture

```
    ┌─────────────────────┐
    │  config_schema.py   │
    │  build_components   │
    └──────────┬──────────┘
               │ after building io + controller + safety
               ▼
    ┌─────────────────────┐     mismatch     ┌──────────────────────┐
    │  validate_runtimes  │─────────────────▶│  insert ArrayBridge  │
    │     (new fn)        │                  │     (new class)      │
    └──────────┬──────────┘                  └──────────────────────┘
               │ compatible or bridged
               ▼
         runner.run_preset(...)
```

### New files

#### `deploy/array_bridge.py`   (~180 LOC)

Single module providing:

```python
# Pseudo-signature
def convert(arr: Any, target_runtime: str) -> Any:
    """numpy ↔ jax ↔ torch ↔ rust[stub], lazy-imported per target."""

class ArrayBridge:
    """Wraps an IO so its outputs/inputs are transparently converted.

    Composition:
        bridge = ArrayBridge(io, target_runtime="torch")
        bridge.get_state()       # → RobotState with torch tensors
        bridge.send_control(cmd) # torch tensors → io.array_runtime

    The bridge owns ~0 state: it re-reads the wrapped io every call.
    """
```

Lazy-imports `jax`, `torch` only when actually needed so CPU-only installs
stay lean.  Falls back to `numpy` with a clear error if the target runtime
is unavailable.

Conversion rules (the full table lives in the module docstring):

| source → target | implementation                      |
|-----------------|--------------------------------------|
| numpy → jax     | `jnp.asarray(x)`                     |
| jax → numpy     | `jax.device_get(x)` → `np.asarray`   |
| numpy → torch   | `torch.from_numpy(x.copy())`         |
| torch → numpy   | `x.detach().cpu().numpy()`           |
| jax ↔ torch     | go via numpy (no dlpack optimization in v1) |
| anything → same | identity (no copy)                   |

Note: `RobotState` and `ControlCommand` carry multiple optional arrays, so
`ArrayBridge._wrap_state(state)` walks every `Optional[np.ndarray]` field
and converts each.  Keep the walk explicit — no `dataclasses.fields()`
magic, because shape/semantics matter (e.g. `contact` is a dict of arrays,
not a single array).

#### `deploy/runtime_check.py`   (~80 LOC)

```python
def validate_runtimes(built: BuiltComponents) -> BuiltComponents:
    """Raise RuntimeMismatchError or insert bridges.

    Policy (in order):
      1. If io.array_runtime == controller.runtime == safety.runtime, return
         built unchanged.
      2. If all components share a single runtime after pair-wise collapse,
         allow it (e.g. a passthrough safety that declares "any").
      3. If controller.runtime is a strict subset (torch needed but io is
         numpy), wrap io with ArrayBridge and log the wrap at build time.
      4. Otherwise raise RuntimeMismatchError with the full triple.

    The "any" runtime is a new sentinel for safety filters that work on
    pure numpy-API ControlCommand without caring about the IO's backend.
    """
```

`RuntimeMismatchError` is a new exception in `deploy/interfaces/__init__.py`
(or `interfaces/errors.py` if we want to keep interface modules type-only).

### Modified files

#### `deploy/config_schema.py`

* `build_components()` gains a trailing call
  `return validate_runtimes(built)` before returning.
* No other change — everything is additive.

#### `deploy/runner.py`

* `run_preset()` already calls `build_from_preset()` → automatically
  picks up the validation.  No runner loop changes.

#### `deploy/interfaces/safety.py`

* Add `"any"` to the runtime-tag docstring.  `SafetyFilter.runtime` already
  exists as a `str`, so no schema change.

#### `deploy/safety/base.py`

* Default `BaseSafetyFilter.runtime = "numpy"` stays; specific filters can
  override to `"any"` if they only touch `ControlCommand` metadata.

### Tests

Add `tests/deploy/test_runtime_bridge.py` (~120 LOC):

1. Numpy-only round-trip: `numpy → numpy` returns unchanged (identity).
2. `convert(np.zeros(3), "jax")` returns a `jax.Array` of shape `(3,)`.
3. `convert(torch.zeros(3), "numpy")` returns a `np.ndarray`.
4. `ArrayBridge(MjxRobotIO, "numpy")` produces a `RobotState` whose
   `qpos` is numpy.
5. `validate_runtimes` on a matching preset returns unchanged.
6. `validate_runtimes` on a `MjxRobotIO + HumanoidWBCController` preset
   inserts a bridge and emits one `logging.info` line.
7. `validate_runtimes` on an incompatible triple raises
   `RuntimeMismatchError`.

Tests mock `jax` and `torch` via the Python 3.10+ Docker image (existing
CI pattern from Phase 10).

### Acceptance criteria

* `validate_runtimes()` rejects every known bad combination with a
  message that names the three runtime tags.
* Existing presets (G1 corridor mujoco / wbc / sport-mode, Go2 stepping,
  G1 teleop) all pass `validate_runtimes()` unchanged.
* `ArrayBridge` round-trip is lossless for `RobotState`
  (qpos/qvel/joint_torque/base_pose/base_twist) and `ControlCommand`
  (joint_pos/joint_vel/joint_torque/kp/kd).
* Docstring of `runner.run_preset` documents the new behaviour.
* Zero changes to any controller or existing IO implementation.

### Open questions

* **Q16.1:** Should bridges be *logged and allowed* or *explicit opt-in*?
  → Proposal: logged and allowed.  Users can set
  `DeployConfig.strict_runtime = True` (default `False`) to fail-closed.
* **Q16.2:** Does `ControlCommand.extras` (a dict) need walking?
  → Proposal: no.  Extras are controller-specific blobs; if a controller
  stuffs a torch tensor there it is the controller's problem.
* **Q16.3:** dlpack bridges for `jax ↔ torch`?
  → Deferred to Phase 19.  Going via numpy is fast enough at 200 Hz for
  state vectors under ~200 floats.

---

## Phase 17 — BraxRobotIO

### Objective

Add `deploy/io/brax_io.py` exposing a Brax MJX-compatible simulation as a
`RobotIO` with `physics_backend="brax"`, `array_runtime="jax"`.

### Why Brax, not more MJX

* Brax's `positional` and `generalized` backends produce contact gradients
  that MJX does not (yet) support, which is useful for gradient-based RL.
* Some RL papers (CrossQ, PPO-Brax variants) ship Brax-tuned envs directly.
* Same runtime as MJX (`jax`), so Phase 16's bridge covers all pairings
  for free.

### Scope

* Single Brax env: **G1** (humanoid).  Go2 / quadruped comes later if
  needed — not blocking.
* Single backend config: **`mjx`** (Brax's MJX backend, for parity with
  `MjxRobotIO`).  `positional` / `generalized` are one-line follow-ups.
* Reset / step semantics identical to `MjxRobotIO` so presets can swap
  `io.registry_key = "io.brax"` and nothing else.

### Architecture

```
┌──────────────────────────────┐
│      brax.envs.Env (g1)      │
│  brax.System / brax.State    │
└──────────────┬───────────────┘
               │ jax arrays in/out
               ▼
┌──────────────────────────────┐
│   BraxRobotIO (RobotIO)      │
│   physics_backend = "brax"   │
│   array_runtime   = "jax"    │
│   accepts = ("joint_pos",    │
│              "torque")       │
└──────────────┬───────────────┘
               │ validate_runtimes → host controllers bridged
               ▼
            runner
```

### New files

#### `deploy/io/brax_io.py`   (~350 LOC)

Class layout:

```python
class BraxRobotIO(BaseRobotIO):
    physics_backend = "brax"
    array_runtime   = "jax"
    accepts         = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        *,
        env_name: str = "g1",
        backend: str = "mjx",          # "mjx" | "positional" | "generalized"
        seed: int = 0,
        sim_dt: float = 1.0 / 500.0,
        spec: Optional[G1RobotSpec] = None,
        sync_host_shadow: bool = True, # parity with MjxRobotIO
    ) -> None: ...

    def reset(self)           -> RobotState: ...
    def get_state(self)       -> RobotState: ...
    def send_control(self, cmd: ControlCommand) -> None: ...
    def step(self, dt: float) -> RobotState: ...
    def close(self)           -> None: ...
```

Implementation notes:

* Wraps `brax.envs.create(env_name, backend=backend)` and stores the
  returned `env` + current `brax.State`.
* `reset()` calls `env.reset(jax.random.PRNGKey(seed))` and extracts
  `state.pipeline_state.q`, `state.pipeline_state.qd` into a `RobotState`.
* `send_control()` latches an action array; `step(dt)` calls
  `env.step(state, action)` and returns the new RobotState.
* `sync_host_shadow=True` mirrors `MjxRobotIO` — maintains a numpy
  shadow of qpos/qvel via `jax.device_get()` so numpy controllers paired
  through Phase 16's bridge pay only one transfer per step.
* Contact handling: Brax does not expose per-geom contact wrenches the
  same way MuJoCo does.  For the humanoid corridor use case we accumulate
  foot contacts via `state.pipeline_state.contact.link_idx` + penetration
  depth; exported as a minimal `FootContactSnapshot`-compatible dict.
  **Known limitation:** contact richness is lower than `MujocoRobotIO`.
  Document in the class docstring; do not pretend parity.

#### `deploy/presets/g1_corridor_brax.py`   (~60 LOC)

Identical structure to `g1_corridor_mujoco_sport_mode.py` except:

```python
class io(ComponentConfig):
    registry_key = "io.brax"
    env_name     = "g1"
    backend      = "mjx"
    sim_dt       = 1.0 / 500.0
```

Everything else — controller, safety, task, observers — unchanged.

### Modified files

#### `deploy/registries.py`

Insert next to the `mjx` io registration:

```python
try:
    from genedynamics.deploy.io.brax_io import BraxRobotIO
    io_registry.register("brax", BraxRobotIO)
except ImportError:
    pass  # brax is an optional dependency
```

#### `deploy/presets/__init__.py`

Add `G1CorridorBraxPreset` to the public re-exports.

#### `deploy/io/__init__.py` (if one exists)

Add `BraxRobotIO` to `__all__` symmetrically with `MjxRobotIO`.

### Dependencies

* `brax >= 0.10` (pip-installable, pulls `jax`).
* No new deploy-side dep.  Brax is optional: `deploy/registries.py`
  swallows `ImportError` so CPU-only dev laptops still import cleanly.

### Tests

Add `tests/deploy/test_brax_io.py` (~150 LOC, marked `@pytest.mark.brax`
for CI opt-in):

1. `BraxRobotIO().reset()` returns a `RobotState` with non-empty
   `qpos` / `qvel`.
2. `get_state() == reset()` immediately after reset (no hidden step).
3. `send_control(joint_pos=zeros) + step(dt)` advances qpos.
4. `physics_backend == "brax"` and `array_runtime == "jax"`.
5. Integration test: `G1CorridorBraxPreset` runs for 100 steps end-to-end
   without exception (WBC bridged via Phase 16).

### Acceptance criteria

* Switching `io.registry_key` from `"io.mjx"` to `"io.brax"` in a corridor
  preset runs the same WBC / safety / observer stack unmodified.
* Brax is fully optional — absent import does not break any existing
  import path.
* `BraxRobotIO` docstring clearly lists the contact-observation
  limitations compared to `MujocoRobotIO`.

### Open questions

* **Q17.1:** Use Brax's built-in `g1` env, or ship a custom env with our
  XML?  → Proposal: built-in first, fork only if joint layout disagrees.
* **Q17.2:** Should we port `HumanoidFootContactSnapshot` into the Brax
  world, or leave followers that depend on rich contacts blocked?
  → Proposal: ship the minimal snapshot; document that
  `HumanoidContactScheduler`'s touchdown latching may behave slightly
  differently on Brax.

---

## Phase 18 — IsaacLabRobotIO

### Objective

Add `deploy/io/isaac_lab_io.py` — a `RobotIO` wrapping Isaac Lab's G1
environment with `physics_backend="isaac_lab"`, `array_runtime="torch"`.
This is the deploy layer's first **torch-runtime IO**, unlocking native
pairing with `UnitreeRLGymG1Controller` and any future torch-side WBC.

### Why Isaac Lab

* Official NVIDIA stack, large robot zoo, better GPU utilisation than
  MJX on newer CUDA cards in our tests.
* Already used outside the deploy layer by some of our RL training
  scripts — having an IO means we can **run the same controller code**
  at deploy time that was used in training.
* Torch runtime gives a clean testing target for Phase 16's bridge.

### Scope

* Humanoid G1 only.
* Single-env (num_envs = 1) for deploy.  Batched multi-env rollouts are
  a training concern, out of scope.
* No differentiable physics — Isaac Lab is used as a "fast numpy-ish"
  backend; gradients stay inside the policy.

### Architecture

```
┌───────────────────────────────┐
│  omni.isaac.lab.envs.G1Env    │
│  returns torch tensors on GPU │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  IsaacLabRobotIO (RobotIO)    │
│  physics_backend="isaac_lab"  │
│  array_runtime  ="torch"      │
│  accepts = ("joint_pos",      │
│             "torque", "mixed")│
└──────────────┬────────────────┘
               │ validate_runtimes
               ▼
   numpy controllers bridged,
   torch controllers pass-through
```

### New files

#### `deploy/io/isaac_lab_io.py`   (~400 LOC)

```python
class IsaacLabRobotIO(BaseRobotIO):
    physics_backend = "isaac_lab"
    array_runtime   = "torch"
    accepts         = ("joint_pos", "torque", "mixed")

    def __init__(
        self,
        *,
        task_name: str = "Isaac-G1-Stand-v0",  # or custom corridor task
        device: str = "cuda:0",
        sim_dt: float = 1.0 / 200.0,
        headless: bool = True,
        spec: Optional[G1RobotSpec] = None,
    ) -> None: ...
```

Design choices:

* **Lazy import** everything under `omni.isaac.lab.*` inside `__init__`
  so module import costs nothing on a machine without Isaac Lab.
* Construct one `ManagerBasedRLEnv` with `num_envs=1`.  Keep a reference
  to the underlying `env.unwrapped` for direct tensor access.
* `reset()` calls `env.reset()` and unpacks `obs["policy"]` into a
  `RobotState` with torch tensors.  Joint ordering comes from
  `env.unwrapped.scene.articulations["robot"].data.joint_names` and
  must match `G1RobotSpec.actuated_joints`.  If ordering disagrees, build
  a permutation index at IO-init time and apply it every step.
* `send_control()` expects torch tensors; if paired with a numpy
  controller, Phase 16's bridge handles the conversion.  No inline
  conversion code here.
* `step(dt)` calls `env.step(action)`; Isaac Lab is rate-driven by its
  own task config, so `dt` is used only for `state.t` bookkeeping.
* `close()` calls `env.close()` and, on final episode, `simulation_app.close()`.

Known quirks to handle:

* Isaac Lab's `ManagerBasedRLEnv.step` returns `(obs, reward, terminated,
  truncated, extras)`.  We ignore reward / termination (tasks own that).
* Observation tensors live on GPU by default; `sync_host_shadow` option
  forces a numpy mirror analogous to `MjxRobotIO`.

#### `deploy/presets/g1_corridor_isaac_lab.py`   (~60 LOC)

```python
class runtime(ComponentConfig):
    name = "torch"

class io(ComponentConfig):
    registry_key = "io.isaac_lab"
    task_name    = "Isaac-G1-Corridor-v0"
    device       = "cuda:0"

class controller(ComponentConfig):
    registry_key = "controller.rl_unitree_rl_gym"
    # Native torch runtime — no bridge needed.
```

Safety and observers are unchanged from the mujoco preset (they're
`numpy`, bridged automatically via Phase 16).

### Modified files

#### `deploy/registries.py`

```python
try:
    from genedynamics.deploy.io.isaac_lab_io import IsaacLabRobotIO
    io_registry.register("isaac_lab", IsaacLabRobotIO)
except ImportError:
    pass  # Isaac Lab is optional
```

#### `deploy/presets/__init__.py`

Add `G1CorridorIsaacLabPreset`.

#### `docker/` layer

New Dockerfile stage `docker/isaac-lab.Dockerfile` inheriting from
NVIDIA's official `isaac-lab` image and installing `genedynamics` on top.
(Isaac Lab is too heavy to include in the default dev image.)

### Dependencies

* Isaac Lab (via NVIDIA's image) — optional.
* Torch ≥ 2.1 — already required by `UnitreeRLGymG1Controller`, so no
  new dep.

### Tests

* Unit tests marked `@pytest.mark.isaac_lab`; CI runs them only on the
  `isaac-lab` Docker image, not on the default dev image.
* Round-trip: `IsaacLabRobotIO.reset()` → torch tensors; after Phase 16
  bridge, `HumanoidWBCController.act()` returns a numpy command; bridge
  converts back to torch for `send_control`.
* Smoke preset: `G1CorridorIsaacLabPreset` runs 50 steps without crashing.

### Acceptance criteria

* Deploy layer imports cleanly on a machine without Isaac Lab
  (no-op try/except in registries).
* `UnitreeRLGymG1Controller` runs end-to-end on `IsaacLabRobotIO` with
  **zero** cross-runtime conversions (both declare `torch`).
* `HumanoidWBCController` runs on `IsaacLabRobotIO` via the Phase 16
  bridge, with a measured per-step conversion overhead logged in the
  smoke test.

### Open questions

* **Q18.1:** Should we require Isaac Lab's in-repo task name, or build
  a custom `Isaac-G1-Corridor-v0` task that mirrors our corridor spec?
  → Proposal: start with a stock stand task for Phase 18; build the
  corridor task as a Phase 18.1 follow-up.
* **Q18.2:** Do we need to respect Isaac Lab's `episode_length` termination?
  → Proposal: no; `TeleopTask` / `CorridorFollowTask` own termination,
  so we pass `env.step(action)` the current action every call and ignore
  the env's own done signal.

---

## Cross-phase dependencies

```
Phase 16 ──┬──▶ Phase 17 (Brax)
           └──▶ Phase 18 (Isaac Lab)
```

* Phases 17 and 18 are independent of each other.
* Both need Phase 16 because they introduce (jax, torch) backends that
  would silently break WBC / sport-mode / teleop presets without the
  bridge.

## Risks & mitigations

| Risk                                                    | Mitigation                                                     |
|---------------------------------------------------------|----------------------------------------------------------------|
| Phase 16 bridge adds latency > 1 ms at 200 Hz loop       | Benchmark during tests.  If hit, raise an error suggesting the user switch to a matching-runtime controller. |
| Brax G1 joint order disagrees with `G1RobotSpec`         | Build a permutation table at IO init; fail loudly if mismatch. |
| Isaac Lab install matrix conflicts with core torch       | Use a dedicated Docker stage; do not touch the main `pyproject.toml`. |
| "`import brax`" triggers unwanted GPU init at import-time | Gate behind `try/except ImportError`; unit-tested on CPU-only host. |

## Effort estimate (not time estimate)

| Phase | New LOC | Modified LOC | New tests | New dep (optional) |
|-------|---------|--------------|-----------|-----------------------|
| 16    | ~260    | ~30          | ~120      | none                   |
| 17    | ~410    | ~15          | ~150      | `brax`                 |
| 18    | ~460    | ~20          | ~130      | `isaac-lab` (docker)   |
| **Total** | **~1,130** | **~65**  | **~400**  | 2 optional             |

## Rollout order

1. **Phase 16 first.**  Merge behind a `strict_runtime=False` default so
   existing scripts keep working; enable strict mode in CI only.
2. **Phase 17 second** (Brax).  Lower risk than Isaac because no new
   runtime — it's jax, which already works via the bridge from 16.
3. **Phase 18 last** (Isaac Lab).  First use of the torch path through
   the bridge; surfaces any bugs before they affect users.
4. After all three merge, a final **Phase 19** can enable
   `strict_runtime=True` by default once every preset is runtime-clean.

---

## Appendix A — Example user-facing flow after all three phases

Today:

```python
# Fails silently at controller.act()
preset = G1CorridorMujocoWBCPreset()
preset.io.registry_key = "io.mjx"    # oops, jax
run_preset(preset)                   # crashes 30 seconds in
```

After Phase 16:

```python
# Validates at build time, inserts bridge, logs once
preset = G1CorridorMujocoWBCPreset()
preset.io.registry_key = "io.mjx"
run_preset(preset)
# → [build] RuntimeBridge: wrapping MjxRobotIO (jax) for HumanoidWBCController (numpy)
# → ep_0001 begins...
```

After Phase 17:

```python
preset = G1CorridorBraxPreset()      # new preset
run_preset(preset)                   # WBC via bridge, contact-rich tasks flagged
```

After Phase 18:

```python
# torch-native, no bridge, zero conversion overhead
preset = G1CorridorIsaacLabPreset()
preset.controller.registry_key = "controller.rl_unitree_rl_gym"
run_preset(preset)
```

## Appendix B — Files added / modified (full list)

**Phase 16**
- ADD `deploy/array_bridge.py`
- ADD `deploy/runtime_check.py`
- ADD `tests/deploy/test_runtime_bridge.py`
- MOD `deploy/config_schema.py` (+1 call)
- MOD `deploy/interfaces/safety.py` (docstring only)
- MOD `deploy/interfaces/__init__.py` or new `interfaces/errors.py` (new exception)

**Phase 17**
- ADD `deploy/io/brax_io.py`
- ADD `deploy/presets/g1_corridor_brax.py`
- ADD `tests/deploy/test_brax_io.py`
- MOD `deploy/registries.py` (+try/except block)
- MOD `deploy/presets/__init__.py` (+1 export)

**Phase 18**
- ADD `deploy/io/isaac_lab_io.py`
- ADD `deploy/presets/g1_corridor_isaac_lab.py`
- ADD `tests/deploy/test_isaac_lab_io.py`
- ADD `docker/isaac-lab.Dockerfile`
- MOD `deploy/registries.py` (+try/except block)
- MOD `deploy/presets/__init__.py` (+1 export)

No deletions.  No changes to controllers, followers, tasks, observers,
safety filters, or the teleop stack.
