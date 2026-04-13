# Deploy

Modular execution layer for running robot controllers in simulation and on
real hardware. One config line swaps the physics backend, the controller, or
the safety stack — everything else stays the same.

```
Preset (nested-class config)
    |
    v
build_components()
    |-- IO            (sim / real hardware adapter)
    |-- Controller    (WBC / sport-mode / RL policy)
    |-- Follower      (trajectory intent generator)
    |-- Safety        (joint limit / torque limit / CBF / composite)
    |-- Task          (episode termination / metrics)
    |-- Observers     (logger / recorder / ROS2 publisher / web viz)
    v
runner.run_preset()   (episode loop)
```

## Quick Start

### Python API

```python
from genedynamics.deploy.runner import run_preset
from genedynamics.deploy.presets import G1CorridorMujocoSportModePreset

# Run a single episode in MuJoCo simulation
run_preset(G1CorridorMujocoSportModePreset, episode_id="ep_0001", max_steps=5000)
```

### CLI

```bash
python -m genedynamics.deploy.runner \
    --preset genedynamics.deploy.presets:G1CorridorMujocoSportModePreset \
    --max-steps 5000 --episode-id ep_0001
```

### Legacy YAML CLI

```bash
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml
genedynamics-deploy --robot humanoid --model g1 --mode sim --episodes 2
```

---

## Components

### IO Backends

Adapters that abstract away whether the robot is simulated or real hardware.
All implement `RobotIO` protocol: `reset()`, `get_state()`, `send_control()`,
`step()`, `close()`.

| Registry Key | Class | Runtime | Backend | Description |
|-------------|-------|---------|---------|-------------|
| `io.mujoco` | `MujocoRobotIO` | numpy | mujoco | CPU MuJoCo; exposes mass matrix, jacobians, foot contacts |
| `io.mjx` | `MjxRobotIO` | jax | mjx | JAX-compiled MuJoCo XLA on GPU; optional host shadow sync |
| `io.brax` | `BraxRobotIO` | jax | brax | Brax GPU environment; contact gradients for RL training |
| `io.isaac_lab` | `IsaacLabRobotIO` | torch | isaac_lab | Isaac Lab torch-native GPU; zero-overhead with torch controllers |
| `io.unitree_g1` | `UnitreeG1RobotIO` | numpy | (real HW) | Real Unitree G1 via SDK; supports localization plugins |
| `io.ros2` | `ROS2RobotIO` | numpy | (real HW) | Generic ROS2 hardware adapter |
| `io.stub` | `StubRobotIO` | numpy | stub | Zero-physics test stub; records all sent commands |

**Brax / Isaac Lab are optional** -- `ImportError` is caught silently at
registration so CPU-only dev machines work fine.

### Controllers

Map `(RobotState, Intent) -> ControlCommand`. Each declares a `runtime` tag
and a `produces` tuple of command kinds.

| Registry Key | Class | Runtime | Description |
|-------------|-------|---------|-------------|
| `controller.wbc` | `HumanoidWBCController` | numpy | Inverse-dynamics QP (osqp); full-body humanoid control |
| `controller.sport_mode` | `SportModeController` | numpy | Mixed: legs from LocoClient, upper body PD |
| `controller.rl_unitree_rl_gym` | `UnitreeRLGymG1Controller` | torch | Adapter for unitree_rl_gym policies |
| `controller.rl_passthrough` | `PassthroughRLController` | numpy | Minimal callable-based RL controller for testing |
| `controller.quadruped_stepping` | `QuadrupedSteppingController` | numpy | Batch wrapper for quadruped gait control |

**RL controller framework** (`controllers/rl/base.py`): adding a new policy
means writing one `ObsBuilder` + one `ActionMapper` (~100 LOC), not a full
controller. Decimation, last-action feedback, and telemetry are handled
automatically.

### Safety Filters

Project unsafe commands onto the safe set. Stacked via `CompositeSafetyFilter`.

| Registry Key | Class | Description |
|-------------|-------|-------------|
| `safety.joint_limit` | `JointLimitFilter` | Clip joint positions to spec ranges (configurable margin) |
| `safety.torque_limit` | `TorqueLimitFilter` | Clip torque magnitude to per-joint bounds |
| `safety.self_collision` | `SelfCollisionFilter` | Capsule-distance check; retreats command toward current qpos |
| `safety.workspace` | `WorkspaceFilter` | Cartesian bounding-box for arm teleoperation |
| `safety.singularity` | `SingularityFilter` | Attenuate arm commands near kinematic singularities |
| `safety.cbf` | `CBFFilter` | Control-Barrier-Function QP projection |
| `safety.composite` | `CompositeSafetyFilter` | Chain multiple filters in declaration order |

### Followers

Generate per-step `Intent` from trajectory plans or user input.

| Registry Key | Class | Description |
|-------------|-------|-------------|
| `follower.humanoid_corridor` | Humanoid follower stack | Contact scheduling, footstep planning, WBC task building |
| `follower.teleop` | `TeleopFollower` | Bridge gamepad / keyboard / ROS2 input to intent |

### Tasks

Episode termination logic and metrics. Return `StepInfo(done=True)` to end
an episode.

| Registry Key | Class | Description |
|-------------|-------|-------------|
| `task.noop` | `BaseExecutionTask` | Never terminates, no metrics |
| `task.corridor_follow` | `CorridorFollowTask` | Track progress toward goal; done when within tolerance |
| `task.teleop` | `TeleopTask` | Detect fall (height/angle), E-stop, or max_steps |

### Observers

Side-effect hooks called at episode start, every step, and episode end.

| Registry Key | Class | Description |
|-------------|-------|-------------|
| `observer.logger` | `LoggerObserver` | Per-step JSON-Lines + summary.json to disk |
| `observer.recorder` | `RecorderObserver` | In-memory buffer flushed to .npz per episode |
| `observer.ros2_publisher` | `ROS2PublisherObserver` | Publish JointState + control commands to ROS2 |

Additional (not registered): `WebVizService` for HTML/GIF rendering,
`render_episode_to_gif()` for MuJoCo offline rendering.

### Teleoperation

| Module | Class | Description |
|--------|-------|-------------|
| `teleop/gamepad_source.py` | `GamepadSource` | Pygame gamepad (Xbox/DualShock) |
| `teleop/keyboard_source.py` | `KeyboardSource` | WASD + QE + arrows + Space/Esc |
| `teleop/ros2_source.py` | `ROS2TeleopSource` | ROS2 Twist subscriber |
| `teleop/teleop_follower.py` | `TeleopFollower` | Bridges any `InputSource` to `Intent` |

### Localization Plugins

Provide world-frame base pose for real-hardware IO.

| Module | Class | Description |
|--------|-------|-------------|
| `localization/mock_plugin.py` | `MockLocalizationPlugin` | Fixed pose for testing |
| `localization/ros2_odometry_plugin.py` | `ROS2OdometryPlugin` | ROS2 nav_msgs/Odometry |
| `localization/vicon_shm_plugin.py` | `ViconShmPlugin` | VICON shared memory |

### Runtime Utilities

| Module | Class | Description |
|--------|-------|-------------|
| `runtime/clock.py` | `Clock`, `MonotonicClock`, `FakeClock` | Pluggable clocks |
| `runtime/rate_limiter.py` | `RateLimiter` | Fixed-period rate limiter with jitter tracking |
| `runtime/decimation.py` | `Decimation` | Keep one event per N calls |

---

## Presets

Ready-made configurations that wire all components together. Override one
inner class to customize a single layer.

| Preset | IO | Controller | Use Case |
|--------|----|------------|----------|
| `G1CorridorMujocoSportModePreset` | MuJoCo CPU | Sport-mode (SparkRL legs) | Sim baseline |
| `G1CorridorMujocoWBCPreset` | MuJoCo CPU | WBC (inverse-dynamics QP) | Full-body control |
| `G1CorridorBraxPreset` | Brax GPU (jax) | Sport-mode (auto-bridged) | GPU RL training |
| `G1CorridorIsaacLabPreset` | Isaac Lab GPU (torch) | RL (torch-native) | Torch RL deploy |
| `G1CorridorRealSportModePreset` | Unitree G1 HW | Sport-mode (real SDK) | Real robot |
| `G1TeleopMujocoPreset` | MuJoCo CPU | Sport-mode + gamepad | Sim teleoperation |
| `G1TeleopRealPreset` | Unitree G1 HW | Sport-mode + gamepad | Real teleoperation |
| `Go2SteppingStonesMujocoPreset` | MuJoCo CPU | Quadruped stepping | Quadruped gait |

---

## Usage Examples

### 1. Run a sim episode with the default G1 corridor preset

```python
from genedynamics.deploy.runner import run_preset
from genedynamics.deploy.presets import G1CorridorMujocoSportModePreset

built = run_preset(G1CorridorMujocoSportModePreset, max_steps=3000)
print(f"Episode finished in {built.io.step_count} steps")
```

### 2. Switch from MuJoCo CPU to MJX GPU — one line change

```python
from genedynamics.deploy.presets import G1CorridorMujocoWBCPreset
from genedynamics.deploy.config_schema import ComponentConfig

class MyGPUPreset(G1CorridorMujocoWBCPreset):
    class io(ComponentConfig):
        registry_key = "io.mjx"       # <- only this line changes
        sim_dt = 1.0 / 500.0
        keyframe_name = "stand"

run_preset(MyGPUPreset)
# -> RuntimeBridge: wrapping MjxRobotIO (jax) for HumanoidWBCController (numpy)
```

### 3. Use Brax for gradient-based RL training

```python
from genedynamics.deploy.presets import G1CorridorBraxPreset
run_preset(G1CorridorBraxPreset)
```

### 4. Use Isaac Lab with a torch-native RL controller

```python
from genedynamics.deploy.presets import G1CorridorIsaacLabPreset
run_preset(G1CorridorIsaacLabPreset)
# -> torch IO + torch controller: zero cross-runtime conversion
```

### 5. Build components without running (for inspection / testing)

```python
from genedynamics.deploy.config_schema import build_components
from genedynamics.deploy.presets import G1CorridorMujocoSportModePreset

built = build_components(G1CorridorMujocoSportModePreset, strict=True)
print(built.io)           # MujocoRobotIO
print(built.controller)   # SportModeController
print(built.safety)       # CompositeSafetyFilter
```

### 6. Write a custom preset with different safety filters

```python
from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig

class MyPreset(DeployConfig):
    control_hz = 50.0
    sim_dt = 1.0 / 500.0
    max_steps = 10_000

    class io(ComponentConfig):
        registry_key = "io.mujoco"
        sim_dt = 1.0 / 500.0

    class controller(ComponentConfig):
        registry_key = "controller.wbc"

    class safety(ComponentConfig):
        registry_key = "safety.composite"
        filters = [
            {"registry_key": "safety.joint_limit", "margin": 0.05},
            {"registry_key": "safety.torque_limit", "safety_margin": 0.9},
            {"registry_key": "safety.self_collision"},
        ]

    observers = [
        {"registry_key": "observer.logger", "out_dir": "results/my_experiment"},
    ]
```

### 7. Teleoperation (sim)

```python
from genedynamics.deploy.presets import G1TeleopMujocoPreset
run_preset(G1TeleopMujocoPreset)
# -> Opens gamepad/keyboard input; sends velocity commands to the G1
```

### 8. Register a custom controller and use it in a preset

```python
from genedynamics.deploy.registries import controller_registry

class MyController:
    runtime = "numpy"
    produces = ("joint_pos",)
    def reset(self, io=None): ...
    def act(self, state, intent):
        return ControlCommand(kind="joint_pos", joint_pos=np.zeros(29))

controller_registry.register("my_ctl", MyController)

class MyPreset(DeployConfig):
    class io(ComponentConfig):
        registry_key = "io.stub"
    class controller(ComponentConfig):
        registry_key = "controller.my_ctl"
```

---

## Runtime Compatibility

The pipeline validates at build time that IO, controller, and safety filter
share compatible array runtimes (`numpy` / `jax` / `torch`).

**Auto-bridge**: if the IO runtime differs from the controller runtime, an
`ArrayBridge` wraps the IO and converts arrays transparently. Example:
`MjxRobotIO (jax)` + `HumanoidWBCController (numpy)` -> bridge inserted
automatically.

**Command kind validation**: `controller.produces` must be a subset of
`io.accepts`, checked at build time. Mismatches raise `CommandKindError`.

Config flags:
- `strict_runtime = True` -> fail on any runtime mismatch instead of bridging
- `strict_build = True` -> fail if `io` or `controller` sections are missing

---

## Directory Structure

```
deploy/
  config_schema.py        Config parsing + build_components()
  registries.py           Component registries (io, controller, safety, ...)
  runner.py               Episode loop entry point
  array_bridge.py         Cross-runtime array conversion (numpy/jax/torch)
  runtime_check.py        Runtime + command-kind validation
  interfaces/             Protocol definitions (RobotIO, Controller, ...)
  io/                     IO backends (mujoco, mjx, brax, isaac_lab, ...)
  controllers/            Controller implementations (wbc, sport_mode, rl)
  safety/                 Safety filters (joint_limit, torque, cbf, ...)
  followers/              Intent generators (humanoid corridor, teleop)
  tasks/                  Episode termination (corridor_follow, teleop)
  observers/              Data recording (logger, recorder, ros2, web_viz)
  teleop/                 Input sources (gamepad, keyboard, ros2)
  localization/           Base pose plugins (ros2 odom, vicon, mock)
  runtime/                Clocks, rate limiters, decimation
  presets/                Ready-made configurations
```

## Optional Dependencies

| Feature | Package | Install |
|---------|---------|---------|
| MJX GPU sim | `mujoco-mjx`, `jax` | `pip install mujoco-mjx jax[cuda]` |
| Brax GPU sim | `brax >= 0.10` | `pip install brax` |
| Isaac Lab GPU sim | Isaac Lab | `docker build -f docker/isaac-lab.Dockerfile` |
| Real G1 hardware | Unitree SDK | Unitree SDK install guide |
| ROS2 integration | `rclpy` | ROS2 workspace |
| Teleoperation | `pygame`, `pynput` | `pip install pygame pynput` |
