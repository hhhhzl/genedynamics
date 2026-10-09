# Roadmap

The next steps extend the planning and execution interfaces already in the
framework. These are development priorities, not a support matrix or delivery
dates. Current support is recorded in [compatibility](compatibility.md).

## Development priorities

| Priority | Direction | What it enables | Release gate |
| --- | --- | --- | --- |
| Next | **GPU qualification and portable compute** | Qualify MGA on CUDA, extend planner support to Torch, and evaluate MuJoCo Warp for large rollout batches. Scope Rust/C++ first to controller and I/O components. | Matched task-quality checks; cold compile, warm p50/p95/p99, memory and transfer measurements on each advertised device. Backend parity before a backend is called supported. |
| Next | **More robot controllers and hardware adapters** | Reuse robot profiles with joint-trajectory and Cartesian impedance controllers; add reviewed arm deployment recipes for Panda and xArm7 alongside the G1 path. Integrate `ros2_control` execution actions. | Joint/frame/calibration checks, tracking error, force/torque limits, cancellation and disconnect recovery on each named robot. A simulation profile alone does not establish hardware support. |
| Next | **Predictable closed-loop execution** | Separate slower planning from faster control, expire stale plans and commands, and make stop/fallback behavior explicit across adapters. | Planner overruns, stale sensors, connection loss and controller exceptions produce recorded, bounded responses; publish deadline-miss and jitter distributions in soak runs. |
| Later | **Learned-policy and VLA adapters** | Use external action chunks as proposals or warm starts for constrained planning. Explore OpenPI clients and LeRobot recording/replay without coupling the core package to model weights. | Explicit observation/action/frame/time contracts; replayable examples comparing raw policy proposals and executed commands, with intervention and task-quality metrics. |
| Later | **Live scene and perception inputs** | Update obstacle geometry and object poses from timestamped sensor or ROS inputs; extend existing scene-frame localization to calibrated manipulation scenes. | Frame and timestamp checks, geometry-version consistency during planning, and measured behavior under delayed observations and moving obstacles. |
| Ongoing | **Reusable benchmarks and integrations** | Make new planners, robot recipes and simulator adapters easy to compare and contribute. Publish complete runs with task-quality and execution metrics. | Versioned configs and result schemas, installable optional plugins, replayable artifacts and hardware manifests; see the [benchmark contract](metrics.md). |

MGA GPU qualification and humanoid walking/push remain follow-up work. Broader
robot coverage should follow working controller/driver recipes, rather than
the number of robot meshes that can be loaded.

## What already exists

- **Planning and compute:** a shared solver interface, JAX planner kernels,
  runtime backend classes, and dedicated Torch planner integrations. The
  [array bridge](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/deploy/array_bridge.py) currently converts
  JAX/Torch arrays through host NumPy; general Torch planner parity and DLPack
  transfers remain future work.
- **Robot composition:** [robot profiles](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/robots/profile.py)
  bind semantic joints, frames and capabilities to simulator models, including
  Panda and xArm7. [RobotIO](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/deploy/interfaces/robot_io.py) and
  [Controller](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/deploy/interfaces/controller.py) keep execution
  components separate. MuJoCo, MJX, Brax, Isaac Lab, ROS2 topic and Unitree G1
  adapters are present; their presence does not imply equal qualification.
- **Execution tooling:** controller/follower/safety/observer protocols,
  WBC/RL/SDK controllers, composable safety filters, timing/decimation helpers,
  recording and scene-frame localization. The generic ROS2 adapter publishes
  joint commands and twists; it is not a `FollowJointTrajectory` action client.
- **Policy proposals:** [shared prior protocols](https://github.com/hhhhzl/genedynamics/blob/release/v1-open-source/genedynamics/learning/priors/base.py)
  and JAX RL/diffusion priors already expose warm-start and horizon-proposal
  seams. No OpenPI or LeRobot integration is advertised in this release.

## Design references

These external projects inform the proposed integration boundaries; they are
not dependencies or claims of interoperability in V1.

- [ROS2 joint trajectory controller](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
  provides monitored action execution, tolerances and cancellation.
  [Controller Manager](https://control.ros.org/jazzy/doc/ros2_control/controller_manager/doc/userdoc.html)
  supplies lifecycle and hardware-interface management.
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) targets NVIDIA
  batch throughput and interoperates with JAX/Torch. Its throughput focus must
  be evaluated separately from single-plan latency.
- [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
  separates GPU policy serving from robot-side clients and returns action
  chunks. [LeRobot](https://huggingface.co/docs/lerobot/index) connects recording,
  datasets, training, replay and deployment.

The choices and release gates above are engineering proposals based on the
current interfaces and these references.
