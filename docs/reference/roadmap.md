# 🗺️ Roadmap

Checked items are available in V1. Unchecked items are planned work; current
robot and device support is listed in [compatibility](compatibility.md).

## Available

- [x] Policy learning, learned diffusion and reusable priors
- [x] JAX planning and composable constraints
- [x] Six task families and planner baselines
- [x] Configured experiments, metrics, traces and replay

## Next

- [ ] MGA CUDA qualification and humanoid walking/push
- [ ] Torch planner parity and device-resident array transfers
- [ ] Rust/C++ control and I/O components
- [ ] Panda/xArm7 deployment controllers and `ros2_control` actions
- [ ] Async planning/control, command expiry and fault recovery
- [ ] Reproducible hardware latency, reliability and task-quality benchmarks

## Explore

- [ ] MuJoCo Warp rollout integration
- [ ] Action-chunk/VLA proposals through OpenPI and LeRobot adapters
- [ ] Live perception, timestamped geometry and dynamic scenes

[Backend acceptance gates](compatibility.md#backend-acceptance-contract) ·
[Benchmark contract](metrics.md)

<details>
<summary>Integration references</summary>

- [ROS2 joint trajectory controller](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
- [LeRobot](https://huggingface.co/docs/lerobot/index)

</details>
