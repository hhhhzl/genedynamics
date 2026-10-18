# Human2Sim2Robot solver

`Human2Sim2RobotSolver` is a thin frontend over the checkpoint-compatible Torch
actor in `backends/`. The backend owns recurrent state, native normalization,
action scaling, and strict actor checkpoint restore.

The object-trajectory task binding lives in
`genedynamics.experiments.plugins.methods.transfer.human2sim2robot`. Dataset,
task-environment, and training code live in `genedynamics.solvers.single.human2sim2robot.learning`.
Simulator assets and task acceptance remain environment/task responsibilities.

The port follows `tylerlum/human2sim2robot` at commit
`894eae2ec3ae39a573b81bd1860d14cc6bdfa6df`. See `LICENSE.upstream`,
`source_manifest.json`, and `requirements/baselines/sources.lock.json`.
