# FADA solver

`FADASolver` is the stable frontend for the source-derived Planner-IDM policy.
The Torch numerical implementation and model primitives live under `backends/`.
The solver consumes public history tensors and returns policy actions; it does
not construct task observations or own experiment lifecycle.

Four-task history, command, body-chart, and action-decoder bindings live in
`genedynamics.experiments.plugins.methods.transfer.fada`. Training, datasets,
normalization, LoRA adaptation, and checkpoint production live in
`genedynamics.solvers.single.fada.learning`.

The port is derived from `LeCAR-Lab/FADA` at commit
`290b3a9c5c0ebe2e9851b61350617544be05e9bc`. See `LICENSE`, `NOTICE`,
`source_manifest.json`, and `requirements/baselines/sources.lock.json`.
