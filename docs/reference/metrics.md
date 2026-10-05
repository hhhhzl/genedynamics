# Metrics and benchmark contract

Industrial readiness needs measurements that users can reproduce. A large
benchmark number without its compilation, hardware, or failure policy is not a
release claim.

## Package and compatibility evidence

- clean wheel build, metadata validation, install, and import;
- wheel size and file inventory;
- supported Python/OS/device matrix;
- optional dependency import and missing-dependency behavior;
- configuration catalogue validation and public plugin inventory.

## Planner performance

Report both cold and steady-state behavior:

| Metric | Required definition |
| --- | --- |
| Compile time | First trace/compile wall time, isolated from execution |
| Cold latency | End-to-end first plan including compile and transfers |
| Warm latency | p50, p95, p99 after declared warm-up count |
| Throughput | Plans or candidate trajectories per second at stated batch/horizon |
| Peak memory | Host RSS and device high-water mark |
| Scaling | Latency/memory versus horizon, state/action dimension, samples, and modes |
| Jitter | Standard deviation and tail ratio across repeated warm runs |

Every row records CPU/GPU model, memory, OS, Python, JAX/jaxlib, simulator,
commit, config hash, dtype, horizon, sample count, diffusion steps, seed count,
synchronization points, and timing method.

## Task quality and safety

- task success rate with confidence intervals;
- safe success rate, with incomplete/aborted execution counted explicitly;
- maximum and integrated constraint violation;
- collision, contact-loss, force/torque, balance, and recovery events;
- cost/return distribution and terminal error;
- plan-to-execution tracking error;
- robustness across seeds, named suites, model mismatch, and sensing noise.

Task metrics must state the evaluation window and whether they use planned or
executed trajectories.

## Operational reliability

- completed, failed, aborted, skipped, and resumed matrix entries;
- time to detect invalid configuration or missing checkpoint;
- resume correctness after interruption;
- artifact completeness and schema validation;
- deterministic replay rate within declared numerical tolerances;
- long-run error rate and resource leakage over repeated invocations.

## Publication rule

Do not place a performance number in the README until the repository contains:

1. a versioned benchmark configuration;
2. the command and environment manifest;
3. raw per-run measurements;
4. an aggregation script;
5. the commit that produced the result.

Until then, describe the capability and the gate, not an unverified speedup.
