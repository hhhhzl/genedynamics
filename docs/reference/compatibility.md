# Compatibility matrix

This matrix distinguishes implemented code from a supported V1 product path.
Automation reads the matching machine-readable contract from
`release/v1_compatibility.json`.

## Compute backends

| Backend | General solver coverage | Device coverage | V1 status |
| --- | --- | --- | --- |
| JAX | Published core planners and constraints | CPU supported; CUDA install path available | Supported solver backend |
| NumPy | Types, adapters, utilities, selected geometry/constraint paths | CPU | Supporting runtime, not full solver parity |
| Torch | DPCC, SafeDiffuser, runtime tensor adapter | CPU/CUDA depends on Torch install | Optional specialized integration |
| Rust | None | None | Planned |
| C++ | Native simulator/library interop only | Depends on integration | Planned as a solver backend |

The V1 dependency contract is JAX/JAXlib `>=0.6.2,<0.7`. The release CPU
suite was validated with JAX and JAXlib 0.6.2, NumPy 2.2.6, and SciPy 1.15.3.
Changes to that numerical stack require the same configuration, unit, and
integration gates as a code change.

MGA has a CPU release path. GPU qualification is deferred until numerical
parity, compilation, memory, and reliability gates are measured.

## Platforms and simulators

| Capability | Linux | macOS | V1 status |
| --- | --- | --- | --- |
| Core JAX CPU | CI target | Supported development path | Supported |
| MuJoCo/MJX | Simulator CI target | Supported with platform wheels | Supported with `simulation` extra |
| Brax | Simulator CI target | Development path | Optional |
| D3IL | Source/submodule setup | Source/submodule setup | Optional integration |
| Isaac Lab | NVIDIA container required | Not available | Experimental deployment adapter |
| JAX CUDA | NVIDIA driver/runtime required | Not available | Installable; task qualification varies |

## Backend acceptance contract

A future backend becomes supported only after it passes:

1. dtype, shape, broadcasting, and array-conversion conformance;
2. seeded randomness and reproducibility tests;
3. autodiff/Jacobian behavior required by each solver;
4. JIT or ahead-of-time compilation lifecycle and cache reporting;
5. CPU/device transfer and synchronization semantics;
6. numerical parity on reference tasks with declared tolerances;
7. serialization of configs, checkpoints, arrays, and failure state;
8. latency, throughput, peak memory, and repeated-run reliability gates.

Rust and C++ should implement a narrow backend protocol first. Rewriting the
experiment runner or result model in each language would destroy the stable
application boundary.
