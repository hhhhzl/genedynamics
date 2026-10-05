# V1 open-source scope

This document is the source of truth for the first public GenerativeDynamics
release. It prevents experimental research branches from becoming accidental
public APIs.

## Product promise

GenerativeDynamics is a unified planning and control stack for robotics. V1
lets users run interchangeable planners against shared environments,
constraints, metrics, visualizations, and experiment protocols from versioned
configuration files.

## Included capabilities

- Config-driven experiment execution with plugin registries for methods,
  environments, metrics, obstacle generators, and visualizations.
- JAX-first accelerated planning and control on supported CPU and GPU devices.
- Planning methods: MBD, EB-MBD, MDOC, CFS-MBD, 2GO, MPPI, DIAL, MGA,
  ATACOM, ISSA, PegasusFlow, standalone RL, and model-based-only comparisons.
- Specialized integrations: DPCC and SafeDiffuser when their optional Torch
  dependencies are installed.
- Reproducible runs with explicit seeds, resolved configurations, metrics, and
  saved artifacts.
- Robot task recipes for planar navigation, D3IL avoiding, quadruped stepping
  stones, humanoid corridor navigation, peg insertion, and surface scanning.

MDCOAS, MDCOAS-F, and MDCOAS-A are published as configuration variants of the
full CFS-MBD implementation. `2go` and `twogo` name the same method family.

## Deferred from V1

- Soft-robot simulation and morphology co-design.
- 3D Gaussian Splatting and active scene reconstruction.
- MGA GPU readiness claims.
- Humanoid walking and push-to-line task readiness claims.
- Torch, Rust, and C++ backends. V1 defines the boundary they must implement;
  it does not claim that those backends exist.

Soft-robot, co-design, and 3DGS work must not be imported, registered,
documented as available, or installed through the default V1 package. MGA GPU
and humanoid push remain visible only as deferred development work until their
release gates pass.

## Release task matrix

| Task | Configuration root | V1 status |
| --- | --- | --- |
| Planar navigation | `configs/single_2d` | Included |
| D3IL avoiding | `configs/d3il_avoiding` | Included |
| Quadruped stepping stones | `configs/quadruped/stepping_stones_2d` | Included |
| Humanoid corridor | `configs/humanoid/corridor_2d` | Included |
| Manipulator peg insertion | `configs/arm/peg_insert` | Included |
| Manipulator surface scan | `configs/arm/surface_scan` | Included |
| Humanoid push-to-line | `configs/humanoid/push_to_line` | Deferred |

## Public release gates

V1 is ready to tag only when all of the following are true:

1. A clean environment can install the core package and at least one supported
   simulator extra.
2. Every included recipe has one documented command and a deterministic smoke
   test.
3. CI covers package build, installation, import, unit tests, configuration
   validation, and the CPU smoke path.
4. The README states measured capabilities and limitations without promising
   deferred work.
5. License, citation, contribution, security, and code-of-conduct files are
   complete.
6. Published benchmark numbers include hardware, software versions, warm-up,
   compilation treatment, batch size, horizon, seeds, and raw result artifacts.
