# GenerativeDynamics

**A modular engine for robot planning and control.**

![Robot planning and control across six tasks](assets/showcase.gif)

Compose trajectory generation, geometric constraints, simulation, and
closed-loop execution in one robotics stack. Start with a task recipe, choose a
planner, and connect the components your application needs.

[Watch the full-quality video](assets/showcase.mp4) ·
[Explore the task recipes](recipes/index.md)

## Start here

1. [Install the package](getting-started/installation.md).
2. [Run and inspect a planar experiment](getting-started/quickstart.md).
3. Learn the [architecture and ownership boundaries](concepts/architecture.md).
4. Select a published [robotics recipe](recipes/index.md).
5. Check the [compatibility matrix](reference/compatibility.md) before choosing
   a simulator or device.

## The platform

- **Planning:** diffusion, sampling MPC, governed optimization, and
  contact-aware control through reusable task interfaces.
- **Geometry and constraints:** obstacles, signed-distance geometry, collision
  constraints, safety filters, and recovery components.
- **Simulation and execution:** environment adapters, controllers, governors,
  robot I/O, and observers.
- **Computation and tooling:** JAX-native planning, configuration-driven runs,
  resumable task matrices, metrics, and visualization plugins.

![Platform architecture](assets/architecture.svg)

## Release status

V1 is JAX-first and includes six task families. Dedicated Torch integrations
support DPCC and SafeDiffuser; broader Torch support and Rust/C++ backends are
planned. MGA GPU qualification and humanoid walking/push are scheduled for a
later release. See the [compatibility matrix](reference/compatibility.md) and
[V1 scope](releases/v1_scope.md) for the current release boundaries.
