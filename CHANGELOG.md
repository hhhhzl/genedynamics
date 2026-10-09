# Changelog

This project follows semantic versioning after the first public tag.

## Unreleased

- Adds native MD-COAS multi-candidate trajectory animations for planar navigation and 7-DoF arm avoidance, paired with their diffusion views in the homepage gallery.

- Restores the complete task-to-execution architecture workflow, with learning, constraints, simulation, deployment and evaluation connected explicitly.
- Removes the obsolete 2GO corridor and stepping-stone poster images.

- Aligns the homepage with generative dynamics on manifold and gives Planning, Control and Learning equal prominence.
- Explains batched constraints inside generative sampling, including learned diffusion integrations.
- Repairs the unfetchable Menagerie submodule pin and preserves the local Go2 MJX adaptation as an idempotent patch.
- Installs the simulation CI dependencies needed for TorchScript execution and JAX kinematics.

- Positions the framework around generative models for robot learning, planning and control.
- Documents policy training, learned trajectory diffusion and MGA checkpoint reuse.

- Adds environment and constraint-solver catalogues with implementation links.
- Simplifies the roadmap into a checklist and adds a coordinated visual palette.

- Adds a direct environment + 2GO Python example and a paper-linked planner catalogue.
- Rebuilds the homepage around a text-free, five-group native demonstration gallery.
- Adds controller, compute, learned-policy and perception roadmap priorities.

- Select and add the public open-source license.
- Complete release-candidate validation and publish measured benchmark artifacts.

## 0.1.0 release candidate

- Defines a JAX-first plugin architecture for robotics planning and control.
- Publishes planar navigation, D3IL avoiding, quadruped stepping stones,
  humanoid corridor, peg insertion, and surface scanning configurations.
- Packages MBD, EB-MBD, MDOC, MD-COAS, 2GO, MPPI, MGA, contact-control
  baselines, and optional learned planner integrations.
- Adds standard wheel packaging, dependency extras, CI quality gates, user
  documentation, selected paper media, configuration validation, and release
  governance.
- Excludes soft-robot, co-design, and 3D Gaussian Splatting research systems.
