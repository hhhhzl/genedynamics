# Third-party notices

GenerativeDynamics' original code is licensed under the root [MIT license](LICENSE).
Bundled third-party code, robot descriptions, meshes and pretrained policies
retain the licenses and copyright notices below. The project license does not
replace these licenses or the licenses of separately installed dependencies.

## Bundled source

| Component and local paths | Upstream | License text retained in this repository |
| --- | --- | --- |
| D3IL simulation framework: `third_party/environments/d3il/` | [ALRhub/d3il](https://github.com/ALRhub/d3il), copyright © 2024 Autonomous Learning Robots Lab @ KIT | [MIT](third_party/environments/d3il/LICENSE) |
| OpenAI Gym rotation helpers referenced by D3IL's `d3il_sim/utils/geometric_transformation.py` | [openai/gym](https://github.com/openai/gym), copyright © 2016 OpenAI | [Upstream license and model notice](third_party/environments/d3il/LICENSE.gym) |
| DPCC diffusion and trajectory-projection implementation: `third_party/diffuser/` | [ralfroemer99/dpcc](https://github.com/ralfroemer99/dpcc), copyright © 2025 Learning Systems and Robotics Lab (LSY) | [MIT](third_party/diffuser/LICENSE) |
| Original Diffuser implementation underlying the DPCC subtree | [jannerm/diffuser](https://github.com/jannerm/diffuser), copyright © 2020 Phil Wang; © 2022 Michael Janner and Yilun Du | [MIT](third_party/diffuser/LICENSE.diffuser) |
| SafeDiffuser reference implementation for the integration in `genedynamics/solvers/single/safediffuser/` | [Weixy21/SafeDiffuser](https://github.com/Weixy21/SafeDiffuser), copyright © 2025 Wei Xiao | [MIT](genedynamics/solvers/single/safediffuser/LICENSE) |

The vendored Diffuser/DPCC subtree contains local compatibility and sampling
changes; it is not an unmodified upstream release. The SafeDiffuser integration
adapts the invariance-correction workflow to the shared DPCC denoiser and local
D3IL constraint/task interfaces. D3IL is bundled as a simulation subtree rather
than as the complete upstream benchmark repository.

## Robot models and pretrained policy

| Asset and local paths | Upstream and attribution | License text retained in this repository |
| --- | --- | --- |
| Unitree Go2 descriptions and meshes: `genedynamics/envs/assets/unitree_go2/` | [MuJoCo Menagerie / Unitree Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/a03e87bf13502b0b48ebbf2808928fd96ebf9cf3/unitree_go2); copyright © 2016–2022 HangZhou YuShu TECHNOLOGY CO., LTD. (Unitree Robotics) | [BSD-3-Clause](genedynamics/envs/assets/unitree_go2/LICENSE) |
| Unitree H1 descriptions and meshes: `genedynamics/envs/assets/unitree_h1/` | [MuJoCo Menagerie / Unitree H1](https://github.com/google-deepmind/mujoco_menagerie/tree/a03e87bf13502b0b48ebbf2808928fd96ebf9cf3/unitree_h1); copyright © 2016–2023 Unitree Robotics | [BSD-3-Clause](genedynamics/envs/assets/unitree_h1/LICENSE) |
| Adapted G1 model: `genedynamics/envs/assets/unitree_g1/mjx_scene_g1_box_push.xml` | [MuJoCo Menagerie / Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/a03e87bf13502b0b48ebbf2808928fd96ebf9cf3/unitree_g1); copyright © 2016–2023 Unitree Robotics | [BSD-3-Clause](genedynamics/envs/assets/unitree_g1/LICENSE) |
| Reduced Panda arm: `genedynamics/envs/assets/franka_panda/panda_arm.xml` | [MuJoCo Menagerie / Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/a03e87bf13502b0b48ebbf2808928fd96ebf9cf3/franka_emika_panda); derived from the publicly available Franka robot description | [Apache-2.0](genedynamics/envs/assets/franka_panda/LICENSE) |
| Adapted humanoid: `genedynamics/envs/assets/humanoid_run/humanoidrun.xml` | [Brax humanoid model](https://github.com/google/brax/blob/6d039374b34a266107bd1e952180eea33ac4eab8/brax/envs/assets/humanoid.xml), The Brax Authors | [Apache-2.0](genedynamics/envs/assets/humanoid_run/LICENSE) |
| G1 walking checkpoint: `genedynamics/deploy/controllers/sport_mode/assets/g1_motion.pt` | [Unitree RL Gym pretrained G1 policy](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/pre_train/g1/motion.pt); copyright © 2016–2023 Unitree Robotics | [BSD-3-Clause](genedynamics/deploy/controllers/sport_mode/assets/LICENSE) |

The local Go2, H1 and G1 descriptions include GenerativeDynamics scene, actuator,
contact and simulation adaptations. The Panda arm removes meshes and the
gripper, adjusts dynamics/actuators and adds an end-effector site. The humanoid
changes the timestep and actuator gear parameters. The two Apache-licensed XML
files carry modification notices. These changes do not replace the upstream
asset licenses.

The G1 checkpoint is byte-identical to the linked upstream policy (145,745
bytes; SHA-256
`cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`).
It remains a third-party pretrained policy rather than a GenerativeDynamics-trained
model.

The optional `third_party/mujoco_menagerie` submodule is pinned to
`a03e87bf13502b0b48ebbf2808928fd96ebf9cf3`. Its models have individual licenses;
retain the license for each model you redistribute. The local Go2 patch in
`third_party/patches/` changes the upstream BSD-licensed model. The submodule is
not included in the Python wheel.

## Separately installed dependencies

Python dependencies are distributed by their respective publishers under their
own licenses. The [dependency inventory](release/v1_dependency_licenses.json)
covers the core dependency closure only; it is not an audit of all optional
extras and it does not substitute for bundled-source notices.

In particular, the optional `optimization` and `all` extras install
[CVXOPT, licensed GPL-3.0-or-later](https://cvxopt.org/copyright.html). Selecting
the project's MIT license does not change CVXOPT's terms. A distribution that
includes CVXOPT must retain and comply with its license and any applicable
source-distribution obligations. The default core installation does not require
CVXOPT. Torch's optional QP integration uses [qpth](https://github.com/locuslab/qpth),
whose upstream license is Apache-2.0; its dependencies also retain their own
licenses.

## Source verification

Upstream license texts and source comparisons were verified on 2026-10-09 at:

- D3IL: `1d9c71850d4fc1477cee17b557c8b97d70b13071`.
- Diffuser: `7ea422860cc0106e5ca5949d980f04b799d5462c`.
- DPCC: `c2f25ae32e0061ba4cc2e7eb56f5ab7a800bc500`.
- SafeDiffuser: `e41f38ce289175b4294ab25e6565bfeb7d108f3f`.
- Brax: `6d039374b34a266107bd1e952180eea33ac4eab8`.
- MuJoCo Menagerie: pinned `a03e87bf13502b0b48ebbf2808928fd96ebf9cf3`.
- Unitree RL Gym: `276801e46c5d433564f24658bac64f254b7d2d4b`.

These are the revisions used to verify licensing and provenance, not a claim
that every local adaptation was originally imported at those revisions. D3IL's
`d3il_sim/core/Robots.py` matches the verified upstream file exactly; DPCC's
`diffuser/sampling/projection.py` differs only in a raw docstring and whitespace.
