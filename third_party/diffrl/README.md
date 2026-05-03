# DiffRL (NVIDIA SHAC reference)

External dependency placeholder — **clone before running Phase 4 (SHAC integration)**.

## What this is

NVlabs/DiffRL is the official source release for *Accelerated Policy Learning
with Parallel Differentiable Simulation* (Xu et al., ICLR 2022). It contains:

- `algorithms/shac.py` — Short-Horizon Actor-Critic (truncated h + terminal critic + td-λ)
- `envs/` — PyTorch + Warp differentiable rigid-body sims (Cartpole, Ant, Humanoid, Humanoid MTU)
- `models/` — actor / critic MLPs, observation normalizers

We use it as the reference implementation for our `genedynamics/solvers/single/shac/`
adapter and (optionally) as the simulator backend for the SHAC-only baseline
in writeup §13 / Table 1.

## Setup

```sh
cd third_party/diffrl
git clone https://github.com/NVlabs/DiffRL.git .
# follow upstream README for Warp + PyTorch install
```

Do **not** vendor the source into the repo; the directory is intentionally a
checkout point so upstream updates are pull-able.

## What we use

| File / module | Purpose in our pipeline |
|---|---|
| `algorithms/shac.py` | Reference for `genedynamics/solvers/single/shac/shac.py` (Phase 4.1) |
| `models/actor.py`, `models/critic.py` | Network shapes for fair comparison |
| `envs/dflex_env.py` | Optional Warp simulator backend for the SHAC-only baseline |

## What we do NOT use

- The Warp simulator for our actual soft-robot experiments — we keep
  `genedynamics/envs/jax_mpm` as the sim of record (MPM, not rigid-body). DiffRL
  is consulted for algorithm structure only.
