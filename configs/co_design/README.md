# Co-Design Configs

## Directory Structure

```
co_design/
├── main/              # Full MRMFMBD (S1 mode marginalization + S3 fidelity ladder)
│   └── crawling_ground.yaml
├── ablation/          # Single-factor ablations
│   ├── no_mode_marginalization.yaml   # S1 off (num_modes=1)
│   └── no_fidelity_ladder.yaml        # S3 off (num_fidelity_levels=1)
└── baselines/
    └── cmaes_crawling.yaml            # CMA-ES baseline
```

## How to Run

```bash
python scripts/tasks/soft_robot/run_co_design_with_gif.py \
    configs/co_design/main/crawling_ground.yaml
```

GIF output is saved to `output_dir` when `save_gif: true`.

---

## Current Controller

Sinusoidal with velocity feedback + time envelope:

```
raw_i(t) = tanh( W_i · sin(ωt + phases) + b_i + g_i · (1000 · v_com_x) )
act_i(t) = raw_i(t) · σ( a_i + c_i · t/T )
```

- `W`: (10, 4) sin-wave weight matrix
- `b`: (10,) bias vector
- `g`: (10,) per-actuator velocity feedback gain (v_com_x scaled ×1000)
- `a`: (10,) envelope offset
- `c`: (10,) envelope slope
- `ω = 20.0 rad/s`, `frame_dt = 8ms` → period ≈ 39 env steps
- `phi = [W.flatten(), b, g, a, c]` → 80 parameters
- `phi ∈ [-0.5, 0.5]` to keep tanh in its linear region (prevent saturation)

**Velocity feedback** (`g · v_com_x_scaled`): When the robot decelerates,
the feedback term adapts actuator output to compensate. v_com_x is scaled
by 1000 so the feedback is O(0.1), comparable to the sin-wave terms.

**Time envelope** (`σ(a + c · t/T)`): Per-actuator sigmoid modulation
over the rollout horizon. Enables phased strategies: e.g. front actuators
push hard early, then taper off as body deforms.

---

## Controller Improvement Proposals

### A. COM Velocity Feedback (recommended first step)

```
act_i(t) = tanh( W_i · sin(ωt + phases) + b_i + g_i · v_com_x )
```

Add a per-actuator gain `g_i` on the forward COM velocity `v_com_x`:

- Robot decelerating → `v_com_x` drops → feedback term changes → actuators compensate
- Robot reversing → `v_com_x < 0` → feedback flips sign → corrective actuation
- Directly addresses the stall problem: when motion stops, the controller adapts

| Property         | Value                                |
|------------------|--------------------------------------|
| New parameters   | +10 (one gain per actuator)          |
| Total phi_dim    | 60                                   |
| Implementation   | Pass `v_com_x` from carry into controller; already computed in `_env_step` |
| MBD compatible   | Yes (smooth, differentiable)         |

**Implementation** (in `scene.py:_env_step`):
```python
# v_com_x is already available from mass-weighted velocity
feedback = gain_per_actuator * v_com_x   # (n_actuators,)
act_raw = W @ sin_input + b + feedback
act = tanh(act_raw)
```

### B. Time-Varying Amplitude Envelope

```
act_i(t) = tanh( W_i · sin(ωt + phases) + b_i ) · σ(a_i + c_i · t/T)
```

Each actuator learns a slow sigmoid envelope that modulates its amplitude over
the rollout. This enables phased strategies: "push hard with front actuators
for 50 steps, then switch to rear actuators."

| Property         | Value                                |
|------------------|--------------------------------------|
| New parameters   | +20 (a_i, c_i per actuator)          |
| Total phi_dim    | 70                                   |
| Implementation   | Pure math on top of existing controller, no physics changes |
| MBD compatible   | Yes                                  |

**Implementation**:
```python
t_frac = env_t / num_env_steps   # ∈ [0, 1]
envelope = jax.nn.sigmoid(a + c * t_frac)  # (n_actuators,)
act = tanh(W @ sin_input + b) * envelope
```

### C. Piecewise (Segmented) Controller

```
segment = env_t // segment_length
act_i(t) = tanh( W_i[segment] · sin(ωt + phases) + b_i[segment] )
```

Each time segment has independent W and b. The optimizer learns different
actuation patterns for different locomotion phases:

- Segment 0: initial gait launch
- Segment 1: steady-state crawling
- Segment 2: compensate for body drift
- Segment 3: maintain / recover

| Property         | Value                                |
|------------------|--------------------------------------|
| New parameters   | ×N_segments (2 segments → 100, 4 → 200) |
| Total phi_dim    | 100–200                              |
| Implementation   | `jax.lax.switch` to select segment; JAX-compatible |
| MBD compatible   | Marginal — high dim needs larger M   |

**Trade-off**: Powerful but expensive. M=64 insufficient for 200-dim theta;
would need M=128+ or two-phase optimization (morphology first, then controller).

### D. CPG Phase Coupling (most biologically motivated)

```
phase_dot = ω_base + k · v_com_x
phase += phase_dot · dt
act_i(t) = tanh( W_i · sin(phase + phases_i) + b_i )
```

Replace the fixed-frequency clock `ωt` with a velocity-coupled phase oscillator.
Gait frequency adapts to locomotion speed:

- Moving fast → phase advances faster → higher step frequency
- Stalling → phase slows → body has time to recover elastic shape
- Analogous to biological Central Pattern Generators (CPG)

| Property         | Value                                |
|------------------|--------------------------------------|
| New parameters   | +1 (global coupling k) or +10 (per-actuator) |
| Total phi_dim    | 51–60                                |
| Implementation   | Add `phase` to scan carry state; moderate complexity |
| MBD compatible   | Yes, but phase is a new state variable |

**Implementation** (in `_env_step`):
```python
# carry now includes phase: (x, v, C, F, phase)
v_com_x = ...  # from mass-weighted velocity
phase_dot = omega_base + coupling_k * v_com_x
new_phase = phase + phase_dot * dt
sin_input = new_phase + per_actuator_phases
act = tanh(W @ sin(sin_input) + b)
```

---

## Recommended Implementation Order

1. **A (velocity feedback)** — smallest change, directly addresses the stall problem
2. **B (envelope)** — if A alone isn't enough, layer on top of A
3. **D (CPG)** — if periodic gaits still break down, replace the clock entirely
4. **C (piecewise)** — last resort; powerful but expensive to optimize

A and B can be combined: feedback + envelope = 80 phi dims, still tractable for M=64.

---

## Morphology Design

- Voxel grid: 4×3×4 = 48 voxels (configurable via `evaluator_runtime.voxel_dims`)
- Z-axis symmetry: optimizer sees 4×3×2 = 24 dims, mirrored before rollout
- Occupancy range: `x_lo=0.2` to `x_hi=1.0` (density ratio 5:1 for differential deformation)
- Prior: `x_mean=0.6, x_std=0.3`

Inspired by DiffuseBot (NeurIPS 2023): morphology regularization through
bounded occupancy and symmetry constraints, rather than a pretrained generative model.
