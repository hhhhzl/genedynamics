# Co-Design Experiment Configs

Soft robot co-design experiments for the MR-MF-MBD paper.
All configs use the unified `scheduler_config` format (same as `configs/single_2d/mbd.yaml`).

## How to Run

```bash
# In Docker (native x86_64 Linux required for Taichi):
docker compose -f docker/compose.cpu.yml run genedynamics-softzoo-cpu

# Inside container:
python scripts/tasks/soft_robot/run_smoke.py          # Phase A smoke test
python scripts/tasks/soft_robot/run_co_design.py \
    --config configs/co_design/phase_b_s1_id.yaml      # Any experiment

# Generate paper figures after experiments complete:
python scripts/tasks/soft_robot/plot_co_design_results.py \
    --results-root results/ --output-dir figures/co_design/
```

GIF output is saved to the `output_dir` specified in each YAML when `save_gif: true`.

## Experiment Phases

### Phase A: Smoke Test
| Config | Purpose |
|--------|---------|
| `phase_a_smoke.yaml` | End-to-end pipeline verification. K=2, M=2, 1 mode, 1 fidelity. Should complete in <1 min. |

### Phase B: S1 Mode Marginalization (Exp-2 in paper)
Tests whether mode-marginalized guidance (Eq. 41-73) improves robustness vs single-mode optimization.

| Config | Modes | Task | Expected Result |
|--------|-------|------|-----------------|
| `phase_b_nomode_id.yaml` | 1 (no S1) | crawling_ground (ID) | Baseline ID return |
| `phase_b_nomode_ood.yaml` | 1 (no S1) | crawling_desert (OOD) | Large OOD return drop vs ID |
| `phase_b_s1_id.yaml` | 4 (full S1) | crawling_ground (ID) | Comparable or slightly lower ID return |
| `phase_b_s1_ood.yaml` | 4 (full S1) | crawling_desert (OOD) | **Much smaller OOD drop** — this is the S1 contribution |

**Key plots:**
- ID vs OOD grouped bar chart: S1 should show smaller gap
- Mode responsibilities heatmap: w_c should shift across bridge steps as the optimizer discovers which modes matter

### Phase C: S3 Multi-Fidelity Efficiency (Exp-1 in paper)
Tests whether the multi-fidelity ladder (Eq. 88-137) reduces wall-clock time while maintaining return quality.

| Config | Fidelity | Ladder | Expected Result |
|--------|----------|--------|-----------------|
| `phase_c_single_fidelity.yaml` | 1 level (always fine) | none | Best return, highest wall-clock |
| `phase_c_multi_fidelity_geometric.yaml` | 3 levels | geometric | Similar return, **lower wall-clock** |
| `phase_c_multi_fidelity_linear.yaml` | 3 levels | linear | Ablation: geometric should beat linear |

**Key plots:**
- Return vs wall-clock bar chart: multi-fidelity should reach same return faster
- Fidelity usage histogram: geometric allocates more steps to coarse levels

### Phase D: Full Scale (S1 + S3 Joint)
Full method with both contributions enabled. Tests generalization across tasks.

| Config | Task | Notes |
|--------|------|-------|
| `phase_d_full_id.yaml` | crawling_ground | Full MRMFMBD, 4 modes, 3 fidelity levels |
| `phase_d_full_ood.yaml` | crawling_desert | OOD test for full method |
| `phase_d_obstacle.yaml` | obstacle_crossing | Different task to show generalization |

### Phase E: CMA-ES Baseline
Black-box evolutionary optimization with same total rollout budget for fair comparison.

| Config | Task | Expected Result |
|--------|------|-----------------|
| `phase_e_cmaes_id.yaml` | crawling_ground | Lower sample efficiency than MRMFMBD |
| `phase_e_cmaes_ood.yaml` | crawling_desert | No mode awareness, larger OOD drop |

### Ablations

| Config | What it tests | Expected Result |
|--------|---------------|-----------------|
| `ablation_no_prior.yaml` | Value of diffusion prior p0(x) | Slower convergence, lower final return |
| `ablation_temperature_sweep.yaml` | Sensitivity to Boltzmann temperature tau | Too low: greedy collapse. Too high: uniform (no guidance). tau=0.1 is sweet spot |

## Config Format

All configs follow the same structure:

```yaml
baseline_name: mrmfmbd          # or "cmaes"
task_id: crawling_ground         # registered SoftZoo task

scheduler_config:                # same format as single_2d/mbd.yaml
  type: composite
  diffusion_schedulers:
    - type: fixed
      M_k: 6                    # proposals per step
      T_k: 0.1                  # reward temperature (tau)
      Ndiffuse: 30              # annealing steps (K)
      beta0: 1e-6               # annealing start
      betaT: 1.0                # annealing end
  diffusion_merge_strategy: merge

method_params:                   # method-specific
  num_modes: 4                   # S1: contact/friction modes
  num_fidelity_levels: 3         # S3: coarse/medium/fine
  fidelity_ladder_type: geometric
  fidelity_step_ratio: 1.5
  show_tqdm: true
```

Parameters are extracted from `scheduler_config` by the framework — `M_k` maps to MCSA batch size, `Ndiffuse` maps to bridge steps K, `T_k` maps to reward temperature, `beta0/betaT` define the annealing schedule.
