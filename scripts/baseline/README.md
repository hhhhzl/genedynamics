# Baseline Freeze & Regression (Phase 0)

## Quick Start

1. **Run baseline experiments** (produces `results/single2d/*`, `results/d3il_avoiding/*`):
   ```bash
   ./scripts/baseline/run_baseline.sh
   ```
   Smoke (minimal set):
   ```bash
   ./scripts/baseline/run_baseline.sh smoke
   ```

2. **Freeze baseline** (copy results for later comparison):
   ```bash
   ./scripts/baseline/freeze_baseline.sh
   ```
   Saves to `results_baseline_YYYYMMDD_HHMM/`. Or specify:
   ```bash
   ./scripts/baseline/freeze_baseline.sh results_baseline_golden
   ```

3. **After code changes, compare**:
   ```bash
   python scripts/baseline/compare_regression.py results_baseline_golden results
   ```

## Thresholds

- `success`, `collision`: must match exactly
- `ssr`, `execution_ssr`: `|new - baseline| <= 1e-6`
- `best_idx`: must match
- `candidate_costs`: `max|new - baseline| <= 1e-6`

## Configs Covered

- **single_2d**: mbd, ebmbd, mdoc, mdcoas, mdcoas-a, mdcoas-f
- **d3il_avoiding**: ebmbd, mdoc, mdcoas, mdcoas-f, dpcc, dpcc_9d, safediffuser, safediffuser_9d
