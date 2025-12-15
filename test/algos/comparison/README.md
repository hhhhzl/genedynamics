# CFS JAX vs NumPy Comparison Tests

This directory contains test scripts to analyze differences between JAX and NumPy implementations of CFS projection.

## Test Scripts

### 1. `test_qp_solver_comparison.py`
**Purpose**: Isolate QP solver differences

Compares:
- NumPy enumeration method (full active set search)
- JAXOpt BoxOSQP (general QP solver)
- JAX simplified method (only most violated constraint)

**Run**: `python test/comparison/test_qp_solver_comparison.py`

### 2. `test_cfs_projection_comparison.py`
**Purpose**: Direct CFS projection comparison

Compares NumPy and JAX CFS projection on:
- Same input trajectory
- Same obstacles
- Same clearance

**Run**: `python test/comparison/test_cfs_projection_comparison.py`

### 3. `test_edoc_jax_vs_python.py`
**Purpose**: EDOC loop differences

Compares:
- EDOC without CFS (JAX loop baseline)
- EDOC with NumPy CFS (Python loop)
- EDOC with JAX CFS (JAX loop)

**Run**: `python test/comparison/test_edoc_jax_vs_python.py`

### 4. `test_cfs_step_by_step.py`
**Purpose**: Step-by-step component comparison

Breaks down CFS into steps:
1. SDF computation
2. Gradient computation
3. Constraint building
4. QP solving
5. Iteration loop

**Run**: `python test/comparison/test_cfs_step_by_step.py`

## Usage

Run all tests:
```bash
cd /path/to/enerdynamics
python test/comparison/test_qp_solver_comparison.py
python test/comparison/test_cfs_projection_comparison.py
python test/comparison/test_edoc_jax_vs_python.py
python test/comparison/test_cfs_step_by_step.py
```

Or run individually to focus on specific differences.

## Expected Findings

Based on code analysis, expect to find differences in:

1. **QP Solver**: JAX simplified method is suboptimal compared to NumPy enumeration
2. **SDF Accuracy**: SDF texture is an approximation (depends on resolution)
3. **Iteration Control**: JAX always runs max_iterations, NumPy can break early
4. **Trajectory QP**: NumPy has trajectory-level QP, JAX doesn't

## Next Steps

After running tests, use results to:
1. Identify which component causes the largest differences
2. Prioritize fixes (e.g., implement trajectory QP in JAX, improve QP solver)
3. Add numerical tolerances or approximations where acceptable

