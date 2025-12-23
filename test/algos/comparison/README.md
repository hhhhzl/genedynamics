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

### 5. `test_trajectory_qp_difference.py` ⭐ **NEW**
**Purpose**: Diagnose trajectory QP vs pointwise projection differences

This diagnostic script identifies if trajectory QP is the main source of differences:
- Compares NumPy trajectory QP vs NumPy pointwise projection
- Compares NumPy trajectory QP vs JAX pointwise (actual experiment scenario)
- Measures smoothness differences
- Provides actionable recommendations

**Run**: `python test/algos/comparison/test_trajectory_qp_difference.py`

**Use this when**: You see different results between NumPy and JAX in experiments with `use_trajectory_qp=True`

## Usage

Run all tests:
```bash
cd /path/to/enerdynamics
python test/algos/comparison/test_qp_solver_comparison.py
python test/algos/comparison/test_cfs_projection_comparison.py
python test/algos/comparison/test_edoc_jax_vs_python.py
python test/algos/comparison/test_cfs_step_by_step.py
python test/algos/comparison/test_trajectory_qp_difference.py
```

Or run individually to focus on specific differences.

## Expected Findings

Based on code analysis, expect to find differences in:

1. **QP Solver**: JAX simplified method is suboptimal compared to NumPy enumeration
2. **SDF Accuracy**: SDF texture is an approximation (depends on resolution)
3. **Iteration Control**: JAX always runs max_iterations, NumPy can break early
4. **Trajectory QP**: NumPy has trajectory-level QP, JAX doesn't

## Known Differences Between JAX and NumPy Versions

### 1. **Trajectory QP**
- **NumPy**: Supports `use_trajectory_qp=True` for trajectory-level optimization
- **JAX**: Only supports pointwise projection (trajectory QP not implemented)
- **Impact**: JAX version may produce different results when `use_trajectory_qp=True`

### 2. **SDF Computation**
- **NumPy**: Uses exact per-obstacle SDF computation
- **JAX**: May use SDF texture (grid-based approximation) when `build_sdf_texture_2d()` is called
- **Impact**: SDF texture introduces discretization error (depends on resolution)
- **Note**: If all obstacles support `jax_sdf`, JAX will use exact per-obstacle SDF

### 3. **QP Solver**
- **NumPy**: Uses enumeration method (full active set search)
- **JAX**: Uses qpax or simplified method (only most violated constraint)
- **Impact**: Different QP solutions can lead to different projections

### 4. **Performance**
- **First Call**: JAX version is slower due to JIT compilation (warmup added to mitigate)
- **Subsequent Calls**: JAX version is typically faster due to JIT optimization
- **Note**: Warmup mechanism pre-compiles JAX functions to avoid first-call slowdown

## Diagnostic Workflow

If you're seeing different results between NumPy and JAX:

1. **First, run the diagnostic script**:
   ```bash
   python test/algos/comparison/test_trajectory_qp_difference.py
   ```
   This will tell you if trajectory QP is the issue.

2. **If trajectory QP is the problem**:
   - **Quick fix**: Set `use_trajectory_qp=False` in your experiment config
   - **Long-term**: Implement trajectory QP in JAX version

3. **If differences persist**, run other comparison tests to isolate the component:
   - `test_cfs_step_by_step.py` - Check SDF/gradient/QP differences
   - `test_qp_solver_comparison.py` - Check QP solver differences
   - `test_cfs_projection_comparison.py` - Check overall projection differences

## Next Steps

After running tests, use results to:
1. Identify which component causes the largest differences
2. Prioritize fixes (e.g., implement trajectory QP in JAX, improve QP solver)
3. Add numerical tolerances or approximations where acceptable
4. Consider using `use_trajectory_qp=False` for better JAX/NumPy alignment

