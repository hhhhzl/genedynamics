# profile_edoc_numpy.py
"""
Profile EDOC NumPy backend to identify performance bottlenecks.
"""
import cProfile
import pstats
import io
import numpy as np
from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_env, make_energy
from enerdynamics.core.constraints import ConstraintManager
from enerdynamics.core.backends.runtime import RuntimeBackendManager

def profile_edoc_numpy():
    """Profile EDOC NumPy backend."""
    # Force NumPy backend before creating planner
    RuntimeBackendManager.set_backend("numpy", device="cpu")
    
    # Setup environment (adjust parameters as needed)
    env = make_env("single_integrator_box_2d")
    env.dt = 0.05
    env.horizon = 80
    env.control_limit = 1.0
    
    energy = make_energy("single_integrator_box_2d")
    
    # Create planner (will use NumPy backend from RuntimeBackendManager)
    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=80,
        dt=0.05,
        action_diffuse_steps=10,  # Reduce for faster profiling
        action_nsample=64,  # Reduce for faster profiling
        action_score_mode="energy",
    )
    
    # Initial state
    x0 = np.array([-0.2, -1.5], dtype=np.float32)
    rng = 42
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = planner.plan(rng)
    except Exception as e:
        print(f"Error during planning: {e}")
        profiler.disable()
        return
    
    profiler.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("=" * 80)
    print("Top 30 functions by cumulative time:")
    print("=" * 80)
    print(s.getvalue())
    
    # Also print by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(30)
    
    print("=" * 80)
    print("Top 30 functions by total time (excluding subcalls):")
    print("=" * 80)
    print(s2.getvalue())

if __name__ == "__main__":
    profile_edoc_numpy()