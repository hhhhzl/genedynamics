from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Dummy tqdm if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter, EnvDynamicsAdapter
from enerdynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from enerdynamics.core.backends import Backend, JaxBackend
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.types import State, Action, Trajectory
from enerdynamics.core.constraints.legacy.base import project_box
from enerdynamics.core.constraints import ConstraintManager
from enerdynamics.core.constraints.core import HighPerformanceConstraintPipeline
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.envs.factories import make_env, make_energy

# Register solver to registry
try:
    from enerdynamics.core.registry.solvers import register_solver
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_solver = None

# Register solver to registry
try:
    from enerdynamics.core.registry.solvers import register_solver
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    register_solver = None

# Register backends to registry
try:
    from .backends import edoc_jax, edoc_numpy
except ImportError:
    pass


# ============================================================================
# EDOC Planner (Core Implementation)
# ============================================================================

class EDOCPlanner:
    """
    EDOC planner implementation.
    
    This is the core EDOC algorithm. It can be used directly or wrapped
    in the EDOCSolver class for the unified interface.
    """

    def __init__(
        self,
        env,
        energy: LegacyEnergyFunctional,
        horizon: int,
        dt: float,
        noise_std: float = 0.05,
        state_box: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        action_space: bool = True,
        diffusion_mode: str = "reverse",
        action_diffuse_steps: int = 100,
        action_beta0: float = 1e-4,
        action_betaT: float = 1e-1,
        action_temp: float = 0.5,
        action_extra_sigma: float = 0.0,
        # NOTE: action_stage_ratio is deprecated; EDOC now uses guided scoring at all steps.
        action_stage_ratio: float = 0.95,
        action_score_mode: str = "reward",
        action_nsample: int = 256,
        use_antithetic: bool = False,
        dyn_loss_coeff: float = 1.0,
        dyn_loss_mode: str = "terminal",
        np_random_seed: Optional[int] = None,
        # ===== UX / profiling =====
        show_tqdm: bool = True,
        tqdm_chunk_len: int = 10,
        # ===== Constraint system integration =====
        constraint_manager: Optional[ConstraintManager] = None,
        constraint_pipeline: Optional[HighPerformanceConstraintPipeline] = None,
        lambda_energy: float = 1.0,  # Energy scaling in E_soft = (1/λ)J + S
        use_constraint_in_scoring: bool = True,  # Include soft constraints in scoring
        # ===== Terminal cost (for energy-mode) =====
        terminal_energy_weight: float = 0.0,
    ):
        if action_score_mode not in {"reward", "energy", "learned"}:
            raise ValueError(f"Unknown action_score_mode {action_score_mode}")
        if dyn_loss_mode not in {"terminal", "trajectory"}:
            raise ValueError(f"Unknown dyn_loss_mode {dyn_loss_mode}")
        assert horizon > 0 and isinstance(horizon, int), "Horizon must be a positive integer"
        assert dt > 0 and isinstance(dt, float), "Time step must be a positive float"
        assert noise_std >= 0 and isinstance(noise_std, float), "Noise standard deviation must be non-negative"
        assert diffusion_mode in {"reverse", "forward"}, "Invalid diffusion mode"
        assert action_diffuse_steps > 0 and isinstance(action_diffuse_steps, int), "Action diffuse steps must be a positive integer"
        assert action_beta0 > 0 and isinstance(action_beta0, float), "Action beta0 must be positive"
        assert action_betaT > 0 and isinstance(action_betaT, float), "Action betaT must be positive"
        assert action_temp > 0 and isinstance(action_temp, float), "Action temp must be positive"
        assert action_extra_sigma >= 0 and isinstance(action_extra_sigma, float), "Action extra sigma must be non-negative"
        assert 0 <= action_stage_ratio <= 1, "Action stage ratio must be between 0 and 1"
        assert action_score_mode in {"reward", "energy", "learned"}, "Invalid action score mode"
        assert action_nsample > 0 and isinstance(action_nsample, int), "Action sample size must be a positive integer"
        assert dyn_loss_coeff >= 0 and isinstance(dyn_loss_coeff, float), "Dynamics loss coefficient must be non-negative"
        assert dyn_loss_mode in {"terminal", "trajectory"}, "Invalid dynamics loss mode"
        assert np_random_seed is None or isinstance(np_random_seed, int), "Random seed must be an integer or None"
        assert isinstance(show_tqdm, bool), "Show tqdm must be a boolean"
        assert tqdm_chunk_len > 0 and isinstance(tqdm_chunk_len, int), "Tqdm chunk length must be a positive integer"
        assert constraint_manager is None or isinstance(constraint_manager, ConstraintManager), "Constraint manager must be a ConstraintManager or None"
        assert constraint_pipeline is None or isinstance(constraint_pipeline, HighPerformanceConstraintPipeline), "Constraint pipeline must be a HighPerformanceConstraintPipeline or None"
        assert lambda_energy >= 0 and isinstance(lambda_energy, float), "Lambda energy must be non-negative"
        assert isinstance(use_constraint_in_scoring, bool), "Use constraint in scoring must be a boolean"
        assert terminal_energy_weight >= 0 and isinstance(terminal_energy_weight, float), "Terminal energy weight must be non-negative"

        self.use_antithetic = bool(use_antithetic)
        self.env = env
        self.energy = energy
        self.horizon = horizon
        self.dt = dt
        self.noise_std = noise_std
        self.state_box = state_box
        self.action_space = action_space
        self.diffusion_mode = diffusion_mode
        self.action_diffuse_steps = action_diffuse_steps
        self.action_beta0 = action_beta0
        self.action_betaT = action_betaT
        self.action_temp = action_temp
        self.action_extra_sigma = action_extra_sigma
        # Deprecated (kept for backward compatibility)
        self.action_stage_ratio = np.clip(action_stage_ratio, 0.0, 1.0)
        self.action_score_mode = action_score_mode
        self.action_nsample = max(1, int(action_nsample))
        self.dyn_loss_coeff = float(max(0.0, dyn_loss_coeff))
        self.dyn_loss_mode = dyn_loss_mode
        self._np_rng = np.random.default_rng(np_random_seed)
        self.show_tqdm = bool(show_tqdm)
        
        # ===== Constraint system =====
        # Initialize constraint_manager and constraint_pipeline early so they're available for backend initialization
        self.constraint_manager = constraint_manager
        self.constraint_pipeline = constraint_pipeline
        self.lambda_energy = float(lambda_energy) if lambda_energy > 0 else 1.0
        self.use_constraint_in_scoring = bool(use_constraint_in_scoring)
        self.terminal_energy_weight = float(max(0.0, terminal_energy_weight))
        self._reverse_diffuse_chunk_len = int(max(1, tqdm_chunk_len))
        
        # ===== Backend detection =====
        # Detect current backend to determine which backend implementation to use
        self._backend_impl = None
        self._use_numpy_backend = False  # Initialize to avoid AttributeError
        try:
            from enerdynamics.core.registry.edoc_backends import get_edoc_backend_registry
            from enerdynamics.core.backends.runtime import RuntimeBackendManager
            backend = RuntimeBackendManager.get_backend()
            backend_name = backend.name
            
            registry = get_edoc_backend_registry()
            backend_class = registry.get(backend_name)
            if backend_class is not None:
                self._backend_impl = backend_class(self)
                self._use_numpy_backend = (backend_name == "numpy")
                print(f"[EDOC] Using {backend_name} backend implementation")
            else:
                # Fallback to JAX if available, otherwise NumPy
                backend_class = registry.get("jax") or registry.get("numpy")
                if backend_class is not None:
                    self._backend_impl = backend_class(self)
                    fallback_name = "jax" if registry.get("jax") is not None else "numpy"
                    self._use_numpy_backend = (fallback_name == "numpy")
                    print(f"[EDOC] Backend {backend_name} not available, falling back to {fallback_name}")
                else:
                    # Last resort: direct import
                    try:
                        if backend_name == "numpy" or backend_name != "jax":
                            from .backends.edoc_numpy import EDOCBackendNumpy
                            self._backend_impl = EDOCBackendNumpy(self)
                            self._use_numpy_backend = True
                            print(f"[EDOC] Registry failed, using direct import for NumPy backend")
                        else:
                            from .backends.edoc_jax import EDOCBackendJax
                            self._backend_impl = EDOCBackendJax(self)
                            self._use_numpy_backend = False
                            print(f"[EDOC] Registry failed, using direct import for JAX backend")
                    except Exception as e2:
                        print(f"[EDOC] ERROR: Could not import any backend: {e2}")
                        self._backend_impl = None
        except (ImportError, AttributeError, Exception) as e:
            # Registry not available or error, use fallback detection
            self._use_numpy_backend = False
            try:
                backend = RuntimeBackendManager.get_backend()
                backend_name = backend.name
                self._use_numpy_backend = (backend_name == "numpy")
                
                # Try to get registry again or use direct import
                try:
                    registry = get_edoc_backend_registry()
                except Exception:
                    # Direct import fallback
                    if backend_name == "numpy":
                        from .backends.edoc_numpy import EDOCBackendNumpy
                        backend_class = EDOCBackendNumpy
                    else:
                        from .backends.edoc_jax import EDOCBackendJax
                        backend_class = EDOCBackendJax
                    
                    if backend_class is not None:
                        self._backend_impl = backend_class(self)
                        print(f"[EDOC] Registry not available, using direct import for {backend_name} backend")
                else:
                    # Registry available, try to get backend class
                    backend_class = registry.get(backend_name)
                    if backend_class is None:
                        # Fallback to numpy if requested backend not found
                        if self._use_numpy_backend:
                            backend_class = registry.get("numpy")
                        else:
                            backend_class = registry.get("jax") or registry.get("numpy")
                    
                    if backend_class is not None:
                        self._backend_impl = backend_class(self)
                        print(f"[EDOC] Using {backend_name} backend implementation (fallback)")
                    else:
                        # Last resort: direct import
                        try:
                            if self._use_numpy_backend:
                                from .backends.edoc_numpy import EDOCBackendNumpy
                                self._backend_impl = EDOCBackendNumpy(self)
                            else:
                                from .backends.edoc_jax import EDOCBackendJax
                                self._backend_impl = EDOCBackendJax(self)
                            print(f"[EDOC] Registry returned None, using direct import for {backend_name} backend")
                        except Exception as e3:
                            print(f"[EDOC] ERROR: Could not import backend: {e3}")
                            self._backend_impl = None
            except Exception as e2:
                # Last resort: try to import numpy backend directly
                try:
                    from .backends.edoc_numpy import EDOCBackendNumpy
                    self._backend_impl = EDOCBackendNumpy(self)
                    self._use_numpy_backend = True
                    print(f"[EDOC] All detection failed, using NumPy backend as last resort")
                except Exception as e3:
                    print(f"[EDOC] ERROR: Could not initialize any backend: {e3}")
                    self._backend_impl = None
        
        # If using NumPy backend, ensure CFS uses Python backend (cvxopt)
        if self._use_numpy_backend and self.constraint_manager is not None:
            feasibility_op = self.constraint_manager.feasibility_operator
            if feasibility_op is not None:
                # Use registry system if available, otherwise use legacy attribute
                if hasattr(feasibility_op, '_backend_impl') and feasibility_op._backend_impl is not None:
                    # Registry-based: backend_impl should already be set correctly by CFSProjection.__init__
                    # Check if it's using NumPy backend
                    backend_impl_type = type(feasibility_op._backend_impl).__name__
                    if "Numpy" not in backend_impl_type:
                        print(f"[EDOC] Warning: NumPy backend detected but CFS is using {backend_impl_type}")
                elif hasattr(feasibility_op, '_use_python_backend'):
                    # Legacy: force CFS to use Python backend
                    feasibility_op._use_python_backend = True
                    print(f"[EDOC] Forced CFS to use Python backend (cvxopt) for NumPy backend")
        
        # Validate action_space requirement
        if self.action_space:
            # Backend will handle environment interface validation
            pass


    # ------------------- main interface -------------------
    def plan(self, rng: Any) -> Dict[str, Any]:
        """
        Main planning interface.
        """
        is_jax_key = False
        try:
            if hasattr(rng, 'shape') and hasattr(rng, 'dtype'):
                if len(rng.shape) == 1 and rng.shape[0] == 2:
                    is_jax_key = True
        except (AttributeError, TypeError):
            pass
        
        if not is_jax_key:
            if isinstance(rng, (int, np.integer)):
                rng = jax.random.PRNGKey(int(rng))
            else:
                try:
                    seed = int(rng) if hasattr(rng, '__int__') else hash(rng) % (2**31)
                    rng = jax.random.PRNGKey(seed)
                except (TypeError, ValueError):
                    # Fallback to default seed
                    rng = jax.random.PRNGKey(0)
        
        try:
            x0, info = self.env.reset(rng)
        except TypeError:
            x0, info = self.env.reset()
        
        if self.action_space:
            return self._run_action_reverse(x0, info, rng)
        else:
            raise ValueError("EDOC planner requires action_space=True for reverse diffusion")


    def _run_action_reverse(self, x0, info, rng):
        """Reverse diffusion in action space."""
        # Use backend implementation if available
        if self._backend_impl is not None:
            rng_out, actions_np_final, reward_history_arr, diffusion_actions_traj_arr, diffusion_samples_traj_arr = \
                self._backend_impl.reverse_diffuse(rng, x0, None, None, None, None)
            
            # Convert backend result to final format
            return self._finalize_action_reverse_result(x0, actions_np_final, reward_history_arr,
                                                       diffusion_actions_traj_arr, diffusion_samples_traj_arr)
        else:
            raise ValueError("No backend implementation found")
    
    def _finalize_action_reverse_result(self, x0, actions_np_final, reward_history_arr,
                                       diffusion_actions_traj_arr, diffusion_samples_traj_arr):
        """
        Finalize action reverse result by performing final trajectory rollout.
        
        Uses backend to compute final trajectory states, energies, and rewards.
        """
        # Use backend to rollout final trajectory
        states_full, energies_seq = self._backend_impl.rollout_states_and_energy(
            x0, actions_np_final, 0.0, False
        )
        
        # Compute rewards if available
        rewards_arr = np.zeros((len(actions_np_final),), dtype=np.float32)
        if hasattr(self.env, "cost"):
            env_states = self._backend_impl.rollout_env_states(x0, actions_np_final, 0.0, False)
            for i in range(1, len(env_states)):
                rewards_arr[i-1] = -float(self.env.cost(env_states[i]))
        
        # Compute terminal energy
        terminal_energy = 0.0
        if self.terminal_energy_weight > 0.0:
            u0 = np.zeros((self.env.act_dim,), dtype=np.float32)
            ctxT = {"t": int(len(actions_np_final))}
            terminal_energy = float(self.terminal_energy_weight) * float(
                self.energy.compute(states_full[-1], u0, ctxT)
            )
        
        # Convert to appropriate array types based on backend
        if not self._use_numpy_backend:
            # JAX backend: convert to JAX arrays
            states_full = jnp.asarray(states_full, dtype=jnp.float32)
            energies_arr = jnp.asarray(energies_seq, dtype=jnp.float32)
            rewards_arr = jnp.asarray(rewards_arr, dtype=jnp.float32)
            actions_np_final = jnp.asarray(actions_np_final, dtype=jnp.float32)
            if reward_history_arr is not None:
                reward_history_arr = jnp.asarray(reward_history_arr, dtype=jnp.float32)
            if diffusion_actions_traj_arr is not None:
                diffusion_actions_traj_arr = jnp.asarray(diffusion_actions_traj_arr, dtype=jnp.float32)
            if diffusion_samples_traj_arr is not None:
                diffusion_samples_traj_arr = jnp.asarray(diffusion_samples_traj_arr, dtype=jnp.float32)
            terminal_energy = jnp.asarray(terminal_energy, dtype=jnp.float32)
        else:
            # NumPy backend: ensure NumPy arrays
            energies_arr = np.asarray(energies_seq, dtype=np.float32)
            if reward_history_arr is not None and isinstance(reward_history_arr, jnp.ndarray):
                reward_history_arr = np.asarray(reward_history_arr, dtype=np.float32)
            if diffusion_actions_traj_arr is not None and isinstance(diffusion_actions_traj_arr, jnp.ndarray):
                diffusion_actions_traj_arr = np.asarray(diffusion_actions_traj_arr, dtype=np.float32)
            if diffusion_samples_traj_arr is not None and isinstance(diffusion_samples_traj_arr, jnp.ndarray):
                diffusion_samples_traj_arr = np.asarray(diffusion_samples_traj_arr, dtype=np.float32)
        
        return {
            "states": states_full,
            "energies": energies_arr,
            "terms": [],
            "rewards": rewards_arr,
            "actions": actions_np_final,
            "initial_state": states_full[0],
            "terminal_energy": terminal_energy,
            "reward_history": reward_history_arr if reward_history_arr is not None else (jnp.asarray([], dtype=jnp.float32) if not self._use_numpy_backend else np.array([], dtype=np.float32)),
            "diffusion_actions_traj": diffusion_actions_traj_arr if diffusion_actions_traj_arr is not None else (jnp.asarray([], dtype=jnp.float32) if not self._use_numpy_backend else np.array([], dtype=np.float32)),
            "diffusion_sampled_actions": diffusion_samples_traj_arr if diffusion_samples_traj_arr is not None else (jnp.asarray([], dtype=jnp.float32) if not self._use_numpy_backend else np.array([], dtype=np.float32)),
        }

# ============================================================================
# Energy Adapter for Legacy Planners
# ============================================================================

class EnergyToLegacyAdapter:
    """
    Adapter that converts a new EnergyFunctional to LegacyEnergyFunctional.
    
    This is a temporary bridge until EDOCPlanner is fully refactored.
    The adapter creates a point-wise energy function that approximates
    the trajectory-based energy by evaluating single-step trajectories.
    """
    
    def __init__(self, energy: EnergyFunctional, dynamics: DynamicsModel):
        """
        Initialize adapter.
        
        Args:
            energy: New EnergyFunctional
            dynamics: Dynamics model (for trajectory rollout)
        """
        from enerdynamics.core.energy import EnergyTerm
        
        self.energy = energy
        self.dynamics = dynamics
        
        # Create a legacy energy that evaluates trajectories point-wise
        def trajectory_energy(x, u, ctx):
            """
            Evaluate energy at a single (state, action) pair.
            
            This creates a single-step trajectory and evaluates it using
            the new EnergyFunctional interface.
            """
            try:
                # Create a single-step trajectory
                x_next = self.dynamics.step(x, u)
                states = [x, x_next]
                actions = [u]
                traj = Trajectory(states=states, actions=actions)
                
                # Evaluate using new energy functional
                # This gives us the energy for this single step
                total_energy = self.energy.total_energy(traj)
                
                # Return as a scalar (JAX array)
                if isinstance(total_energy, (jnp.ndarray, np.ndarray)):
                    return float(total_energy)
                return float(total_energy)
            except Exception as e:
                # Fallback: return a default value
                # In production, you might want to log this
                return 0.0
        
        self._legacy_energy = LegacyEnergyFunctional({
            "total": EnergyTerm(trajectory_energy, 1.0)
        })
    
    def __call__(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy(*args, **kwargs)
    
    def compute(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy.compute(*args, **kwargs)
    
    def breakdown(self, *args, **kwargs):
        """Delegate to legacy energy."""
        return self._legacy_energy.breakdown(*args, **kwargs)
    
    @property
    def legacy_energy(self):
        """Get the legacy energy functional."""
        return self._legacy_energy


# ============================================================================
# EDOC Solver (Unified Interface)
# ============================================================================

class EDOCSolver(SamplingSolver):
    """
    EDOC solver that follows the new unified Solver interface.
    
    This wraps EDOCPlanner to make it compatible with the unified architecture.
    EDOC uses diffusion-based optimization, so it directly optimizes rather than
    sampling and selecting.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        horizon: int = 80,
        dt: float = 0.1,
        noise_std: float = 0.05,
        action_space: bool = True,
        diffusion_mode: str = "reverse",
        action_diffuse_steps: int = 100,
        action_beta0: float = 1e-4,
        action_betaT: float = 1e-1,
        action_temp: float = 2,
        action_extra_sigma: float = 0.0,
        # Deprecated (kept for backward compatibility; EDOC is guided at all steps)
        action_stage_ratio: float = 1.0,
        action_score_mode: str = "energy",
        action_nsample: int = 128,
        use_antithetic: bool = True,
        dyn_loss_coeff: float = 1.0,
        dyn_loss_mode: str = "trajectory",
        state_box: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        seed: int = 0,
        **kwargs
    ):
        """
        Initialize EDOC solver.
        
        Args:
            dynamics: Dynamics model
            energy: Energy functional
            backend: Computational backend (must be JAX for EDOC)
            horizon: Planning horizon
            dt: Time step
            noise_std: Noise standard deviation for Langevin diffusion
            action_space: If True, optimize in action space; if False, optimize in state space
            diffusion_mode: "forward" or "reverse" diffusion
            action_diffuse_steps: Number of diffusion steps
            action_beta0: Initial noise level for diffusion
            action_betaT: Final noise level for diffusion
            action_temp: Temperature for sampling
            action_extra_sigma: Extra noise variance
            action_stage_ratio: Ratio for multi-stage diffusion
            action_score_mode: "energy", "reward", or "learned" scoring mode
            action_nsample: Number of action samples per diffusion step
            use_antithetic: Whether to use antithetic sampling
            dyn_loss_coeff: Coefficient for dynamics mismatch loss
            dyn_loss_mode: "terminal" or "trajectory" dynamics loss
            state_box: Optional state box constraints (low, high)
            seed: Random seed
            **kwargs: Additional EDOC-specific parameters
        """
        super().__init__(dynamics, energy, backend, **kwargs)
        
        # EDOC currently requires JAX backend
        if backend.name != "jax":
            raise ValueError(
                f"EDOC solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )
        
        # Store configuration
        self.horizon = horizon
        self.dt = dt
        self.seed = seed
        self.config.update({
            "noise_std": noise_std,
            "action_space": action_space,
            "diffusion_mode": diffusion_mode,
            "action_diffuse_steps": action_diffuse_steps,
            "action_beta0": action_beta0,
            "action_betaT": action_betaT,
            "action_temp": action_temp,
            "action_extra_sigma": action_extra_sigma,
            "action_stage_ratio": action_stage_ratio,
            "action_score_mode": action_score_mode,
            "action_nsample": action_nsample,
            "use_antithetic": use_antithetic,
            "dyn_loss_coeff": dyn_loss_coeff,
            "dyn_loss_mode": dyn_loss_mode,
        })
        
        # Create adapters
        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        
        # Convert energy to legacy format if needed
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy
        
        # Create EDOCPlanner (lazy initialization)
        self._edoc_planner = None
        self._state_box = state_box
    
    def _get_edoc_planner(self):
        """Get or create EDOCPlanner instance."""
        if self._edoc_planner is None:
            self._edoc_planner = EDOCPlanner(
                env=self._env_adapter,
                energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                noise_std=self.config["noise_std"],
                state_box=self._state_box,
                action_space=self.config["action_space"],
                diffusion_mode=self.config["diffusion_mode"],
                action_diffuse_steps=self.config["action_diffuse_steps"],
                action_beta0=self.config["action_beta0"],
                action_betaT=self.config["action_betaT"],
                action_temp=self.config["action_temp"],
                action_extra_sigma=self.config["action_extra_sigma"],
                action_stage_ratio=self.config["action_stage_ratio"],
                action_score_mode=self.config["action_score_mode"],
                action_nsample=self.config["action_nsample"],
                use_antithetic=self.config["use_antithetic"],
                dyn_loss_coeff=self.config["dyn_loss_coeff"],
                dyn_loss_mode=self.config["dyn_loss_mode"],
                np_random_seed=self.seed,
            )
        return self._edoc_planner
    
    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs
    ) -> List[Trajectory]:
        """
        Sample candidate trajectories using EDOC diffusion process.
        
        This method uses EDOC's diffusion to generate multiple candidate trajectories.
        Note: EDOC typically optimizes directly, so this is mainly for visualization/debugging.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of trajectories to sample
            **kwargs: Additional sampling parameters
            
        Returns:
            List of candidate trajectories
        """
        planner = self._get_edoc_planner()
        rng = jax.random.PRNGKey(self.seed)
        
        # Extract state data
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        
        trajectories = []
        for i in range(n_samples):
            # Use different random keys for each sample
            rng, sample_key = jax.random.split(rng)
            
            # Set initial state in environment adapter
            self._env_adapter._current_state = x0_data
            
            # Run EDOC planning
            result = planner.plan(sample_key)
            
            # Convert result to Trajectory
            states = result.get("states", [])
            actions = result.get("actions", None)
            
            if actions is not None:
                # Convert to list of actions
                actions_list = [actions[i] for i in range(len(actions))]
            else:
                # Generate actions from state transitions
                actions_list = []
                for j in range(len(states) - 1):
                    # Infer action from state transition
                    # This is approximate
                    actions_list.append(np.zeros(self._env_adapter.act_dim, dtype=np.float32))
            
            # Convert states to list
            states_list = [states[i] for i in range(len(states))]
            
            traj = Trajectory(states=states_list, actions=actions_list)
            trajectories.append(traj)
        
        return trajectories
    
    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs
    ) -> Trajectory:
        """
        Solve for optimal trajectory using EDOC.
        
        This method uses EDOCPlanner's diffusion-based optimization to find
        the optimal trajectory.
        
        Args:
            x0: Initial state
            horizon: Planning horizon (overrides constructor value if different)
            **kwargs: Additional solver parameters (e.g., rng_key for random seed)
            
        Returns:
            Optimized trajectory
        """
        planner = self._get_edoc_planner()
        
        # Use provided horizon or default
        if horizon != self.horizon:
            # Recreate planner with new horizon
            self.horizon = horizon
            self._edoc_planner = None
            planner = self._get_edoc_planner()
        
        # Get random key
        rng_key = kwargs.get("rng_key", None)
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        
        # Extract state data
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        
        # Set initial state in environment adapter
        self._env_adapter._current_state = x0_data
        
        # Run EDOC planning
        result = planner.plan(rng_key)
        
        # Convert result to Trajectory
        states = result.get("states", [])
        actions = result.get("actions", None)
        
        if actions is not None:
            # Convert JAX array to list of actions
            if isinstance(actions, jnp.ndarray):
                actions_list = [np.asarray(actions[i], dtype=np.float32) for i in range(len(actions))]
            else:
                actions_list = [actions[i] for i in range(len(actions))]
        else:
            # Generate dummy actions if not available
            actions_list = [np.zeros(self._env_adapter.act_dim, dtype=np.float32) 
                          for _ in range(len(states) - 1)]
        
        # Convert states to list
        if isinstance(states, jnp.ndarray):
            states_list = [np.asarray(states[i], dtype=np.float32) for i in range(len(states))]
        else:
            states_list = [states[i] for i in range(len(states))]
        
        # Create trajectory
        traj = Trajectory(
            states=states_list,
            actions=actions_list,
            info=result
        )
        
        return traj


# Register EDOC solver to registry
if REGISTRY_AVAILABLE and register_solver is not None:
    register_solver("edoc", EDOCPlanner)


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_edoc(args):
    """
    Run EDOC planner (main entry point for backward compatibility).
    
    Args:
        args: EDOC configuration arguments (EDOCArgs from configs)
        
    Returns:
        Dictionary with planning results
    """
    rng = jax.random.PRNGKey(args.seed)
    np_seed = args.np_random_seed if args.np_random_seed is not None else args.seed
    if np_seed is not None:
        np.random.seed(np_seed)

    # env & energy
    env = make_env(args.env_name)
    energy = make_energy(args.env_name)

    state_box = None
    if args.use_state_box:
        low = jnp.array([args.state_low, args.state_low], dtype=jnp.float32)
        high = jnp.array([args.state_high, args.state_high], dtype=jnp.float32)
        state_box = (low, high)

    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=args.horizon,
        dt=args.dt,
        noise_std=args.noise_std,
        state_box=state_box,
        action_space=args.action_space,
        diffusion_mode=args.diffusion_mode,
        action_diffuse_steps=args.action_diffuse_steps,
        action_beta0=args.action_beta0,
        action_betaT=args.action_betaT,
        action_temp=args.action_temp,
        action_extra_sigma=args.action_extra_sigma,
        action_stage_ratio=args.action_stage_ratio,
        action_score_mode=args.action_score_mode,
        action_nsample=args.action_nsample,
        use_antithetic=args.use_antithetic,
        dyn_loss_coeff=args.dyn_loss_coeff,
        dyn_loss_mode=args.dyn_loss_mode,
        np_random_seed=np_seed,
    )

    out = planner.plan(rng)

    if args.verbose:
        if not out:
            print("Planner returned no output.")
        else:
            energies = out.get("energies")
            if energies is not None:
                print("energies:", energies)
            print("initial state:", out.get("initial_state"))
            states = out.get("states")
            if states is not None:
                print("final state:", states[-1])
            rewards = out.get("rewards")
            if rewards is not None:
                print("rewards:", rewards)
                print("total reward:", float(jnp.sum(rewards)))
            if "actions" in out:
                print("actions:", out["actions"])
            if "energy_iterations" in out:
                print("energy Iter history:", out["energy_iterations"])
            if "reward_history" in out:
                print("reward history:", out["reward_history"])

    return out


# ============================================================================
# CLI (for backward compatibility)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("EDOC Planner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--np_random_seed", type=int, default=None)
    parser.add_argument("--env_name", type=str, default="double_integrator_box")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--use_state_box", action="store_true", default=True)
    parser.add_argument("--state_low", type=float, default=-2.0)
    parser.add_argument("--state_high", type=float, default=2.0)
    parser.add_argument("--action_space", action="store_true", default=True)
    parser.add_argument("--diffusion_mode", type=str, default="reverse")
    parser.add_argument("--action_diffuse_steps", type=int, default=100)
    parser.add_argument("--action_beta0", type=float, default=1e-4)
    parser.add_argument("--action_betaT", type=float, default=1e-2)
    parser.add_argument("--action_temp", type=float, default=0.1)
    parser.add_argument("--action_extra_sigma", type=float, default=0.01)
    parser.add_argument("--action_stage_ratio", type=float, default=1.0)
    parser.add_argument("--action_score_mode", type=str, default="reward")
    parser.add_argument("--action_nsample", type=int, default=256)
    parser.add_argument("--no_antithetic", action="store_true", default=False, help="Disable antithetic pairing when sampling action trajectories.")
    parser.add_argument("--dyn_loss_coeff", type=float, default=0.0)
    parser.add_argument("--dyn_loss_mode", type=str, default="terminal", choices=["terminal", "trajectory"])
    parser.add_argument("--verbose", action="store_true", default=True)

    cli_args = parser.parse_args()
    # Import EDOCArgs from configs
    from configs.double_integrator_box.edoc import EDOCArgs
    args = EDOCArgs(
        seed=cli_args.seed,
        np_random_seed=cli_args.np_random_seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        noise_std=cli_args.noise_std,
        use_state_box=cli_args.use_state_box,
        state_low=cli_args.state_low,
        state_high=cli_args.state_high,
        action_space=cli_args.action_space,
        diffusion_mode=cli_args.diffusion_mode,
        action_diffuse_steps=cli_args.action_diffuse_steps,
        action_beta0=cli_args.action_beta0,
        action_betaT=cli_args.action_betaT,
        action_temp=cli_args.action_temp,
        action_extra_sigma=cli_args.action_extra_sigma,
        action_stage_ratio=cli_args.action_stage_ratio,
        action_score_mode=cli_args.action_score_mode,
        action_nsample=cli_args.action_nsample,
        use_antithetic=not cli_args.no_antithetic,
        dyn_loss_coeff=cli_args.dyn_loss_coeff,
        dyn_loss_mode=cli_args.dyn_loss_mode,
        verbose=cli_args.verbose,
    )

    run_edoc(args)