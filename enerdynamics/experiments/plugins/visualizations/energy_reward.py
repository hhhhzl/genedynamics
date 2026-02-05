"""
Energy and reward visualization plugin.
"""

from typing import Dict, Any
import numpy as np

from ...framework.base import VisualizationPlugin
from ...common.visualization import EDOC_COLOR


def invert_xaxis_labels(ax, n_steps: int, one_based: bool = True):
    """
    Set x-axis labels so that diffusion step runs from n_steps down to 1 (or 0).
    Position 0 = noisiest (step n_steps), position n_steps-1 = cleanest (step 1).

    Args:
        ax: Matplotlib axis
        n_steps: Number of diffusion steps (e.g. 100)
        one_based: If True, label cleanest as 1; if False, as 0.
    """
    ticks = ax.get_xticks()
    # Label at position i -> diffusion step (n_steps - i) when one_based, so 100..1
    # When one_based and n_steps=100: i=0 -> 100, i=99 -> 1
    labels = []
    for tick in ticks:
        if 0 <= tick < n_steps:
            lab = int(n_steps - tick) if one_based else int(n_steps - 1 - tick)
            if one_based and lab < 1:
                lab = 1
            labels.append(f"{lab}")
        else:
            labels.append("")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


class EnergyRewardVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing energy and reward over diffusion steps.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "energy_reward"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create energy and reward plots over diffusion steps.
        
        Args:
            fig: Matplotlib figure
            axes: Tuple of (ax_energy, ax_reward) axes
            data: Dictionary containing:
                - result: Planning result dictionary
                - env: Environment instance
            config: Visualization configuration dictionary
        """
        result = data['result']
        env = data['env']
        
        ax_energy, ax_reward = axes
        
        # Get diffusion-related data
        reward_history = result.get('reward_history', None)
        diffusion_actions = result.get('diffusion_actions_traj', None)
        
        # Debug: Check if diffusion_actions_traj exists
        # If it's an empty array, it might have been initialized but not filled
        if diffusion_actions is not None:
            try:
                # Try to convert to numpy array to check shape
                test_arr = np.asarray(diffusion_actions)
                if hasattr(test_arr, 'shape') and len(test_arr.shape) > 0:
                    if test_arr.size == 0:
                        # Empty array - diffusion_actions_traj was not filled
                        diffusion_actions = None
            except (ValueError, TypeError):
                pass
        
        # Get final trajectory data as fallback
        energies = result.get('energies', [])
        rewards = result.get('rewards', [])
        
        # Convert to arrays if needed
        if reward_history is not None:
            if isinstance(reward_history, (list, tuple)):
                reward_history = np.array(reward_history, dtype=np.float32)
            elif hasattr(reward_history, '__len__') and len(reward_history) == 0:
                reward_history = None
        if isinstance(energies, (list, tuple)):
            energies = np.array(energies, dtype=np.float32)
        if isinstance(rewards, (list, tuple)):
            rewards = np.array(rewards, dtype=np.float32)
        
        # Plot reward over diffusion steps
        if reward_history is not None and len(reward_history) > 0:
            # Reverse so that left = noisiest (step 100), right = cleanest (step 1); reward increases left->right
            reward_plot = np.asarray(reward_history)[::-1]
            n_steps = len(reward_history)
            diffusion_steps = np.arange(n_steps)
            ax_reward.plot(diffusion_steps, reward_plot, color=EDOC_COLOR, linewidth=2.0, label='Reward')
            ax_reward.set_xlabel('Diffusion Step')
            ax_reward.set_ylabel('Reward')
            ax_reward.set_title('Reward Over Diffusion Steps')
            invert_xaxis_labels(ax_reward, n_steps, one_based=True)  # Labels 100 down to 1
            ax_reward.legend()
            ax_reward.grid(True, alpha=0.3)
        elif len(rewards) > 0:
            # Fallback: plot final trajectory rewards over time
            dt = getattr(env, 'dt', 0.1)
            time_rewards = np.arange(len(rewards)) * dt
            ax_reward.plot(time_rewards, rewards, color=EDOC_COLOR, linewidth=2.0, label='Reward')
            ax_reward.set_xlabel('Time (s)')
            ax_reward.set_ylabel('Reward')
            ax_reward.set_title('Reward Over Time (Final Trajectory)')
            ax_reward.legend()
            ax_reward.grid(True, alpha=0.3)
        
        # Plot energy over diffusion steps
        # Compute energy for each diffusion step if diffusion_actions is available
        energy_plotted = False
        
        # Check if diffusion_actions_traj exists and is not empty
        # It might be an empty array, so check length after conversion
        if diffusion_actions is not None:
            try:
                if isinstance(diffusion_actions, (list, tuple)):
                    diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
                elif hasattr(diffusion_actions, '__len__'):
                    # Handle JAX arrays and numpy arrays
                    if hasattr(diffusion_actions, 'shape'):
                        diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
                    else:
                        diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
                else:
                    diffusion_actions = None
            except (ValueError, TypeError):
                diffusion_actions = None
        
        # Try to compute energy over diffusion steps if we have diffusion_actions
        if diffusion_actions is not None and hasattr(diffusion_actions, '__len__') and len(diffusion_actions) > 0:
            initial_state = result.get('initial_state', None)
            if initial_state is not None:
                # Try to get energy functional from config
                try:
                    from enerdynamics.envs.factories import make_energy
                    # Get env_name from config - handle both dict and object access
                    config_obj = config.get('config', None)
                    if config_obj is not None:
                        if hasattr(config_obj, 'env_name'):
                            env_name = config_obj.env_name
                        elif isinstance(config_obj, dict):
                            env_name = config_obj.get('env_name', None)
                        else:
                            env_name = None
                    else:
                        env_name = None
                    
                    if env_name:
                        energy_func = make_energy(env_name)
                        # Compute energy for each diffusion step's trajectory
                        energy_history = []
                        for step_idx in range(len(diffusion_actions)):
                            action_seq = diffusion_actions[step_idx]
                            states = env.rollout_actions(initial_state, action_seq)
                            # Compute total energy (sum over trajectory)
                            step_energies = []
                            for state in states:
                                # Use energy functional to compute energy
                                try:
                                    # Use zero action and empty context for state-only energy
                                    zero_action = np.zeros(env.act_dim, dtype=np.float32)
                                    ctx = {}
                                    energy_val = float(energy_func.compute(state, zero_action, ctx))
                                except (ImportError, TypeError, AttributeError, ValueError) as e:
                                    # Fallback: use env cost if available
                                    if hasattr(env, 'cost'):
                                        energy_val = env.cost(state)
                                    else:
                                        energy_val = 0.0
                                step_energies.append(energy_val)
                            total_energy = np.sum(step_energies) if len(step_energies) > 0 else 0.0
                            energy_history.append(total_energy)
                        
                        if len(energy_history) > 0:
                            # Reverse so left = noisiest (step 100), right = cleanest (step 1)
                            energy_plot = np.asarray(energy_history)[::-1]
                            n_steps_energy = len(energy_history)
                            diffusion_steps_energy = np.arange(n_steps_energy)
                            ax_energy.plot(diffusion_steps_energy, energy_plot, color=EDOC_COLOR, linewidth=2.0, label='Energy')
                            ax_energy.set_xlabel('Diffusion Step')
                            ax_energy.set_ylabel('Energy')
                            ax_energy.set_title('Energy Over Diffusion Steps')
                            invert_xaxis_labels(ax_energy, n_steps_energy, one_based=True)  # Labels 100 down to 1
                            ax_energy.legend()
                            ax_energy.grid(True, alpha=0.3)
                            energy_plotted = True
                except (ImportError, AttributeError, ValueError, Exception) as e:
                    pass
        
        # Fallback: plot final trajectory energy over time
        if len(energies) > 0 and not energy_plotted:
            dt = getattr(env, 'dt', 0.1)
            time = np.arange(len(energies)) * dt
            ax_energy.plot(time, energies, color=EDOC_COLOR, linewidth=2.0, label='Energy')
            ax_energy.set_xlabel('Time (s)')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Over Time (Final Trajectory)')
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

