"""
Energy and reward visualization plugin.
"""

from typing import Dict, Any
import numpy as np

from ...framework.base import VisualizationPlugin
from ...common.visualization import EDOC_COLOR


class EnergyRewardVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing energy and reward over time.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "energy_reward"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create energy and reward plots.
        
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
        
        # Get energy and reward data
        energies = result.get('energies', [])
        rewards = result.get('rewards', [])
        reward_history = result.get('reward_history', None)
        
        dt = getattr(env, 'dt', 0.1)
        
        # Convert to arrays if needed
        if isinstance(energies, (list, tuple)):
            energies = np.array(energies, dtype=np.float32)
        if isinstance(rewards, (list, tuple)):
            rewards = np.array(rewards, dtype=np.float32)
        
        # Plot energy
        if len(energies) > 0:
            time = np.arange(len(energies)) * dt
            ax_energy.plot(time, energies, color=EDOC_COLOR, linewidth=2.0, label='Energy')
            ax_energy.set_xlabel('Time (s)')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Over Time')
            ax_energy.legend()
            ax_energy.grid(True, alpha=0.3)
        
        # Plot reward
        if len(rewards) > 0:
            time_rewards = np.arange(len(rewards)) * dt
            ax_reward.plot(time_rewards, rewards, color=EDOC_COLOR, linewidth=2.0, label='Reward')
            
            # Plot diffusion reward history if available
            if reward_history is not None and len(reward_history) > 0:
                if hasattr(reward_history, 'tolist'):
                    reward_hist = np.asarray(reward_history, dtype=np.float32)
                else:
                    reward_hist = np.array(reward_history, dtype=np.float32)
                
                if len(reward_hist) > 0 and len(time_rewards) > 0:
                    diffusion_steps = np.linspace(0, time_rewards[-1], len(reward_hist))
                    ax_reward.plot(
                        diffusion_steps, reward_hist, '--',
                        color=EDOC_COLOR, linewidth=1, alpha=0.5,
                        label='Diffusion Reward'
                    )
            
            ax_reward.set_xlabel('Time (s)')
            ax_reward.set_ylabel('Reward')
            ax_reward.set_title('Reward Over Time')
            ax_reward.legend()
            ax_reward.grid(True, alpha=0.3)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

