"""
Scheduler parameters visualization plugin.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ...framework.base import VisualizationPlugin
from ...common.visualization import EDOC_COLOR


def invert_xaxis_labels(ax, max_step: int):
    """
    Invert x-axis labels to show 100->0 without reversing data.
    
    Args:
        ax: Matplotlib axis
        max_step: Maximum step value (e.g., 100 for 0-100 range)
    """
    # Get current ticks
    ticks = ax.get_xticks()
    
    # Create inverted labels: step i -> label (max_step - i)
    # Only create labels for ticks within valid range [0, max_step]
    labels = []
    for tick in ticks:
        if 0 <= tick <= max_step:
            labels.append(f"{int(max_step - tick)}")
        else:
            labels.append("")
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


class SchedulerParamsVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing scheduler parameters over diffusion steps.
    
    Shows how constraint scheduler parameters (lambda_con, rho, gate probability,
    topK, eps, etc.) evolve during the diffusion process.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "scheduler_params"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create scheduler parameters plots over diffusion steps.
        
        Args:
            fig: Matplotlib figure
            axes: Array of matplotlib axes (will create 2x2 subplots)
            data: Dictionary containing:
                - result: Planning result dictionary
                - env: Environment instance
            config: Visualization configuration dictionary
        """
        result = data['result']
        
        # Get scheduler parameter history
        scheduler_params_history = result.get('scheduler_params_history', [])
        
        if not scheduler_params_history or len(scheduler_params_history) == 0:
            # No data available - show empty plots with message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No scheduler parameter history available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            return
        
        # Extract data from history
        steps = [p.get('step', i) for i, p in enumerate(scheduler_params_history)]
        lambda_con = [p.get('lambda_con', 0.0) for p in scheduler_params_history]
        rho = [p.get('rho', 0.0) for p in scheduler_params_history]
        qp_prob = [p.get('qp_prob', 0.0) for p in scheduler_params_history]
        qp_gate = [p.get('qp_gate', False) for p in scheduler_params_history]
        topK = [p.get('topK', 0) for p in scheduler_params_history if p.get('topK') is not None]
        eps = [p.get('eps', 0.0) for p in scheduler_params_history]
        I_QP = [p.get('I_QP', 0) for p in scheduler_params_history]
        q_star = [p.get('q_star', 0.0) for p in scheduler_params_history]
        # Extract q_hat from history (check both top-level and _extra)
        q_hat = []
        for p in scheduler_params_history:
            # First try top-level q_hat
            q_hat_val = p.get('q_hat', None)
            # If not found, try _extra dict
            if q_hat_val is None:
                extra = p.get('_extra', {})
                if isinstance(extra, dict):
                    q_hat_val = extra.get('q_hat', None)
                elif hasattr(extra, 'q_hat'):
                    q_hat_val = getattr(extra, 'q_hat', None)
            # Default to 1.0 if not available
            q_hat.append(q_hat_val if q_hat_val is not None else 1.0)
        
        # Convert to numpy arrays
        steps = np.array(steps)
        lambda_con = np.array(lambda_con)
        rho = np.array(rho)
        qp_prob = np.array(qp_prob)
        topK = np.array(topK) if topK else np.array([])
        eps = np.array(eps)
        I_QP = np.array(I_QP)
        q_star = np.array(q_star)
        q_hat = np.array(q_hat)
        
        # Flatten axes if needed
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)
        ax_flat = axes.flat
        
        # Get max step for label inversion
        max_step = int(steps.max()) if len(steps) > 0 else 100
        
        # Plot 1: Lambda (dual variable) over diffusion steps
        ax = next(ax_flat, None)
        if ax is not None:
            ax.plot(steps, lambda_con, color=EDOC_COLOR, linewidth=2.0, label='λ^con')
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Lambda (dual variable)')
            ax.set_title('Dual Variable Over Diffusion Steps')
            invert_xaxis_labels(ax, max_step)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Rho (slack penalty) over diffusion steps
        ax = next(ax_flat, None)
        if ax is not None:
            ax.plot(steps, rho, color=EDOC_COLOR, linewidth=2.0, label='ρ_k')
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Slack Penalty')
            ax.set_title('Slack Penalty Over Diffusion Steps')
            invert_xaxis_labels(ax, max_step)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Gate probability over diffusion steps
        ax = next(ax_flat, None)
        if ax is not None:
            ax.plot(steps, qp_prob, color=EDOC_COLOR, linewidth=2.0, label='p_k (gate prob)')
            # Also show actual gate decisions as scatter points
            gate_steps = steps[qp_gate] if len(qp_gate) == len(steps) else steps
            gate_mask = np.array(qp_gate) if len(qp_gate) == len(steps) else np.array([])
            if len(gate_mask) > 0:
                ax.scatter(steps[gate_mask], qp_prob[gate_mask], 
                          color='green', marker='o', s=20, alpha=0.5, label='Gate ON')
                ax.scatter(steps[~gate_mask], qp_prob[~gate_mask],
                          color='red', marker='x', s=20, alpha=0.5, label='Gate OFF')
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Gate Probability')
            ax.set_title('QP Gate Probability Over Diffusion Steps')
            ax.set_ylim(-0.05, 1.05)
            invert_xaxis_labels(ax, max_step)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: TopK and eps over diffusion steps (dual y-axis)
        ax = next(ax_flat, None)
        if ax is not None:
            if len(topK) > 0 and len(topK) == len(steps):
                ax_twin = ax.twinx()
                line1 = ax.plot(steps, topK, color='blue', linewidth=2.0, label='topK')
                line2 = ax_twin.plot(steps, eps, color='red', linewidth=2.0, label='eps')
                ax.set_xlabel('Diffusion Step')
                ax.set_ylabel('topK (Active Constraints)', color='blue')
                ax_twin.set_ylabel('eps (Solver Tolerance)', color='red')
                ax.tick_params(axis='y', labelcolor='blue')
                ax_twin.tick_params(axis='y', labelcolor='red')
                ax.set_title('Active Constraints & Tolerance')
                invert_xaxis_labels(ax, max_step)
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left')
                ax.grid(True, alpha=0.3)
            else:
                # Only show eps if topK not available
                ax.plot(steps, eps, color=EDOC_COLOR, linewidth=2.0, label='eps')
                ax.set_xlabel('Diffusion Step')
                ax.set_ylabel('eps (Solver Tolerance)')
                ax.set_title('Solver Tolerance Over Diffusion Steps')
                invert_xaxis_labels(ax, max_step)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot 5: I_QP (QP iterations) over diffusion steps
        ax = next(ax_flat, None)
        if ax is not None:
            ax.plot(steps, I_QP, color=EDOC_COLOR, linewidth=2.0, label='I_QP', marker='o', markersize=3)
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('QP Iterations')
            ax.set_title('QP Iterations Over Diffusion Steps')
            invert_xaxis_labels(ax, max_step)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: q_tilde, q_hat and q_star over diffusion steps (dual y-axis)
        ax = next(ax_flat, None)
        if ax is not None:
            # Extract q_tilde from history
            q_tilde = []
            for p in scheduler_params_history:
                q_tilde_val = p.get('q_tilde', None)
                if q_tilde_val is None:
                    extra = p.get('_extra', {})
                    if isinstance(extra, dict):
                        q_tilde_val = extra.get('q_tilde', None)
                q_tilde.append(q_tilde_val if q_tilde_val is not None else p.get('q_hat', 1.0))
            q_tilde = np.array(q_tilde)
            
            ax_twin = ax.twinx()
            # Plot q_tilde (synthesized) as main line, q_hat as dashed, q_star as target
            line1 = ax.plot(steps, q_tilde, color='green', linewidth=2.5, label='q̃ (synthesized)', linestyle='-')
            line2 = ax.plot(steps, q_hat, color='blue', linewidth=1.5, label='q̂ (actual)', linestyle='--', alpha=0.7)
            line3 = ax_twin.plot(steps, q_star, color='red', linewidth=2.0, label='q* (target)', linestyle='-')
            ax.set_xlabel('Diffusion Step')
            ax.set_ylabel('Feasibility (q̃, q̂)', color='black')
            ax_twin.set_ylabel('Target Feasibility q*', color='red')
            ax.tick_params(axis='y', labelcolor='black')
            ax_twin.tick_params(axis='y', labelcolor='red')
            ax.set_title('Synthesized vs Target Feasibility')
            ax.set_ylim(0.0, 1.05)  # q_tilde and q_hat are in [0, 1]
            ax_twin.set_ylim(0.0, 1.05)  # q_star is in [0, 1]
            invert_xaxis_labels(ax, max_step)
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide remaining unused axes
        for ax in ax_flat:
            ax.set_visible(False)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

