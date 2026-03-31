"""
Gate dynamics visualization plugin.

Four-panel view of the flow/diffusion gating behaviour over diffusion steps:
1. gamma (binary gate) vs diffusion time
2. sigma_eff (effective noise) vs diffusion time
3. pi_multi (multimodality proxy) vs theta (threshold)
4. v_rate and cvar vs diffusion time

X-axis convention: diffusion time runs right-to-left (K→0) matching the
``invert_xaxis_labels`` convention used by ``scheduler_params.py``.
Data index 0 = step_k 0 (early, high noise); index K-1 = step_k K-1 (late).
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from ...framework.base import VisualizationPlugin
from ...common.visualization import EDOC_COLOR


def _invert_xaxis(ax, K: int):
    """Label x-axis as diffusion time K→0 (high noise → converged)."""
    ticks = ax.get_xticks()
    labels = []
    for t in ticks:
        if 0 <= t <= K:
            labels.append(f"{int(K - t)}")
        else:
            labels.append("")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


class GateDynamicsVisualizationPlugin(VisualizationPlugin):
    """Visualizes the flow/diffusion gate switching behaviour."""

    @property
    def name(self) -> str:
        return "gate_dynamics"

    def visualize(
        self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        result = data["result"]

        gamma = np.asarray(result.get("gamma_hist", []), dtype=np.float32).ravel()
        sigma = np.asarray(result.get("sigma_hist", []), dtype=np.float32).ravel()
        theta = np.asarray(result.get("theta_hist", []), dtype=np.float32).ravel()
        pi_multi = np.asarray(result.get("pi_multi_hist", []), dtype=np.float32).ravel()
        v_rate = np.asarray(result.get("v_rate_hist", []), dtype=np.float32).ravel()
        cvar = np.asarray(result.get("cvar_hist", []), dtype=np.float32).ravel()

        K = gamma.size
        if K == 0:
            for ax in np.array(axes).flat:
                ax.text(0.5, 0.5, "No gate data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            return

        steps = np.arange(K)
        ax_flat = np.array(axes).flat

        # --- Panel 1: gamma (gate on/off) --------------------------------
        ax = next(ax_flat, None)
        if ax is not None:
            ax.fill_between(
                steps, 0, gamma,
                color=EDOC_COLOR, alpha=0.35, step="mid", label="diffusion ON",
            )
            ax.plot(steps, gamma, color=EDOC_COLOR, linewidth=1.0)
            ax.set_ylabel(r"$\gamma$")
            ax.set_xlabel("Diffusion Time")
            ax.set_title("Flow / Diffusion Gate")
            ax.set_ylim(-0.05, 1.15)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            _invert_xaxis(ax, K)

        # --- Panel 2: sigma_eff -------------------------------------------
        ax = next(ax_flat, None)
        if ax is not None:
            ax.plot(steps, sigma, color=EDOC_COLOR, linewidth=2.0, label=r"$\sigma_{\mathrm{eff}}$")
            ax.set_ylabel(r"$\sigma_{\mathrm{eff}}$")
            ax.set_xlabel("Diffusion Time")
            ax.set_title("Effective Noise")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            _invert_xaxis(ax, K)

        # --- Panel 3: pi_multi vs theta -----------------------------------
        ax = next(ax_flat, None)
        if ax is not None:
            if pi_multi.size == K:
                ax.plot(steps, pi_multi, color=EDOC_COLOR, linewidth=2.0,
                        label=r"$\Pi_{\mathrm{route}}$")
            if theta.size == K:
                ax.plot(steps, theta, color="red", linewidth=1.5, linestyle="--",
                        label=r"$\theta_{\mathrm{route}}(t)$")
            ax.set_ylabel("Proxy / Threshold")
            ax.set_xlabel("Diffusion Time")
            ax.set_title(r"Route Ambiguity Proxy vs $\theta$")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            _invert_xaxis(ax, K)

        # --- Panel 4: v_rate and cvar -------------------------------------
        ax = next(ax_flat, None)
        if ax is not None:
            if v_rate.size == K:
                ax.plot(steps, v_rate, color="steelblue", linewidth=2.0,
                        label="v_rate")
            if cvar.size == K:
                ax_twin = ax.twinx()
                ax_twin.plot(steps, cvar, color="tomato", linewidth=1.5,
                             linestyle="--", label="CVaR")
                ax_twin.set_ylabel("CVaR", color="tomato")
                ax_twin.tick_params(axis="y", labelcolor="tomato")
                lines_a, labels_a = ax.get_legend_handles_labels()
                lines_b, labels_b = ax_twin.get_legend_handles_labels()
                ax.legend(lines_a + lines_b, labels_a + labels_b,
                          loc="upper right", fontsize=8)
            else:
                ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("Violation Rate", color="steelblue")
            ax.set_xlabel("Diffusion Time")
            ax.set_title("Feasibility (v_rate & CVaR)")
            ax.tick_params(axis="y", labelcolor="steelblue")
            ax.grid(True, alpha=0.3)
            _invert_xaxis(ax, K)

        # Hide unused axes.
        for ax in ax_flat:
            ax.set_visible(False)

    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        dpi = kwargs.get("dpi", 150)
        bbox_inches = kwargs.get("bbox_inches", "tight")
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
