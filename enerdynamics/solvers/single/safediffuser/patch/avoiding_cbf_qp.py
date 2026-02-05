from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

try:
    from qpth.qp import QPFunction  # type: ignore
except Exception:  # pragma: no cover
    QPFunction = None


@dataclass(frozen=True)
class AvoidingCBFConfig:
    """
    CBF-QP safety correction config (discrete-time, linearized barrier).
    """

    constraint_config_path: str
    exp: str = "avoiding-d3il"
    # Use desired position (x_des,y_des) as controlled position
    des_idx: Tuple[int, int] = (0, 1)
    # Action is delta desired: u=[dx,dy]
    act_dim: int = 2
    # CBF parameter: b_{t+1} >= (1-alpha) b_t
    alpha: float = 0.2
    # Correct only first action (t=0) by default (fast + MPC-like)
    correct_all_steps: bool = False
    # Which halfspace variant to enforce. If None, enforce all halfspace constraints.
    halfspace_variant: Optional[str] = None


def _default_constraint_config_path() -> str:
    here = Path(__file__).resolve()
    project_root = here.parents[4]  # .../enerdynamics/enerdynamics/solvers/single/safediffuser
    # Prefer SafeDiffuser-local constraint config.
    return str(project_root / "config" / "avoiding_d3il.yaml")


def _load_constraint_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return dict(cfg)


class AvoidingCBFQPCorrector:
    """
    Per-step CBF-QP correction that adjusts diffusion-proposed actions to satisfy
    obstacle-avoidance constraints.

    This is intended to run *inside* diffusion sampling (each denoise step),
    but (by default) only corrects action at t=0 to keep runtime manageable.
    """

    def __init__(
        self,
        *,
        normalizer: Any,
        config: Optional[AvoidingCBFConfig] = None,
    ):
        if QPFunction is None:
            raise ImportError("qpth is required for CBF-QP safety correction. Install `qpth`.")

        self.normalizer = normalizer
        self.cfg = config or AvoidingCBFConfig(constraint_config_path=_default_constraint_config_path())

        constraint_config = _load_constraint_config(self.cfg.constraint_config_path)
        exp = self.cfg.exp

        # Load obstacle constraints for this exp.
        obstacles_all = constraint_config.get("obstacle_constraints", {})
        if isinstance(obstacles_all, dict):
            obstacle_constraints = obstacles_all.get(exp, [])
        else:
            obstacle_constraints = obstacles_all or []

        # Keep only sphere_outside in XY.
        self.obstacles: List[Tuple[np.ndarray, float]] = []
        for oc in obstacle_constraints:
            if not isinstance(oc, dict):
                continue
            if oc.get("type") != "sphere_outside":
                continue
            center = np.asarray(oc.get("center", [0.0, 0.0]), dtype=np.float32).reshape(2)
            radius = float(oc.get("radius", 0.0))
            self.obstacles.append((center, radius))

        # Halfspace constraints for this exp.
        halfspaces_all = constraint_config.get("halfspace_constraints", {})
        if isinstance(halfspaces_all, dict):
            halfspace_constraints = halfspaces_all.get(exp, [])
        else:
            halfspace_constraints = halfspaces_all or []
        self.halfspaces_raw = list(halfspace_constraints)

        # Optional bounds on the per-step delta (dx,dy).
        # Key: `denoise_step_bounds` (SafeDiffuser-local, unambiguous).
        bounds_all = constraint_config.get("denoise_step_bounds", {})
        if isinstance(bounds_all, dict):
            bounds_spec = bounds_all.get(exp, []) or []
        else:
            bounds_spec = bounds_all or []

        delta_min = np.full((self.cfg.act_dim,), -np.inf, dtype=np.float32)
        delta_max = np.full((self.cfg.act_dim,), np.inf, dtype=np.float32)
        for b in bounds_spec:
            if not isinstance(b, dict):
                continue
            dims = b.get("dimensions", [])
            vals = b.get("values", [])
            if len(dims) != len(vals):
                continue
            for d, v in zip(dims, vals):
                if d == "vx":
                    idx = 0
                elif d == "vy":
                    idx = 1
                else:
                    continue
                if b.get("type") == "lower":
                    delta_min[idx] = max(delta_min[idx], float(v))
                elif b.get("type") == "upper":
                    delta_max[idx] = min(delta_max[idx], float(v))
        self.delta_min = torch.tensor(delta_min, dtype=torch.float32)
        self.delta_max = torch.tensor(delta_max, dtype=torch.float32)

        # Cache tensors per-device for speed
        self._cached_device: Optional[torch.device] = None
        self._obs_centers_t: Optional[torch.Tensor] = None  # (K,2)
        self._obs_radii_t: Optional[torch.Tensor] = None  # (K,)
        self._half_c_t: Optional[torch.Tensor] = None  # (Hh,2) inequality c·p <= d
        self._half_d_t: Optional[torch.Tensor] = None  # (Hh,)

    def _select_halfspaces(self) -> List:
        hs = self.halfspaces_raw
        v = self.cfg.halfspace_variant
        if v is None:
            return hs
        # Match dpcc selection convention (by index)
        if v == "top-left-hard" and len(hs) >= 1:
            return [hs[0]]
        if v == "top-right-hard" and len(hs) >= 2:
            return [hs[1]]
        if v == "both-hard" and len(hs) >= 4:
            return [hs[2], hs[3]]
        return hs

    @staticmethod
    def _halfspace_to_ineq(constraint) -> Tuple[np.ndarray, float]:
        """
        Convert constraint [[x1,y1],[x2,y2],side] into inequality c·p <= d.
        """
        p1 = np.asarray(constraint[0], dtype=np.float32)
        p2 = np.asarray(constraint[1], dtype=np.float32)
        side = str(constraint[2])
        m = float((p2[1] - p1[1]) / (p2[0] - p1[0]))
        d_line = float(p1[1] - m * p1[0])
        if side == "below":
            # y <= m x + d  => (-m)x + y <= d
            c = np.array([-m, 1.0], dtype=np.float32)
            d = d_line
        elif side == "above":
            # y >= m x + d  => (m)x - y <= -d
            c = np.array([m, -1.0], dtype=np.float32)
            d = -d_line
        else:
            raise ValueError(f"Unknown halfspace side: {side}")
        return c, d

    def _ensure_cache(self, device: torch.device) -> None:
        if self._cached_device == device:
            return
        self._cached_device = device

        if len(self.obstacles) > 0:
            centers = torch.tensor([c for c, _r in self.obstacles], dtype=torch.float32, device=device)
            radii = torch.tensor([float(r) for _c, r in self.obstacles], dtype=torch.float32, device=device)
            self._obs_centers_t = centers
            self._obs_radii_t = radii
        else:
            self._obs_centers_t = None
            self._obs_radii_t = None

        # Halfspaces
        hs_sel = self._select_halfspaces()
        if len(hs_sel) > 0:
            c_list = []
            d_list = []
            for hs in hs_sel:
                c, d = self._halfspace_to_ineq(hs)
                c_list.append(c)
                d_list.append(d)
            self._half_c_t = torch.tensor(np.stack(c_list, axis=0), dtype=torch.float32, device=device)
            self._half_d_t = torch.tensor(np.asarray(d_list, dtype=np.float32), dtype=torch.float32, device=device)
        else:
            self._half_c_t = None
            self._half_d_t = None

    def invariance(self, x: torch.Tensor, xp1: torch.Tensor) -> torch.Tensor:
        """
        SafeDiffuser-style safety hook (upstream semantics).

        This corrects the *denoising update* from `x` -> `xp1` by solving a small CBF-QP
        in terms of the update delta in desired position (x_des, y_des). It writes the
        corrected desired position back into the *new* sample `xp1` and returns it.

        Notes:
        - We intentionally do NOT overwrite the action slice here (matching upstream's
          invariance-style hooks that primarily correct state/position samples).
        - This is a per-(batch, time) independent correction; it does not enforce
          trajectory-time rollout consistency across t.
        """
        if len(self.obstacles) == 0 and len(self.halfspaces_raw) == 0:
            return xp1

        if x.shape != xp1.shape:
            return xp1

        device = xp1.device
        self._ensure_cache(device)

        B, H, _D = xp1.shape
        A = int(getattr(self.normalizer, "action_dim", self.cfg.act_dim))
        if A != self.cfg.act_dim:
            return xp1

        # Split normalized tensors (we only correct observations).
        normed_obs_prev = x[:, :, A:]
        normed_obs_next = xp1[:, :, A:]

        # Unnormalize observations to physical (torch-native).
        obs_prev_phys = self.normalizer.unnormalize_torch(normed_obs_prev, "observations")
        obs_next_phys = self.normalizer.unnormalize_torch(normed_obs_next, "observations")

        x_des_idx, y_des_idx = self.cfg.des_idx

        t_indices = list(range(H)) if self.cfg.correct_all_steps else [0]

        p_prev = obs_prev_phys[:, t_indices, :][:, :, [x_des_idx, y_des_idx]].reshape(-1, 2)  # (BT,2)
        p_next_nom = obs_next_phys[:, t_indices, :][:, :, [x_des_idx, y_des_idx]].reshape(-1, 2)  # (BT,2)
        delta0 = p_next_nom - p_prev  # (BT,2)
        BT = p_prev.shape[0]

        # Build linear inequality constraints in terms of delta:
        #   Bounds: delta_min <= delta <= delta_max
        #   Halfspace: c·(p_prev + delta) <= d
        #   Obstacle CBF: b(p_prev + delta) >= (1-alpha)b(p_prev) (linearized at p_prev)
        G_list = []
        h_list = []

        if torch.isfinite(self.delta_max).any():
            G_list.append(torch.eye(2, device=device).unsqueeze(0).repeat(BT, 1, 1))
            h_list.append(self.delta_max.to(device).unsqueeze(0).repeat(BT, 1))
        if torch.isfinite(self.delta_min).any():
            G_list.append((-torch.eye(2, device=device)).unsqueeze(0).repeat(BT, 1, 1))
            h_list.append((-self.delta_min.to(device)).unsqueeze(0).repeat(BT, 1))

        if self._half_c_t is not None and self._half_d_t is not None:
            c = self._half_c_t  # (Hh,2)
            d = self._half_d_t  # (Hh,)
            G_half = c.unsqueeze(0).repeat(BT, 1, 1)  # (BT,Hh,2)
            h_half = d.unsqueeze(0).repeat(BT, 1) - (p_prev @ c.T)  # (BT,Hh)
            G_list.append(G_half)
            h_list.append(h_half)

        alpha = float(self.cfg.alpha)
        if self._obs_centers_t is not None and self._obs_radii_t is not None:
            centers = self._obs_centers_t  # (K,2)
            radii = self._obs_radii_t  # (K,)
            dvec = p_prev.unsqueeze(1) - centers.unsqueeze(0)  # (BT,K,2)
            b_val = (dvec[..., 0] ** 2 + dvec[..., 1] ** 2) - (radii.unsqueeze(0) ** 2)  # (BT,K)
            grad = 2.0 * dvec  # (BT,K,2)
            G_obs = -grad  # (BT,K,2)
            h_obs = alpha * b_val  # (BT,K)
            G_list.append(G_obs)
            h_list.append(h_obs)

        if not G_list:
            return xp1

        G = torch.cat(G_list, dim=1)
        h = torch.cat(h_list, dim=1)

        # QP: min 0.5||delta - delta0||^2
        Q = 2.0 * torch.eye(2, device=device).unsqueeze(0).repeat(BT, 1, 1)
        p = (-2.0 * delta0).to(device)

        try:
            delta_star = QPFunction(verbose=False)(Q, p, G, h, None, None)
        except Exception:
            delta_star = delta0

        obs_next_phys_new = obs_next_phys.clone()
        p_next_corr = (p_prev + delta_star).reshape(B, len(t_indices), 2)
        obs_next_phys_new[:, t_indices, x_des_idx] = p_next_corr[:, :, 0]
        obs_next_phys_new[:, t_indices, y_des_idx] = p_next_corr[:, :, 1]

        # Renormalize corrected observations back into xp1.
        normed_obs_next_new = self.normalizer.normalize_torch(obs_next_phys_new, "observations")
        xp1_out = xp1.clone()
        xp1_out[:, :, A:] = normed_obs_next_new
        return xp1_out

