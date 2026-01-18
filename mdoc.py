import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import einops
from torch_robotics.robots.robot_base import RobotState
from mdoc.utils.mdoc_constraints import *

import torch._dynamo

torch._dynamo.config.suppress_errors = True
# increase this if on a stronger CPU, this will allow a large graph (N_max, H) for ConstrainTensor
# so we can increase constraints_to_check and horizon
torch.set_num_threads(32)
torch.set_num_interop_threads(1)

from dataclasses import dataclass


@dataclass
class ConstraintTensor:
    C2: torch.Tensor  # (M_max, N_max, 2), float32, contiguous
    R: torch.Tensor  # (M_max, N_max),    float32, contiguous
    TM: torch.Tensor  # (M_max, N_max, H), bool,     contiguous


def pack_constraints_dict(constraints: dict[int, list],
                          *, M_max: int, N_max: int, D: int, H: int,
                          device: torch.device) -> ConstraintTensor:
    C2 = torch.zeros(M_max, N_max, 2, dtype=torch.float32, device=device)
    R = torch.zeros(M_max, N_max, dtype=torch.float32, device=device)
    TM = torch.zeros(M_max, N_max, H, dtype=torch.bool, device=device)

    for mid, sph_list in constraints.items():
        if not (0 <= mid < M_max):
            continue
        n_filled = 0
        for sph in sph_list:
            if n_filled >= N_max:
                break
            if not sph:
                continue
            qs = sph.qs.to(device)  # (n_i, D)
            rad = sph.radii.to(device)  # (n_i,)
            tr = sph.traj_ranges.to(device).long()  # (n_i, 2)

            ni = min(qs.shape[0], N_max - n_filled)
            if ni <= 0:
                continue

            C2[mid, n_filled:n_filled + ni, :] = qs[:ni, :2].to(torch.float32)
            R[mid, n_filled:n_filled + ni] = rad[:ni].to(torch.float32)
            t_idx = torch.arange(H, device=device).view(1, H)
            starts = tr[:ni, 0:1]
            ends = tr[:ni, 1:2]
            TM[mid, n_filled:n_filled + ni, :] = (t_idx >= starts) & (t_idx < ends)

            n_filled += ni

    return ConstraintTensor(
        C2=C2.contiguous(),
        R=R.contiguous(),
        TM=TM.contiguous(),
    )


class ModelBasedDiffusionEnsemble(nn.Module):
    """
    Unlike Model-free diffusion,
    model-based diffusion interact with model information,
    and analytically calculate score function through Monte Carlo Approximation
    """

    def __init__(
            self,
            robot,
            env_models: Dict[int, Any],
            transforms: Dict[int, torch.Tensor],
            start_state_pos: torch.tensor,
            goal_state_pos: torch.tensor,
            seed: int = 0,
            enable_demo: bool = False,
            context_model=None,
            disable_recommended_params: bool = True,
            device: str = 'cpu',
            compile: bool = False,
            # compile with torch compile
            use_cuda_graph: bool = False,
            **kwargs
    ):
        """

        Args:
            env_models: model_environments {model_id: env_model}
            env_name: model_env name
            seed: random seed
            enable_demo: whether to use demonstration
            disable_recommended_params: whether to use recommended parames
            device: device
        """
        super().__init__(**kwargs)
        self.device = device
        self.robot = robot
        self.env_models = env_models
        self.enable_demo = enable_demo
        self.disable_recommended_params = disable_recommended_params

        envs_name = self.env_models[0].name
        if "empty" in envs_name.lower():
            from mdoc.config.empty.mdoc_params import MDOCParams as mparams
        elif 'conveyor' in envs_name.lower():
            from mdoc.config.conveyor.mdoc_params import MDOCParams as mparams
        elif 'random' in envs_name.lower():
            from mdoc.config.random.mdoc_params import MDOCParams as mparams
        elif 'drop' in envs_name.lower():
            from mdoc.config.dropregion.mdoc_params import MDOCParams as mparams
        else:
            # general
            from mdoc.config.mdoc_params import MDOCParams as mparams

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.robot_radius = mparams.robot_planar_disk_radius
        self.diffusion_params = {
            'temp_sample': mparams.temp_sample,
            'n_samples': mparams.n_samples,
            'n_diffusion_step': mparams.n_diffusion_steps,
            'horizon': mparams.horizon,
            'beta0': mparams.beta0,
            'betaT': mparams.betaT
        }
        self.H = mparams.horizon

        self.recommended_params = {
            "temp_sample": 0.5,
            "n_diffusion_step": 100,
            'n_samples': 64,
            "horizon": 64
        }

        self.cbf_tau = getattr(mparams, "cbf_tau", 0.05)
        self.cbf_eta = getattr(mparams, "cbf_eta", 0.8)
        self.k_best = getattr(mparams, "k_best", 2)
        self.base_beta = getattr(mparams, "base_beta", 0.05)
        # Cost Function
        self.cost_control = getattr(mparams, "cost_control", 1)
        self.cost_distance_to_goal = getattr(mparams, "cost_distance_to_goal", 5)
        self.cost_time_smoothness = getattr(mparams, "cost_time_smoothness", 1)
        self.cost_acc_smoothness = getattr(mparams, "cost_acc_smoothness", 1)
        self.cost_get_to_goal_early = getattr(mparams, "cost_get_to_goal_early", 0.5)
        self.cost_sdf_collison = getattr(mparams, "cost_sdf_collison", 5000)
        self.cost_terminal = getattr(mparams, "cost_terminal", 8)
        self.projection_score_weight = getattr(mparams, "projection_score_weight", 0.8)
        self.constraints_to_checks = getattr(mparams, "constraints_to_checks", 30)
        self.cbf_margin = getattr(mparams, "cbf_margin", 0.01)
        dt = getattr(mparams, "robot_dt", None)
        if dt:
            self.robot.dt = dt

        # for double integrator
        self.cbf_k0, self.cbf_k1 = 1, 4

        # Initialization
        self.models = {}
        self.soft_constraints = {}
        self.start_state_pos = start_state_pos
        self.goal_state_pos = goal_state_pos
        for model_id, env_model in env_models.items():
            self._setup_model_environment(model_id, env_model)
            self.soft_constraints[model_id] = []
        self.transforms = transforms
        self.context_model = context_model

        self._env_bounds = {}
        for model_id, env_model in env_models.items():
            try:
                if hasattr(env_model, "limits"):
                    limits = env_model.limits
                    # Extract as Python floats BEFORE compilation
                    self._env_bounds[model_id] = (
                        float(limits[0, 0].item()),
                        float(limits[1, 0].item()),
                        float(limits[0, 1].item()),
                        float(limits[1, 1].item())
                    )
                elif hasattr(env_model, "bounds"):
                    bounds = env_model.bounds
                    # Handle tuple/list unpacking BEFORE compilation
                    if isinstance(bounds, (tuple, list)) and len(bounds) >= 4:
                        self._env_bounds[model_id] = (
                            float(bounds[0]),
                            float(bounds[1]),
                            float(bounds[2]),
                            float(bounds[3])
                        )
                    else:
                        self._env_bounds[model_id] = (-2.0, 2.0, -2.0, 2.0)
                else:
                    self._env_bounds[model_id] = (-2.0, 2.0, -2.0, 2.0)
            except Exception:
                self._env_bounds[model_id] = (-2.0, 2.0, -2.0, 2.0)

        # for cache (ECBS calculate collision cost)
        self._centers = None
        self._radii2 = None
        self._time_mask = None

        # for non empty environment
        sdf_cell_size = 0.05
        self.sdf_cell_size = float(sdf_cell_size)
        self._sdf_tex = None
        self._sdf_tex_meta = None

        self._sdf_tex_invW = None
        self._sdf_tex_invH = None
        self._sdf_tex_xbias = None
        self._sdf_tex_ybias = None
        self._sdf_tex_grid = None

        self.compile = compile
        if getattr(mparams, "compile", None) != None:
            self.compile = mparams.compile
        if getattr(mparams, "use_cuda_graph", None) != None:
            use_cuda_graph = mparams.use_cuda_graph

        if self.device == 'mps':
            self.compile = False
        if self.compile:
            print(">>>>>> Preheat Torch Compile && Cuda Graph (if cuda) >>>>>>>>")
            self._preheat(use_cuda_graph)

    def _prepare_soft_constraints(
            self,
            model_id: int,
            H: int,
            device: torch.device
    ):
        """
        Cache for diffusion step：
            _centers    [1,1,N,q_dim]
            _radii2     [1,1,N]
            _time_mask  [1,H,N,1]
        """
        qs_l, radii_l, t0_l, t1_l, w_l = [], [], [], [], []
        for c in self.soft_constraints[model_id]:
            qs_l.append(c.qs.to(device))  # [K,q_dim]
            radii_l.append(torch.as_tensor(c.radii, device=device))
            tr = torch.as_tensor(c.traj_ranges, device=device)  # [K,2]
            t0_l.append(tr[:, 0])
            t1_l.append(tr[:, 1])
            if isinstance(c.priority_weight, torch.Tensor):
                w_l.append(c.priority_weight.to(device))
            else:
                # w_l.append(torch.full((len(c.qs),), c.priority_weight, device=device))
                w_l.append(1)

        centers = torch.cat(qs_l, dim=0)  # [N,q_dim]
        radii = torch.cat(radii_l, dim=0)  # [N]
        self._t0 = torch.cat(t0_l, dim=0)  # [N]
        self._t1 = torch.cat(t1_l, dim=0)  # [N]
        self._N = centers.shape[0]
        self._centers = centers.view(1, 1, -1, self.robot.q_dim)  # [1,1,N,q_dim]
        self._radii = radii.view(1, 1, self._N)  # [1,1,N]
        self._centers_flat = centers  # [N,q_dim]  (contiguous)
        self._radii_flat = radii  # [N]
        # self._weights = torch.cat(w_l).view(1, 1, -1, 1)  # (1,1,N,1)

    def _setup_model_environment(self, model_id: int, env_model: Any):
        params = self.diffusion_params.copy()
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "n_diffusion_step", "n_samples", "horizon"]:
                if param_name in self.recommended_params:
                    params[param_name] = self.recommended_params[param_name]

        setattr(self, f'model_{model_id}_params', params)
        self.state_inits = RobotState(
            q=self.start_state_pos,
            q_dot=torch.Tensor([0, 0]),
            cost=0,
            collision_cost=0,
            control_cost=0,
            pipeline_state=torch.cat([self.start_state_pos, self.start_state_pos])
        )
        self._setup_diffusion_params(model_id, params)

    def _setup_diffusion_params(self, model_id: int, params: dict):
        """set parameter for each model"""
        betas = torch.linspace(params['beta0'], params['betaT'], params['n_diffusion_step'], device=self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt(1 - alphas_bar)

        # Conditional parameters
        shifted_alphas_bar = torch.roll(alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0
        sigmas_cond = ((1 - alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - alphas_bar))
        sigmas_cond = torch.sqrt(sigmas_cond)
        sigmas_cond[0] = 0.0

        self.models[model_id] = {
            'betas': betas,
            'alphas': alphas,
            'alphas_bar': alphas_bar,
            'sigmas': sigmas,
            'sigmas_cond': sigmas_cond,
            'params': params
        }

    def _preheat(self, use_cuda_graph):
        model_params = self.models[0]
        H = self.models[0]['params']['horizon']
        B = model_params['params']['n_samples']
        self._acc_actions = torch.zeros(B, H, 2, device=self.device)
        M_max, N_max, D = 1, self.constraints_to_checks, 2

        constraints = {0: []}
        constraints_t = pack_constraints_dict(
            constraints,
            M_max=M_max, N_max=N_max, D=D, H=H,
            device=self.device
        )

        def _fn(actions: torch.Tensor, constraints):
            return self._rollout_single_batch_new(
                model_id=0,
                state_q=self.state_inits.q,
                us=actions,
                constraints=constraints
            )

        from mdoc.utils.accelerators import RolloutAccelerator
        use_amp = True if self.device in ("cuda", "mps") else False
        amp_dtype = torch.bfloat16 if self.device == "cuda" else torch.float16

        self._acc_rollout = RolloutAccelerator(
            fn=_fn,
            example_args=(self._acc_actions, constraints_t,),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            cuda_graph=(self.device == "cuda" and use_cuda_graph),
            compile_mode="max-autotune",
            fullgraph=True,
        )

        try:
            self.build_sdf_texture(model_id=0, force_rebuild=False)
            # Pre-allocate _sdf_tex_grid before CUDA graph capture to avoid issues
            if self.device == "cuda" and use_cuda_graph:
                if getattr(self, "_sdf_tex", None) is not None:
                    if (getattr(self, "_sdf_tex_grid", None) is None) or (self._sdf_tex_grid.shape[:2] != (B, H)):
                        tex = self._sdf_tex
                        self._sdf_tex_grid = torch.zeros(B, H, 1, 2, device=tex.device, dtype=tex.dtype)
        except Exception:
            pass

        self._acc_rollout.prepare()

    @torch.no_grad()
    def sample_sdf_and_grad(self, q_ws, model_id=0):
        device = q_ws.device
        B, H, _ = q_ws.shape

        if device.type == "cuda":
            try:
                # Only build texture if it doesn't exist to avoid issues during CUDA graph capture
                if getattr(self, "_sdf_tex", None) is None:
                    self.build_sdf_texture(model_id=model_id)
                if getattr(self, "_sdf_tex", None) is not None:
                    tex = self._sdf_tex
                    # Use getattr with None default to avoid Dynamo attribute access issues
                    invW = getattr(self, "_sdf_tex_invW", None)
                    invH = getattr(self, "_sdf_tex_invH", None)
                    xbias = getattr(self, "_sdf_tex_xbias", None)
                    ybias = getattr(self, "_sdf_tex_ybias", None)

                    if invW is None or invH is None or xbias is None or ybias is None:
                        # Fallback: recalculate if not set
                        meta = getattr(self, "_sdf_tex_meta", {})
                        if meta:
                            x_min = meta.get("x0", -2.0)
                            y_min = meta.get("y0", -2.0)
                            res = meta.get("res", 0.01)
                            W = meta.get("W", 1)
                            H_tex = meta.get("H", 1)
                            x_max = x_min + (W - 1) * res
                            y_max = y_min + (H_tex - 1) * res
                            x_range = x_max - x_min
                            y_range = y_max - y_min
                            invW = torch.tensor(2.0 / x_range, device=tex.device, dtype=tex.dtype)
                            invH = torch.tensor(2.0 / y_range, device=tex.device, dtype=tex.dtype)
                            xbias = torch.tensor(-1.0 - 2.0 * x_min / x_range, device=tex.device, dtype=tex.dtype)
                            ybias = torch.tensor(-1.0 - 2.0 * y_min / y_range, device=tex.device, dtype=tex.dtype)

                    x = q_ws[..., 0].to(tex.device)
                    y = q_ws[..., 1].to(tex.device)

                    xn = x.mul(invW).add_(xbias)
                    yn = y.mul(invH).add_(ybias)

                    if (getattr(self, "_sdf_tex_grid", None) is None) or (self._sdf_tex_grid.shape[:2] != (B, H)):
                        self._sdf_tex_grid = torch.zeros(B, H, 1, 2, device=tex.device, dtype=tex.dtype)
                    grid = self._sdf_tex_grid

                    # Reshape xn and yn to match grid[..., 0] and grid[..., 1] shapes
                    # grid[..., 0] has shape (B, H, 1), so we need xn to be (B, H, 1)
                    # Handle different input shapes:
                    # - When q_ws is (B, H, 2): xn is (B, H), need to add dimension -> (B, H, 1)
                    # - When q_ws is (B*H, 1, 2): xn is (B*H, 1), need to reshape -> (B, H, 1)
                    # - When q_ws is (B*H, 2): xn is (B*H,), need to reshape -> (B, H, 1)
                    if xn.shape != grid[..., 0].shape:
                        if xn.dim() == 2:
                            if xn.shape == (B, H):
                                # xn is (B, H), add dimension -> (B, H, 1)
                                xn = xn.unsqueeze(-1)
                            elif xn.shape[1] == 1:
                                # xn is (B*H, 1), reshape to (B, H, 1)
                                xn = xn.view(B, H, 1)
                            else:
                                # Unexpected shape, try to reshape
                                xn = xn.view(B, H, 1)
                        elif xn.dim() == 1:
                            # xn is (B*H,), reshape to (B, H, 1)
                            xn = xn.view(B, H, 1)
                        else:
                            # Already correct shape or unexpected, try to reshape
                            xn = xn.view(B, H, 1)

                    if yn.shape != grid[..., 1].shape:
                        if yn.dim() == 2:
                            if yn.shape == (B, H):
                                # yn is (B, H), add dimension -> (B, H, 1)
                                yn = yn.unsqueeze(-1)
                            elif yn.shape[1] == 1:
                                # yn is (B*H, 1), reshape to (B, H, 1)
                                yn = yn.view(B, H, 1)
                            else:
                                # Unexpected shape, try to reshape
                                yn = yn.view(B, H, 1)
                        elif yn.dim() == 1:
                            # yn is (B*H,), reshape to (B, H, 1)
                            yn = yn.view(B, H, 1)
                        else:
                            # Already correct shape or unexpected, try to reshape
                            yn = yn.view(B, H, 1)

                    grid[..., 0].copy_(xn)
                    grid[..., 1].copy_(yn)

                    out = F.grid_sample(tex.expand(B, -1, -1, -1), grid, mode="bilinear", padding_mode="border",
                                        align_corners=True).squeeze(-1)  # [B,3,H,1]

                    sdf = out[:, 0, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    gx = out[:, 1, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    gy = out[:, 2, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    grad = torch.stack((gx, gy), dim=-1)  # (B,H,2)
                    return sdf, grad
            except Exception:
                pass

        env = self.env_models[model_id]
        q_flat = q_ws.reshape(-1, 2).to(self.device)
        sdf = env.compute_sdf(q_flat, reshape_shape=(B, H))  # (B,H)

        eps = 1e-3
        qx_p, qx_m = q_flat.clone(), q_flat.clone()
        qy_p, qy_m = q_flat.clone(), q_flat.clone()
        qx_p[:, 0] += eps
        qx_m[:, 0] -= eps
        qy_p[:, 1] += eps
        qy_m[:, 1] -= eps
        sdf_xp = env.compute_sdf(qx_p, reshape_shape=(B, H))
        sdf_xm = env.compute_sdf(qx_m, reshape_shape=(B, H))
        sdf_yp = env.compute_sdf(qy_p, reshape_shape=(B, H))
        sdf_ym = env.compute_sdf(qy_m, reshape_shape=(B, H))
        gx_all = (sdf_xp - sdf_xm) / (2 * eps)
        gy_all = (sdf_yp - sdf_ym) / (2 * eps)

        tau = 0.05
        band = (sdf.abs() < tau).to(q_ws.dtype)  # (B,H) -> float
        gx = gx_all * band
        gy = gy_all * band
        grad = torch.stack((gx, gy), dim=-1)  # (B,H,2)
        return sdf, grad

    @torch.no_grad()
    def build_sdf_texture(self, model_id=0, force_rebuild: bool = False, res: float = 0.01):
        if hasattr(self, "_sdf_tex") and (self._sdf_tex is not None) and (not force_rebuild):
            return

        env0 = self.env_models[model_id]
        try:
            x_min, x_max, y_min, y_max = env0.bounds
        except Exception:
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        # Get device from environment model to ensure tensors are on correct device
        # Try to get device from limits tensor, tensor_args, or fallback to self.device
        if hasattr(env0, "limits") and env0.limits is not None:
            device = env0.limits.device
        elif hasattr(env0, "tensor_args") and env0.tensor_args is not None:
            device = env0.tensor_args.get("device", self.device)
            if isinstance(device, str):
                device = torch.device(device)
        else:
            device = torch.device(self.device) if isinstance(self.device, str) else self.device

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        # Create tensors on the correct device
        xs = torch.linspace(x_min, x_max, Wm, device=device)
        ys = torch.linspace(y_min, y_max, Hm, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hm,Wm]
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [Hm*Wm,2]

        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm)).to(torch.float32)  # [Hm,Wm], float32 CPU/GPU
        gx = torch.zeros_like(sdf)
        gy = torch.zeros_like(sdf)
        gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * res)
        gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * res)
        gx[:, 0] = (sdf[:, 1] - sdf[:, 0]) / res
        gx[:, -1] = (sdf[:, -1] - sdf[:, -2]) / res
        gy[0, :] = (sdf[1, :] - sdf[0, :]) / res
        gy[-1, :] = (sdf[-1, :] - sdf[-2, :]) / res

        tex = torch.stack((sdf, gx, gy), dim=0).unsqueeze(0)
        prefer = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tex = tex.to(device=prefer, dtype=torch.float32)

        self._sdf_tex = tex
        self._sdf_tex_meta = {"x0": x_min, "y0": y_min, "res": float(res), "H": int(Hm), "W": int(Wm)}

        x_range = x_max - x_min
        y_range = y_max - y_min
        self._sdf_tex_invW = torch.tensor(2.0 / x_range, device=tex.device, dtype=tex.dtype)
        self._sdf_tex_invH = torch.tensor(2.0 / y_range, device=tex.device, dtype=tex.dtype)
        self._sdf_tex_xbias = torch.tensor(-1.0 - 2.0 * x_min / x_range, device=tex.device, dtype=tex.dtype)
        self._sdf_tex_ybias = torch.tensor(-1.0 - 2.0 * y_min / y_range, device=tex.device, dtype=tex.dtype)
        return

    @torch.no_grad()
    def get_sdf_grid(self, model_id=0, res: float = 0.01):
        env0 = self.env_models[model_id]
        try:
            x_min, x_max, y_min, y_max = env0.bounds
        except Exception:
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        if hasattr(env0, "limits") and env0.limits is not None:
            device = env0.limits.device
        elif hasattr(env0, "tensor_args") and env0.tensor_args is not None:
            device = env0.tensor_args.get("device", self.device)
            if isinstance(device, str):
                device = torch.device(device)
        else:
            device = torch.device(self.device) if isinstance(self.device, str) else self.device

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        xs = torch.linspace(x_min, x_max, Wm, device=device)
        ys = torch.linspace(y_min, y_max, Hm, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm))
        return sdf, (x_min, x_max, y_min, y_max), res

    @torch.inference_mode()
    def _rollout_double_batch(self, model_id, state_q, us, constraints):
        """
        Double integrator rollout.
        Dynamics:
            v_{t+1} = v_t + a_t * dt
            q_{t+1} = q_t + v_t * dt + 0.5 * a_t * dt^2
        Here `us` is interpreted as acceleration a
        """
        # -------------------- setup --------------------
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q0 = state_q.to(device).view(1, 1, 2).expand(B, 1, 2)  # (B,1,2)
        v0 = torch.zeros(B, 1, 2, device=device)  # (B,1,2)

        a_safe = us.clone()  # (B,H,2)
        a_flat = a_safe.reshape(B * H, 2)  # (B*H,2)

        has_tf = (self.transforms is not None) and (model_id in self.transforms)
        offset = self.transforms[model_id].to(device).view(1, 2) if has_tf else None

        q_init_exp = (
            q0.expand(B, H, 2).contiguous().view(B * H, 2).clone()
        )
        q_ws_flat = q_init_exp + offset if has_tf else q_init_exp  # (B*H,2)

        # -------------------- SDF & grad  --------------------
        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)  # (B*H,)
        A = grad_flat.view(B * H, 2)  # (B*H,2)
        Anorm = A.norm(dim=-1)  # (B*H,)

        R = self.robot_radius
        a_cum = torch.cumsum(a_safe, dim=1)  # (B,H,2)
        a_prefix = torch.nn.functional.pad(a_cum[:, :-1, :], (0, 0, 0, 1))  # (B,H,2)
        v_t = v0 + a_prefix * dt  # (B,H,2)
        v_flat = v_t.reshape(B * H, 2)  # (B*H,2)

        margin0 = 0.01
        h = sdf_flat - (R + margin0)  # (B*H,)
        Anorm = A.norm(dim=-1)  # (B*H,)
        goodA = (Anorm > 1e-9)
        n = A / (Anorm.view(-1, 1) + 1e-12)  # (B*H,2)

        hdot = (n * v_flat).sum(dim=-1)  # (B*H,)
        b = -(self.cbf_k1 * hdot + self.cbf_k0 * h)  # (B*H,)
        need = goodA & (h < self.cbf_tau)

        a_nom = a_flat  # (B*H,2)
        lhs = (n * a_nom).sum(dim=-1)  # (B*H,)
        den = (n * n).sum(dim=-1).add_(1e-9)  # ≈1
        delta = ((b - lhs).clamp_min(0.0) / den) * need.to(a_nom.dtype)
        a_flat = a_nom + delta.unsqueeze(-1) * n  # (B*H,2)
        a_safe = a_flat.view(B, H, 2)
        # -------------------- integrate: double integrator --------------------
        v_seq = v0 + torch.cumsum(a_safe * dt, dim=1)  # (B,H,2)
        # q_{t+1} = q_t + v_t*dt + 0.5*a_t*dt^2
        dq = v_seq * dt + 0.5 * a_safe * (dt * dt)  # (B,H,2)
        q_seq = q0 + torch.cumsum(dq, dim=1)  # (B,H,2)

        # -------------------- Vertex Constraints --------------------
        C_list = [constraints.C2[model_id]]
        R_list = [constraints.R[model_id]]
        TM_list = [constraints.TM[model_id]]

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], dim=0).T  # (H,N)

            q_ws_seq = q_seq + offset.view(1, 1, 2) if has_tf else q_seq  # (B,H,2)

            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min_(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            near = (h_s < 0.08) & active

            dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
            k = min(self.k_best, dist.shape[-1])
            topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            idx = topk.indices  # (B,H,K)

            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,H,K,2)
            sel_A = torch.gather(A_s, 2, idx_e)  # (B,H,K,2)
            den_k = (sel_A * sel_A).sum(dim=-1).add_(1e-9)  # (B,H,K)
            idx = torch.topk(dist_masked, k=k, dim=-1, largest=False).indices  # (B,H,K)
            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,H,K,2)
            A_unit_all = diff / (dist.unsqueeze(-1) + 1e-12)  # (B,H,N,2)
            sel_n = torch.gather(A_unit_all, 2, idx_e)  # (B,H,K,2)
            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

            v_rep = v_seq.unsqueeze(2).expand(-1, -1, k, -1)  # (B,H,K,2)
            hdot_k = (sel_n * v_rep).sum(dim=-1)  # (B,H,K)
            sel_b = -(self.cbf_k1 * hdot_k + self.cbf_k0 * sel_h)
            U = a_safe.clone()
            for kk in range(k):
                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                bk = sel_b[:, :, kk]  # (B,H)
                lhsk = (Ak * U).sum(dim=-1)  # (B,H)
                viol = (lhsk < bk).to(U.dtype)
                num = (bk - lhsk).clamp_min(0.0)
                alpha_k = (num / den_k[:, :, kk]) * viol
                U = U + alpha_k.unsqueeze(-1) * Ak
            a_safe = U

        v_seq = v0 + torch.cumsum(a_safe * dt, dim=1)
        dq = v_seq * dt + 0.5 * a_safe * (dt * dt)
        q_seq = q0 + torch.cumsum(dq, dim=1)

        # -------------------- Costs --------------------
        q_goal = self.goal_state_pos.to(device)
        qg = q_goal.view(1, 1, 2)
        dist_to_goal = (q_seq - qg).norm(dim=2)  # (B,H)

        ctrl = (a_safe * a_safe).sum(dim=2)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost.add_(ctrl, alpha=self.cost_control)
        runtime_cost.add_(dist_to_goal, alpha=self.cost_distance_to_goal)

        t_star = max(1, H - 1)
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost.add_((dist_to_goal - d_hat).pow(2), alpha=self.cost_time_smoothness)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)  # (1,H,1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        v_norm = v_seq.norm(dim=-1, keepdim=True)  # (B,H,1)
        v_err = v_norm - v_bar
        runtime_cost.add_((v_err.squeeze(-1) ** 2), alpha=self.cost_acc_smoothness)

        inside_goal_early = (dist_to_goal <= 0.1) & (t_idx < t_star)
        tau_e = ((t_star - t_idx).clamp(min=0).float() / H)
        runtime_cost.add_(tau_e * inside_goal_early.float(), alpha=self.cost_get_to_goal_early)

        q_ws = q_seq + offset.view(1, 1, 2) if has_tf else q_seq
        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (R + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost.add_(pen * pen, alpha=self.cost_sdf_collison)

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        w_T = getattr(self, "w_T_base", self.cost_terminal)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    @torch.inference_mode()
    def _rollout_single_batch_new(self, model_id, state_q, us, constraints=None):
        # -------------------- setup --------------------
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q_init = state_q.to(device).view(1, 1, 2).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)  # (B*H,2)

        has_tf = (self.transforms is not None) and (model_id in self.transforms)
        offset = self.transforms[model_id].to(device).view(1, 2) if has_tf else None

        q_init_exp = (
            q_init.expand(B, H, 2)
            .contiguous()
            .view(B * H, 2)
            .clone()
        )
        q_ws_flat = q_init_exp + offset if has_tf else q_init_exp  # (B*H,2)

        # -------------------- SDF & grad  --------------------
        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)  # (B*H,)
        A = grad_flat.view(B * H, 2)  # (B*H,2)

        # -------------------- CBF --------------------
        R = self.robot_radius

        h = sdf_flat - (R + self.cbf_margin)  # (B*H,)
        u_flat = us_flat.clone()

        Anorm = A.norm(dim=-1)  # (B*H,)
        goodA = (Anorm > 1e-9)  # bool

        b = - (self.cbf_eta / dt) * h  # (B*H,)
        lhs = (A * u_flat).sum(dim=-1)  # (B*H,)
        den = (A * A).sum(dim=-1).add_(1e-9)  # (B*H,)

        need = goodA & (h < self.cbf_tau)  # bool
        alpha = ((b - lhs) / den).clamp_min(0.0) * need.to(u_flat.dtype)
        u_flat = u_flat + alpha.unsqueeze(-1) * A

        lhs2 = (A * u_flat).sum(dim=-1)  # (B*H,)
        bad = goodA & (lhs2 < b)  # bool
        alpha2 = ((b - lhs2) / den).clamp_min(0.0) * bad.to(u_flat.dtype)
        u_flat = u_flat + alpha2.unsqueeze(-1) * A

        n = A / (Anorm.view(-1, 1) + 1e-12)  # (B*H,2)
        step = (-h * 0.5).clamp_min(0.0).clamp_max(0.10)  # (B*H,)
        q_ws_flat_new = q_ws_flat + step.unsqueeze(-1) * n  # (B*H,2)
        mask_si = (h < 0.0).view(-1, 1)  # (B*H,1)
        blended = torch.where(mask_si, q_ws_flat_new - offset, q_init_exp) if has_tf else torch.where(mask_si,
                                                                                                      q_ws_flat_new,
                                                                                                      q_init_exp)
        q_init_exp.copy_(blended)

        # -------------------- integrate --------------------
        u_safe = u_flat.view(B, H, 2)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)  # (B,H,2)

        # -------------------- Vertex Constraints --------------------
        C_list = [constraints.C2[model_id]]  # (N_max, 2)
        R_list = [constraints.R[model_id]]  # (N_max,)
        TM_list = [constraints.TM[model_id]]

        # C_list, R_list, TM_list = [], [], []
        # if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
        # if model_id in constraints:
        #     for sph in (constraints[model_id] or []):
        #         if not sph:
        #             continue
        #         C_cc = sph.qs.to(device)[:, :2]
        #         R_cc = sph.radii.to(device)
        #         TR = sph.traj_ranges.to(device).long()  # (n,2)
        #         n_cc = C_cc.shape[0]
        #         if n_cc == 0:
        #             continue
        #         t_idx_full = torch.arange(H, device=device).view(1, H).expand(n_cc, -1)  # (n,H)
        #         TM_cc = (t_idx_full >= TR[:, 0:1]) & (t_idx_full < TR[:, 1:2])  # (n,H)
        #         C_list.append(C_cc)
        #         R_list.append(R_cc)
        #         TM_list.append(TM_cc)

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], dim=0).T  # (H,N)

            q_ws_seq = q_seq + offset.view(1, 1, 2) if has_tf else q_seq  # (B,H,2)

            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min_(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            near = (h_s < 0.08) & active  # (B,H,N)

            dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
            k = min(self.k_best, dist_masked.shape[-1])
            topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            idx = topk.indices  # (B,H,K)

            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,H,K,2)
            sel_A = torch.gather(A_s, 2, idx_e)  # (B,H,K,2)
            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

            # keep here for future priority selection for deal lock optimization
            sign = 1.0 if (getattr(self, "agent_id", 0) % 2 == 0) else -1.0
            beta = sign * self.base_beta

            U = u_safe.clone()
            den_k = (sel_A * sel_A).sum(dim=-1).add_(1e-9)  # (B,H,K)
            sel_b = -(self.cbf_eta / dt) * sel_h + beta

            for kk in range(k):
                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                bk = sel_b[:, :, kk]  # (B,H)
                lhsk = (Ak * U).sum(dim=-1)  # (B,H)
                viol = (lhsk < bk).to(U.dtype)  # (B,H) float mask
                num = (bk - lhsk).clamp_min(0.0)  # (B,H)
                alpha_k = (num / den_k[:, :, kk]) * viol  # (B,H)
                U = U + alpha_k.unsqueeze(-1) * Ak
            u_safe = U

        # u_safe = project_controls_smooth(u_safe, mu=1e-3, iters=8)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ====================== Cost Function =========================#
        q_goal = self.goal_state_pos.to(device)
        qg = q_goal.view(1, 1, 2)
        diff_g = q_seq - qg
        dist_to_goal = diff_g.norm(dim=2)  # (B,H)
        ctrl = (u_safe * u_safe).sum(dim=2)  # (B,H)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost.add_(ctrl, alpha=self.cost_control)
        runtime_cost.add_(dist_to_goal, alpha=self.cost_distance_to_goal)

        t_star = max(1, H - 1)
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost.add_((dist_to_goal - d_hat).pow(2), alpha=self.cost_time_smoothness)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)  # (1,H,1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        u_norm = u_safe.norm(dim=-1, keepdim=True)  # (B,H,1)
        v_err = u_norm - v_bar
        runtime_cost.add_(v_err.squeeze(-1) * v_err.squeeze(-1), alpha=self.cost_acc_smoothness)

        inside_goal_early = (dist_to_goal <= 0.1) & (t_idx < t_star)
        tau_e = ((t_star - t_idx).clamp(min=0).float() / H)
        runtime_cost.add_(tau_e * inside_goal_early.float(), alpha=self.cost_get_to_goal_early)

        q_ws = q_seq + offset.view(1, 1, 2) if has_tf else q_seq
        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (R + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost.add_(pen * pen, alpha=self.cost_sdf_collison)

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        w_T = getattr(self, "w_T_base", self.cost_terminal)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    def _rollout_single_batch(self, model_id, state_q, us, constraints=None):
        env = self.env_models[model_id]
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q_init = state_q.view(1, 1, 2).to(device).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)  # (B*H, 2)
        q_init_expanded = q_init.expand(B, H, 2).reshape(B * H, 2).clone()  # (B*H, 2)

        q_ws_flat = q_init_expanded
        if self.transforms is not None and model_id in self.transforms:
            q_ws_flat = q_init_expanded + self.transforms[model_id].to(device).view(1, -1)

        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)
        grad_flat = grad_flat.view(B * H, 2)

        R = self.robot_radius
        h = sdf_flat - (R + self.cbf_margin)
        A = grad_flat
        u_flat = us_flat.clone()

        goodA = (A.norm(dim=-1) > 1e-9)
        need = (h < self.cbf_tau) & goodA

        b = - (self.cbf_eta / dt) * h
        lhs = (A * u_flat).sum(dim=-1)
        den = (A.pow(2).sum(dim=-1) + 1e-9)

        mask = need.to(u_flat.dtype)
        alpha = ((b - lhs) / den).clamp_min(0.0) * mask
        u_flat = u_flat + alpha.unsqueeze(-1) * A

        lhs2 = (A * u_flat).sum(dim=-1)
        bad = goodA & (lhs2 < b)
        bad_mask = bad.to(u_flat.dtype)
        alpha2 = ((b - lhs2) / den).clamp_min(0.0) * bad_mask
        u_flat = u_flat + alpha2.unsqueeze(-1) * A

        # still_mask = (h < 0.0).view(-1, 1)  # (B*H,1) bool
        # mask_f = still_mask.to(q_ws_flat.dtype)
        #
        # n = torch.nn.functional.normalize(A, dim=-1)  # (B*H,2)
        # step = (0.5 * (-h).clamp_min(0.0)).clamp_max(0.10)  # (B*H,)
        # q_ws_flat_new = q_ws_flat + step.unsqueeze(-1) * n  # (B*H,2)
        #
        # if self.transforms is not None and model_id in self.transforms:
        #     offset = self.transforms[model_id].to(device).view(1, -1)
        #     q_init_expanded = torch.where(
        #         still_mask,
        #         q_ws_flat_new - offset,
        #         q_init_expanded
        #     )
        # else:
        #     q_init_expanded = torch.where(still_mask, q_ws_flat_new, q_init_expanded)

        # still_in = h < 0.0
        # if still_in.any():
        #     step = (0.5 * (-h[still_in])).clamp_max(0.10)
        #     n = torch.nn.functional.normalize(A[still_in], dim=-1)
        #     q_ws_flat_new = q_ws_flat.clone()
        #     q_ws_flat_new[still_in] = q_ws_flat_new[still_in] + step.unsqueeze(-1) * n
        #     if self.transforms is not None and model_id in self.transforms:
        #         q_init_expanded[still_in] = q_ws_flat_new[still_in] - self.transforms[model_id].to(device).view(1, -1)
        #     else:
        #         q_init_expanded[still_in] = q_ws_flat_new[still_in]

        u_safe = u_flat.reshape(B, H, 2)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ========= Vertex Constraints ======== #
        C_list, R_list, TM_list = [], [], []
        if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
            for sph in (self.soft_constraints[model_id] or []):
                C_cc = sph.qs.to(device)[:, :2]
                R_cc = sph.radii.to(device)
                TR = sph.traj_ranges.to(device).long()  # (n,2)
                n_cc = C_cc.shape[0]
                t_idx_full = torch.arange(H, device=device).view(1, H).expand(n_cc, -1)  # (n,H)
                TM_cc = (t_idx_full >= TR[:, 0:1]) & (t_idx_full < TR[:, 1:2])  # (n,H)

                if n_cc > 0:
                    C_list.append(C_cc)  # (n,2)
                    R_list.append(R_cc)  # (n,)
                    TM_list.append(TM_cc)  # (n,H)

        # C_list = [constraints.C2[model_id]]  # (N_max, 2)
        # R_list = [constraints.R[model_id]]  # (N_max,)
        # TM_list = [constraints.TM[model_id]]

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_cat = torch.cat([tm if tm.dim() == 2 else tm.view(-1, H) for tm in TM_list], dim=0)  # (N,H)
            TM_all = TM_cat.T  # (H,N)

            q_ws_seq = q_seq
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(device)
                q_ws_seq = q_seq + offset.view(1, 1, -1)  # (B,H,2)

            # （B,H,N）
            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            # time window & Narrow
            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            near = (h_s < 0.08) & active  # (B,H,N)

            dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))  # Top-K
            k = min(self.k_best, dist_masked.shape[-1])
            topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            idx = topk.indices  # (B,H,K)

            sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

            # keep here for future priority selection for deal lock optimization
            sign = 1.0 if (getattr(self, "agent_id", 0) % 2 == 0) else -1.0
            scale_pw = 1.0
            beta = sign * self.base_beta * scale_pw

            den_k = (sel_A * sel_A).sum(dim=-1).add_(1e-9)  # (B,H,K)
            sel_b = -(self.cbf_eta / dt) * sel_h + beta
            U = u_safe.clone()

            for kk in range(k):
                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                bk = sel_b[:, :, kk]  # (B,H)
                lhsk = (Ak * U).sum(dim=-1)  # (B,H)
                viol = (lhsk < bk).to(U.dtype)  # (B,H) float mask
                num = (bk - lhsk).clamp_min(0.0)  # (B,H)
                alpha_k = (num / den_k[:, :, kk]) * viol  # (B,H)
                U = U + alpha_k.unsqueeze(-1) * Ak
            u_safe = U

            # if near.any():
            #     dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
            #     k = min(self.k_best, dist_masked.shape[-1])
            #     topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            #     idx = topk.indices  # (B,H,K)
            #
            #     sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
            #     sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)
            #
            #     # ---- priority_weight (for future) ----
            #     sign = + 1.0
            #     if hasattr(self, "agent_id"):
            #         sign = +1.0 if (self.agent_id % 2 == 0) else -1.0
            #     scale_pw = 1.0
            #
            #     beta = sign * self.base_beta * scale_pw
            #     sel_b = - (self.cbf_eta / dt) * sel_h + beta  # (B,H,K)
            #     U = u_safe.clone()  # (B,H,2)
            #
            #     for kk in range(k):
            #         Ak = sel_A[:, :, kk, :]  # (B,H,2)
            #         bk = sel_b[:, :, kk]  # (B,H)
            #         lhs_k = (Ak * U).sum(dim=-1)  # (B,H)
            #         viol = lhs_k < bk
            #         if viol.any():
            #             den_k = (Ak.pow(2).sum(dim=-1) + 1e-9)
            #             alpha_k = torch.zeros_like(lhs_k)
            #             alpha_k[viol] = (bk[viol] - lhs_k[viol]) / den_k[viol]
            #             U = U + alpha_k.unsqueeze(-1) * Ak
            #     u_safe = U  #

        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ===== costs: runtime + terminal =====
        q_goal = self.goal_state_pos.to(device)
        dist_to_goal = (q_seq - q_goal.view(1, 1, -1)).norm(dim=2)  # (B,H)
        ctrl = (u_safe * u_safe).sum(dim=2)  # (B,H)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost += self.cost_control * ctrl
        runtime_cost += self.cost_distance_to_goal * dist_to_goal

        t_star = max(1, H - 1)
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost += self.cost_time_smoothness * (dist_to_goal - d_hat).pow(2)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        u_norm = u_safe.norm(dim=-1, keepdim=True)
        runtime_cost += self.cost_acc_smoothness * (u_norm - v_bar).pow(2).squeeze(-1)

        inside_goal_early = (dist_to_goal <= 0.10) & (t_idx < t_star)
        runtime_cost += self.cost_get_to_goal_early * (
                (t_star - t_idx).clamp(min=0).float() / H) * inside_goal_early.float()

        q_ws = q_seq
        if self.transforms is not None and model_id in self.transforms:
            offset = self.transforms[model_id].to(device)
            q_ws = q_seq + offset.view(1, 1, -1)

        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (self.robot_radius + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost += self.cost_sdf_collison * pen.pow(2)
        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        # terminal
        w_T = getattr(self, "w_T_base", self.cost_terminal)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    @torch.no_grad()
    def _reverse_diffusion_step(
            self,
            model_id: int,
            i: int,
            traj_bars_i: torch.Tensor,
    ):
        """
        Reverse process for single model
        """
        model_params = self.models[model_id]
        temp_step = model_params['params']['temp_sample']
        n_samples = model_params['params']['n_samples']

        # recover the trajectory
        if traj_bars_i.shape[0] != n_samples:
            traj_bars_i = traj_bars_i[:1].repeat(n_samples, 1, 1)

        if i > 0:
            traj_i = traj_bars_i * torch.sqrt(model_params['alphas_bar'][i])
        else:
            traj_i = traj_bars_i.clone()

        #  Monte Carlo Sample from q_i velocity
        eps_u = torch.randn((n_samples, model_params['params']['horizon'], self.robot.q_dim * 2),
                            device=self.device)
        traj_0s = torch.clamp(eps_u * model_params['sigmas'][i] + traj_bars_i, -1, 1)
        actions = traj_0s[..., self.robot.q_dim:]

        if self.compile:
            costs, q_seq, free_mask = self._acc_rollout.run(actions, self.constraints_ro)
        else:
            costs, q_seq, free_mask = self._rollout_single_batch(model_id, self.state_inits.q, actions, self.soft_constraints)

        traj_0s[..., :self.robot.q_dim] = q_seq

        if not free_mask.any():
            raise RuntimeError(f"No collision-free sample after resamples")

        cost_mean = costs.mean()
        cost_std = costs.std().clamp(min=1e-4)
        logp0 = -(costs - cost_mean) / cost_std / temp_step
        logp0[~free_mask] = -torch.inf

        # use demonstration data
        # if self.enable_demo:
        #     xref_logpds = torch.tensor([env_model.eval_xref_logpd(q) for q in qs], device=self.device)
        #     xref_logpds = xref_logpds - xref_logpds.max()
        #     logpdemo = -(xref_logpds + env_model.rew_xref - cost_mean) / cost_std / model_params['params'][
        #         'temp_sample']
        #     demo_mask = logpdemo > logp0
        #     logp0 = torch.where(demo_mask, logpdemo, logp0)
        #     logp0 = (logp0 - logp0.mean()) / logp0.std() / model_params['params']['temp_sample']

        # weighted trajectory
        weights = torch.nn.functional.softmax(logp0, dim=0)
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)

        # calculate score function
        score = 1 / (1.0 - model_params['alphas_bar'][i]) * (
                - traj_i + torch.sqrt(model_params['alphas_bar'][i]) * traj_bar)


        proj_weight = self.projection_score_weight
        traj_i_m1 = (1.0 / torch.sqrt(model_params['alphas'][i]) *
                     (traj_i + (1.0 - model_params['alphas_bar'][i]) * score)
                     ) / torch.sqrt(model_params['alphas_bar'][i - 1])

        traj_proj = traj_i_m1.clone()
        traj_proj[..., :self.robot.q_dim] = q_seq
        traj_i_m1 = (1.0 - proj_weight) * traj_i_m1 + proj_weight * traj_proj
        traj_i_m1[:, :1, :self.robot.q_dim] = self.start_state_pos
        traj_i_m1[:, -1:, :self.robot.q_dim] = self.goal_state_pos

        best_idx = torch.where(free_mask)[0][costs[free_mask].argmin()]
        best_traj_sample = traj_0s[best_idx]

        if self.transforms is not None and model_id in self.transforms:
            best_traj_sample[..., :self.robot.q_dim] += self.transforms[model_id]

        return traj_i_m1, best_traj_sample, costs.mean().item()

    @torch.no_grad()
    def run_inference(
            self,
            contexts: List[dict] = None,
            hard_conds: Dict[int, dict] = None,
            cross_conds: Dict[Tuple[int, int], Tuple[int, int]] = None,
            model_ids: List[int] = None,
            n_samples: int = 100,
            batch_size: int = 32,
            return_chain: bool = False,
            soft_constraints=[],
            **diffusion_kwargs):
        """
        :param contexts:
        :param hard_conds:
        :param cross_conds: Keys are tuples of model indices and values are tuples of trajectory timesteps.
                            For example, {(0, 1): {0, 64}} means that the value of the state 0 in the trajectory
                            of model 0 needs to be the same as the value of the state 64 in the trajectory of model 1.
        :param n_samples:
        :param return_chain:
        :param diffusion_kwargs:
        :return:
            if return_chain is True，return the diffusion chain
            else return final optimized trajectory
        """
        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        results = {}
        chains = {}

        for model_id in self.models.keys():
            traj_n = torch.zeros((n_samples, self.models[model_id]['params']['horizon'], self.robot.q_dim * 2),
                                 device=self.device)
            B, H, _ = traj_n.shape
            # cold start soft constraints (acceleration)
            self.soft_constraints[model_id] = soft_constraints
            if len(self.soft_constraints[model_id]) > 0:
                self._prepare_soft_constraints(model_id, H, traj_n.device)

            self.constraints_ro = pack_constraints_dict(self.soft_constraints, M_max=1,
                                                        N_max=self.constraints_to_checks, D=2, H=self.H,
                                                        device=self.device)
            # Reverse
            traj_i = traj_n
            model_chain = []
            costs = []
            best_cost = float('inf')

            for i in tqdm(range(self.models[model_id]['params']['n_diffusion_step'] - 1, -1, -1),
                          desc=f"Diffusing model {model_id}"):
                traj_i, sample_i, cost_i = self._reverse_diffusion_step(model_id, i, traj_i)
                if return_chain:
                    model_chain.append(traj_i)
                costs.append(cost_i)
                if cost_i < best_cost:
                    best_cost, best_traj = cost_i, sample_i

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

        if return_chain:
            return chains

        return results