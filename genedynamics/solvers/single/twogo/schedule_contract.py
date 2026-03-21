from __future__ import annotations

from typing import Any, Dict

import numpy as np
import jax.numpy as jnp


def resolve_twogo_overlay_config(config: Dict[str, Any] | None, *, rho_ref_default: float) -> Dict[str, float]:
    cfg = dict(config or {})
    hardness_cfg = dict(cfg.get("hardness", {}) or {})
    constraint_cfg = dict(cfg.get("constraint", {}) or {})
    diffusion_cfg = dict(cfg.get("diffusion", {}) or {})

    rho_ref = float(hardness_cfg.get("rho_ref", rho_ref_default if rho_ref_default > 0.0 else 500.0))
    rho_ref = max(rho_ref, 1e-6)

    return {
        "rho_ref": rho_ref,
        "kappa0": float(constraint_cfg.get("kappa0", 1.0)),
        "kappa_rho_gain": float(constraint_cfg.get("kappa_rho_gain", 0.01)),
        "delta_margin_scale": float(constraint_cfg.get("delta_margin_scale", 0.4)),
        "delta_rho_gain": float(constraint_cfg.get("delta_rho_gain", 0.02)),
        "delta_min": float(constraint_cfg.get("delta_min", 5e-4)),
        "sigma_max": float(diffusion_cfg.get("sigma_max", 0.25)),
        "sigma_hardness_power": float(diffusion_cfg.get("sigma_hardness_power", 1.0)),
        "theta_min": float(diffusion_cfg.get("theta_min", 0.2)),
        "theta_max": float(diffusion_cfg.get("theta_max", 0.6)),
        "eta_scale_min": float(diffusion_cfg.get("eta_scale_min", 0.5)),
        "eta_scale_max": float(diffusion_cfg.get("eta_scale_max", 1.0)),
        "eta_hardness_power": float(diffusion_cfg.get("eta_hardness_power", 1.0)),
        "gate_every": int(max(1, int(diffusion_cfg.get("gate_every", 1)))),
    }


def hardness_from_rho_jax(rho: jnp.ndarray, rho_ref: float) -> jnp.ndarray:
    rho_pos = jnp.maximum(rho.astype(jnp.float32), 0.0)
    denom = jnp.maximum(jnp.log1p(jnp.asarray(rho_ref, dtype=jnp.float32)), 1e-6)
    return jnp.clip(jnp.log1p(rho_pos) / denom, 0.0, 1.0)


def hardness_from_rho_numpy(rho: np.ndarray, rho_ref: float) -> np.ndarray:
    rho_arr = np.maximum(np.asarray(rho, dtype=np.float32), 0.0)
    denom = max(float(np.log1p(float(rho_ref))), 1e-6)
    return np.clip(np.log1p(rho_arr) / denom, 0.0, 1.0).astype(np.float32)


def constraint_overlay_jax(margin: jnp.ndarray, rho: jnp.ndarray, cfg: Dict[str, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
    rho_pos = jnp.maximum(rho.astype(jnp.float32), 0.0)
    margin_pos = jnp.maximum(margin.astype(jnp.float32), 0.0)
    kappa = jnp.asarray(cfg["kappa0"], dtype=jnp.float32) + jnp.asarray(cfg["kappa_rho_gain"], dtype=jnp.float32) * rho_pos
    delta0 = jnp.asarray(cfg["delta_margin_scale"], dtype=jnp.float32) * margin_pos
    delta = delta0 / (1.0 + jnp.asarray(cfg["delta_rho_gain"], dtype=jnp.float32) * rho_pos)
    delta = jnp.maximum(delta, jnp.asarray(cfg["delta_min"], dtype=jnp.float32))
    return kappa.astype(jnp.float32), delta.astype(jnp.float32)


def diffusion_overlay_jax(hardness: jnp.ndarray, eta_base: float, cfg: Dict[str, float]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    hard = jnp.clip(hardness.astype(jnp.float32), 0.0, 1.0)
    one_minus_h = 1.0 - hard
    sigma = jnp.asarray(cfg["sigma_max"], dtype=jnp.float32) * (
        one_minus_h ** jnp.asarray(cfg["sigma_hardness_power"], dtype=jnp.float32)
    )
    theta = jnp.asarray(cfg["theta_max"], dtype=jnp.float32) - (
        jnp.asarray(cfg["theta_max"] - cfg["theta_min"], dtype=jnp.float32) * hard
    )
    eta_scale = jnp.asarray(cfg["eta_scale_min"], dtype=jnp.float32) + (
        jnp.asarray(cfg["eta_scale_max"] - cfg["eta_scale_min"], dtype=jnp.float32)
        * (one_minus_h ** jnp.asarray(cfg["eta_hardness_power"], dtype=jnp.float32))
    )
    eta = jnp.asarray(eta_base, dtype=jnp.float32) * eta_scale
    return sigma.astype(jnp.float32), theta.astype(jnp.float32), eta.astype(jnp.float32)


def constraint_overlay_numpy(margin: np.ndarray, rho: np.ndarray, cfg: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rho_arr = np.maximum(np.asarray(rho, dtype=np.float32), 0.0)
    margin_arr = np.maximum(np.asarray(margin, dtype=np.float32), 0.0)
    kappa = cfg["kappa0"] + cfg["kappa_rho_gain"] * rho_arr
    delta0 = cfg["delta_margin_scale"] * margin_arr
    delta = delta0 / (1.0 + cfg["delta_rho_gain"] * rho_arr)
    delta = np.maximum(delta, cfg["delta_min"])
    return np.asarray(kappa, dtype=np.float32), np.asarray(delta, dtype=np.float32)


def diffusion_overlay_numpy(hardness: np.ndarray, eta_base: float, cfg: Dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hard = np.clip(np.asarray(hardness, dtype=np.float32), 0.0, 1.0)
    one_minus_h = 1.0 - hard
    sigma = cfg["sigma_max"] * (one_minus_h ** cfg["sigma_hardness_power"])
    theta = cfg["theta_max"] - (cfg["theta_max"] - cfg["theta_min"]) * hard
    eta_scale = cfg["eta_scale_min"] + (cfg["eta_scale_max"] - cfg["eta_scale_min"]) * (
        one_minus_h ** cfg["eta_hardness_power"]
    )
    eta = float(eta_base) * eta_scale
    return np.asarray(sigma, dtype=np.float32), np.asarray(theta, dtype=np.float32), np.asarray(eta, dtype=np.float32)
