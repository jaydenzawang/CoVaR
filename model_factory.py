"""Validated construction of the model objects used by all entry points."""

from dataclasses import dataclass

import numpy as np

import config
from copula import (
    GaussianCopulaSingleFactor,
    SkewTCopulaSingleFactor,
    TCopulaSingleFactor,
)
from factor_integration import FactorGrid, build_factor_grid
from loss_model import PortfolioLoss


@dataclass(frozen=True)
class ModelContext:
    portfolio: PortfolioLoss
    copula: object
    factors: FactorGrid
    target_i: int
    alpha: float
    beta: float
    marginal_pd_max_abs_error: float
    spa_n_jobs: int
    spa_options: dict | None = None


def format_model_summary(context, condition):
    """Return a compact, model-aware description for numerical output."""
    copula = context.copula
    if isinstance(copula, GaussianCopulaSingleFactor):
        model = "Gaussian"
    elif isinstance(copula, SkewTCopulaSingleFactor):
        model = "Skew-t"
    elif isinstance(copula, TCopulaSingleFactor):
        model = "t"
    else:
        model = type(copula).__name__

    target_i = int(context.target_i)
    p_i = float(copula.p[target_i])
    rho_i = float(copula.rho[target_i])
    nu = f"{float(copula.nu):.1f}" if hasattr(copula, "nu") else "N/A"
    gamma = f"{float(copula.gamma):.1f}" if hasattr(copula, "gamma") else "N/A"
    xi = float(getattr(copula, "xi", 1.0))
    return (
        f"model={model}, target_i={target_i}, p_i={p_i:.8f}, "
        f"rho_i={rho_i:.8f}, nu={nu}, gamma={gamma}, "
        f"xi={xi:.1f}, condition={condition}"
    )


def _validate_configuration(cfg):
    arrays = [np.asarray(getattr(cfg, name), dtype=float) for name in ("u", "a", "b", "p", "rho")]
    n = int(cfg.N)
    if n < 1 or any(np.any(~np.isfinite(arr)) for arr in arrays):
        raise ValueError("N must be positive and portfolio inputs must be finite")
    if any(arr.shape != (n,) for arr in arrays):
        raise ValueError("u, a, b, p and rho must all have shape (N,)")
    if np.any((arrays[3] <= 0) | (arrays[3] >= 1)):
        raise ValueError("marginal default probabilities must lie strictly between zero and one")
    if np.any(np.abs(arrays[4]) >= 1):
        raise ValueError("factor loadings must have absolute value below one")
    if not 0 <= int(cfg.target_i) < n:
        raise IndexError("target_i is outside the portfolio")
    if not (0 < float(cfg.alpha) < 1 and 0 < float(cfg.beta) < 1):
        raise ValueError("alpha and beta must lie strictly between zero and one")
    if not np.isfinite(cfg.XI) or float(cfg.XI) < 1:
        raise ValueError("XI must be at least one")
    if int(getattr(cfg, "SPA_N_JOBS", 1)) < 1:
        raise ValueError("SPA_N_JOBS must be a positive integer")


def build_model_context(cfg=config, n_gh=None, n_gl=None, mgf_grid_size=None):
    _validate_configuration(cfg)
    if n_gh is None:
        n_gh = getattr(cfg, "SPA_GAUSSIAN_FACTOR_ORDER", 512) if cfg.COPULA_TYPE == "Gaussian" else getattr(cfg, "SPA_T_FACTOR_ORDER", 64)
    if n_gl is None:
        n_gl = getattr(cfg, "SPA_SCALE_ORDER", 64)
    if cfg.COPULA_TYPE == "Gaussian":
        copula = GaussianCopulaSingleFactor(cfg.rho, cfg.p)
    elif cfg.COPULA_TYPE == "t":
        copula = TCopulaSingleFactor(
            cfg.rho,
            cfg.p,
            cfg.nu,
            xi=cfg.XI,
            calibration_n_z=n_gh,
            calibration_n_lambda=n_gl,
        )
    elif cfg.COPULA_TYPE == "Skew-t":
        copula = SkewTCopulaSingleFactor(
            cfg.rho,
            cfg.p,
            cfg.nu,
            cfg.gamma,
            xi=cfg.XI,
            calibration_n_z=n_gh,
            calibration_n_lambda=n_gl,
        )
    else:
        raise ValueError(f"unknown COPULA_TYPE: {cfg.COPULA_TYPE}")

    if mgf_grid_size is None:
        mgf_grid_size = int(getattr(cfg, "SPA_MGF_GRID_SIZE", 1501))
    portfolio = PortfolioLoss(
        cfg.u,
        cfg.a,
        cfg.b,
        s_min=float(getattr(cfg, "SPA_MGF_S_MIN", -25.0)),
        s_max=float(getattr(cfg, "SPA_MGF_S_MAX", 25.0)),
        grid_size=mgf_grid_size,
        exact_mgf_radius=getattr(cfg, "SPA_EXACT_MGF_RADIUS", 0.15),
    )
    factors = build_factor_grid(copula, xi=cfg.XI, n_gh=n_gh, n_gl=n_gl)
    integrated_pd = factors.expectation(factors.p_cond)
    marginal_error = float(np.max(np.abs(integrated_pd - np.asarray(cfg.p))))
    if marginal_error > 5e-4:
        raise RuntimeError(
            f"factor quadrature does not preserve calibrated marginal PDs; "
            f"maximum absolute error={marginal_error:.3g}"
        )
    return ModelContext(
        portfolio=portfolio,
        copula=copula,
        factors=factors,
        target_i=int(cfg.target_i),
        alpha=float(cfg.alpha),
        beta=float(cfg.beta),
        marginal_pd_max_abs_error=marginal_error,
        spa_n_jobs=int(getattr(cfg, "SPA_N_JOBS", 1)),
        spa_options={
            "lgd_order": getattr(cfg, "SPA_LGD_ORDER", 32),
            "quantile_tol": getattr(cfg, "SPA_QUANTILE_TOL", 1e-5),
            "saddle_bound": getattr(cfg, "SPA_SADDLE_BOUND", 20.0),
            "root_tol": getattr(cfg, "SPA_ROOT_TOL", 1e-10),
            "mean_region": getattr(cfg, "SPA_MEAN_REGION", 1e-2),
            "tail_bound_tol": getattr(cfg, "SPA_TAIL_BOUND_TOL", 1e-12),
        },
    )


def build_spa_engine(context):
    """Use the numerical options captured when this context was built."""
    from risk_engine import HybridSPARiskEngine
    return HybridSPARiskEngine(context, **(getattr(context, "spa_options", None) or {}))
