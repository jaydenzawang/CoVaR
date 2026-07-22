"""Deterministic integration over the common-factor distribution.

The grid is built once and reused at every CDF evaluation.  This provides
common numerical points during quantile root finding and removes the large
amount of duplicated quadrature code that previously lived in four scripts.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import chi2, norm, skewnorm

from copula import (
    GaussianCopulaSingleFactor,
    SkewTCopulaSingleFactor,
    TCopulaSingleFactor,
)
from gumbel_interplay import gumbel_conditional_inverse


@dataclass(frozen=True)
class FactorGrid:
    """Quadrature nodes, normalized nominal weights and conditional PDs."""

    weights: np.ndarray
    p_cond: np.ndarray
    z: np.ndarray
    lambda_: np.ndarray | None

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        p_cond = np.asarray(self.p_cond, dtype=float)
        if weights.ndim != 1 or p_cond.ndim != 2 or len(weights) != len(p_cond):
            raise ValueError("factor weights and conditional-PD matrix are incompatible")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("factor quadrature produced invalid weights")
        if not np.isclose(np.sum(weights), 1.0, atol=5e-12):
            raise ValueError("factor weights must be normalized")
        if np.any(~np.isfinite(p_cond)) or np.any((p_cond < 0) | (p_cond > 1)):
            raise ValueError("factor quadrature produced invalid conditional PDs")

    def expectation(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.weights):
            raise ValueError("one integrand value is required per factor node")
        return np.tensordot(self.weights, values, axes=(0, 0))


def _uniform_rule(order):
    nodes, weights = leggauss(int(order))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def build_factor_grid(copula, xi=1.0, n_gh=128, n_gl=48):
    """Build a probability-space quadrature using a Rosenblatt transform.

    Integrating over uniforms gives substantially better tail resolution than
    Hermite quadrature for high conditional quantiles.  In the dependent case
    it also avoids direct integration of the corner-singular Gumbel density.
    ``n_gh`` and ``n_gl`` are retained as argument names for API compatibility;
    they now denote the orders of the first and second uniform coordinates.
    """
    if xi < 1:
        raise ValueError("the Gumbel parameter xi must be at least one")
    if n_gh < 8 or n_gl < 8:
        raise ValueError("quadrature orders are too small for stable integration")

    u_nodes, z_weights = _uniform_rule(n_gh)

    if isinstance(copula, GaussianCopulaSingleFactor):
        z_nodes = norm.ppf(u_nodes)
        p_cond = copula.conditional_default_prob(z_nodes)
        return FactorGrid(z_weights, p_cond, z_nodes, None)

    if not isinstance(copula, (TCopulaSingleFactor, SkewTCopulaSingleFactor)):
        raise TypeError(f"unsupported copula class: {type(copula).__name__}")
    if not np.isclose(float(xi), float(copula.xi)):
        raise ValueError("factor-grid xi must match the copula calibration xi")

    if isinstance(copula, SkewTCopulaSingleFactor):
        z_nodes = skewnorm.ppf(u_nodes, copula.gamma)
    else:
        z_nodes = norm.ppf(u_nodes)
    conditional_nodes, lambda_weights = _uniform_rule(n_gl)
    uu, qq = np.meshgrid(u_nodes, conditional_nodes, indexing="ij")
    survival = gumbel_conditional_inverse(uu, qq, xi)
    lambda_grid = chi2.isf(survival, copula.nu)
    zz = np.broadcast_to(z_nodes[:, None], lambda_grid.shape)
    weights = (z_weights[:, None] * lambda_weights[None, :]).ravel()
    z_flat = zz.ravel()
    lambda_flat = lambda_grid.ravel()
    p_cond = copula.conditional_default_prob(z_flat, lambda_flat)
    return FactorGrid(weights, p_cond, z_flat, lambda_flat)
