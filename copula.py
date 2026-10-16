#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:10 2026

@author: jaydenwang

Single-factor dependence models with support for Gumbel-linked factors.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm, t, chi2, skewnorm
from scipy.optimize import root_scalar

# The same Rosenblatt transform is used for threshold calibration and for the production factor integral.  
# This avoids a model-changing mismatch between a density-weighted Hermite--Laguerre grid and a probability-space grid.
from gumbel_interplay import gumbel_conditional_inverse


def _uniform_rule(order):
    nodes, weights = leggauss(int(order))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _calibrate_dependent_thresholds(
    rho,
    marginal_pd,
    nu,
    xi,
    z_ppf,
    n_z,
    n_lambda,
):
    """Calibrate default thresholds on the production Rosenblatt measure."""
    if int(n_z) < 8 or int(n_lambda) < 8:
        raise ValueError("threshold-calibration orders must be at least eight")
    u_nodes, u_weights = _uniform_rule(n_z)
    conditional_nodes, conditional_weights = _uniform_rule(n_lambda)
    uu, qq = np.meshgrid(u_nodes, conditional_nodes, indexing="ij")
    survival = gumbel_conditional_inverse(uu, qq, xi)
    lambda_grid = chi2.isf(survival, nu)
    z_grid = np.broadcast_to(z_ppf(u_nodes)[:, None], lambda_grid.shape)
    weights = (u_weights[:, None] * conditional_weights[None, :]).ravel()
    z_flat = z_grid.ravel()
    sqrt_lambda = np.sqrt(lambda_grid.ravel() / nu)

    thresholds = np.empty_like(np.asarray(marginal_pd, dtype=float))
    for i, (rho_i, p_i) in enumerate(zip(rho, marginal_pd)):
        denominator = np.sqrt(1.0 - rho_i**2)

        def unconditional_prob(threshold):
            argument = (
                rho_i * z_flat - threshold * sqrt_lambda
            ) / denominator
            return np.dot(weights, norm.cdf(np.clip(argument, -10.0, 10.0)))

        result = root_scalar(
            lambda threshold: unconditional_prob(threshold) - p_i,
            bracket=(-10.0, 10.0),
            method="brentq",
            xtol=1e-12,
        )
        thresholds[i] = result.root
    return thresholds

class GaussianCopulaSingleFactor:
    """
    Single-factor Gaussian copula model
    Note: The Gaussian model has no shock factor Lambda, so xi has no effect. Keep it as the benchmark.
    """
    def __init__(self, rho, p):
        self.rho = np.array(rho)
        self.p = np.array(p)
        # Default threshold x_i^d = \Phi^{-1}(1 - p_i)
        self.x_d = norm.ppf(1 - self.p)
        
    
    def conditional_default_prob(self, Z):
        """
        Compute the default probability given the single risk factor Z: p_{i|\Theta}
        """
        Z = np.asarray(Z)
        
        # Support vector input (e.g., 2000 paths per call from the MC engine)
        if Z.ndim > 0:
            num = np.outer(Z, self.rho) - self.x_d
        else:
            # Support scalar input (e.g., one node per call from the quadrature engine)
            num = self.rho * Z - self.x_d
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
class TCopulaSingleFactor:
    """
    Single-factor t-copula model
    At xi = 1, use the standard t-copula with independent factors and a fast closed-form ppf setup;
    at xi > 1, use a 2D grid and the Gumbel joint density to solve for threshold x_d.
    """
    def __init__(
        self,
        rho,
        p,
        nu,
        xi=1.0,
        calibration_n_z=64,
        calibration_n_lambda=64,
    ):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.x_d = np.zeros_like(self.p)
        
        self.xi = float(xi)
        if self.xi < 1.0:
            raise ValueError("xi must be at least one")
        
        if self.xi == 1.0:
            # With independent factors, the asset margin is standard t; use the direct closed-form solution.
            self.x_d = t.ppf(1 - self.p, df=self.nu)
        else:
            self.x_d = _calibrate_dependent_thresholds(
                self.rho,
                self.p,
                self.nu,
                self.xi,
                norm.ppf,
                calibration_n_z,
                calibration_n_lambda,
            )
        
    def conditional_default_prob(self, Z, Lambda):
        """
        Compute the default probability given Z and Lambda
        Defaults are independent given the factors; keep the same form.
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        sqrt_term = np.sqrt(Lambda / self.nu)
        
        if Z.ndim > 0 and Lambda.ndim > 0:
            Z_term = np.outer(Z, self.rho)
            Lambda_term = np.outer(sqrt_term, self.x_d)
            num = Z_term - Lambda_term
        else:
            num = self.rho * Z - self.x_d * sqrt_term
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))


class SkewTCopulaSingleFactor:
    """
    Single-factor Skew-t copula model (updated)
    Systematic risk factor Z ~ Skew-Normal(0, 1, gamma)
    Common shock factor Lambda ~ Chi^2(nu)
    Link the two factors with a Gumbel copula with parameter xi
    """
    def __init__(
        self,
        rho,
        p,
        nu,
        gamma_val,
        xi=1.0,
        calibration_n_z=64,
        calibration_n_lambda=64,
    ):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.gamma = gamma_val
        self.x_d = np.zeros_like(self.p)
        self.xi = float(xi)
        if self.xi < 1.0:
            raise ValueError("xi must be at least one")
        
        self.x_d = _calibrate_dependent_thresholds(
            self.rho,
            self.p,
            self.nu,
            self.xi,
            lambda probability: skewnorm.ppf(probability, self.gamma),
            calibration_n_z,
            calibration_n_lambda,
        )

    
    
    def conditional_default_prob(self, Z, Lambda):
        """
        Compute the default probability given Z and Lambda
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        sqrt_term = np.sqrt(Lambda / self.nu)
        
        # Support vector input
        if Z.ndim > 0 and Lambda.ndim > 0:
            Z_term = np.outer(Z, self.rho)
            Lambda_term = np.outer(sqrt_term, self.x_d)
            num = Z_term - Lambda_term
        else:
            num = self.rho * Z - self.x_d * sqrt_term
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
    
    
