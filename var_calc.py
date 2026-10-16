#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:33:54 2026

@author: jaydenwang

var_calc.py

Functions:
1. Compute the exact unconditional VaR_alpha of one obligor/firm.
2. Use total-probability quadrature to check margins of each copula (Gaussian / t / Skew-t) with Gumbel-linked factors (XI >= 1).
"""

import numpy as np
from scipy.stats import beta as scipy_beta
from scipy.stats import norm, chi2, skewnorm
from scipy.integrate import quad, dblquad

# Import settings and the Gumbel link function
from gumbel_interplay import gumbel_copula_pdf
# Import all copula models to handle all types
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor, SkewTCopulaSingleFactor

def calculate_individual_var(portfolio, copula, idx, alpha):
    """
    Compute the unconditional VaR_alpha of one firm (closed-form version)
    The loss law mixes a point mass with a continuous Beta law, so this closed-form method is exact.
    Note: This marginal closed-form solution also holds for XI > 1, since threshold x_d is already calibrated.
    """
    u_i = portfolio.u[idx]
    a_i = portfolio.a[idx]
    b_i = portfolio.b[idx]
    
    if hasattr(copula, 'p'):
        p_i = copula.p[idx]
    else:
        raise ValueError("The copula object has no unconditional default probability array 'p'.")
        
    prob_no_default = 1.0 - p_i
    
    # Boundary guard: VaR is exactly 0 if alpha is below the no-default probability
    if alpha <= prob_no_default:
        return 0.0
        
    # Map to the truncated Beta percentile given default
    target_prob = (alpha - prob_no_default) / p_i
    epsilon_quantile = scipy_beta.ppf(target_prob, a_i, b_i)
    
    return u_i * epsilon_quantile


def verify_copula_marginal(copula, idx):
    """
    Check for both research and math accuracy
    Use adaptive total-probability quadrature over risk factors to compute the marginal default probability.
    Also check the change in the joint density of Z and Lambda from the Gumbel copula when XI > 1.
    """
    # Use the parameter with which this copula was actually calibrated.
    xi_val = float(getattr(copula, "xi", 1.0))

    # Case A: Single-factor Gaussian copula (1D factor integral, no Lambda, no effect from XI)
    if isinstance(copula, GaussianCopulaSingleFactor):
        def integrand_gauss(Z):
            return copula.conditional_default_prob(Z)[idx] * norm.pdf(Z)
        
        integrated_p, _ = quad(integrand_gauss, -np.inf, np.inf)
        return integrated_p
        
    # Case B: Single-factor t-copula (2D factor integral with the Gumbel joint measure)
    elif isinstance(copula, TCopulaSingleFactor):
        def integrand_t(Z, Lambda):
            p_cond = copula.conditional_default_prob(Z, Lambda)[idx]
            pdf_L = chi2.pdf(Lambda, copula.nu)
            pdf_Z = norm.pdf(Z)
            
            # Compute the Gumbel density adjustment at this point
            u_val = norm.cdf(Z)
            v_val = chi2.sf(Lambda, copula.nu)
            gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
            
            # Multiply the joint measure by the dependence adjustment
            return p_cond * pdf_Z * pdf_L * gumbel_factor
            
        integrated_p, _ = dblquad(
            integrand_t, 0, np.inf,
            lambda lam: -np.inf, lambda lam: np.inf
        )
        return integrated_p

    # Case C: Single-factor Skew-t copula (2D factor integral with skew-normal and Gumbel joint measures)
    elif isinstance(copula, SkewTCopulaSingleFactor):
        def integrand_skew_t(Z, Lambda):
            p_cond = copula.conditional_default_prob(Z, Lambda)[idx]
            pdf_L = chi2.pdf(Lambda, copula.nu)
            # Use the single-factor Skew-Normal marginal density:2 * \phi(z) * \Phi(\gamma * z)
            pdf_Z = norm.pdf(Z) * 2.0 * norm.cdf(copula.gamma * Z)
            
            # Compute the Gumbel density adjustment at this point
            # Note: The Z axis must use the 1D skew-normal CDF skewnorm.cdf
            u_val = skewnorm.cdf(Z, copula.gamma)
            v_val = chi2.sf(Lambda, copula.nu)
            gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
            
            # Multiply the joint measure by the dependence adjustment
            return p_cond * pdf_Z * pdf_L * gumbel_factor
            
        integrated_p, _ = dblquad(
            integrand_skew_t, 0, np.inf,
            lambda lam: -np.inf, lambda lam: np.inf
        )
        return integrated_p
        
    else:
        raise TypeError("Unknown copula type; cannot check margins with total probability.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
