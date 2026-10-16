#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:28:08 2026

@author: jaydenwang
"""


import numpy as np

def gumbel_copula_pdf(u, v, xi):
    """
    Compute the Gumbel copula joint density (PDF) at (u, v) with parameter xi
    """
    if xi == 1.0:
        return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
        
    # Clip values to avoid NaN from log(0) or log(1)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)
    
    x = -np.log(u)
    y = -np.log(v)
    
    A = x**xi + y**xi
    # Gumbel CDF formula (kept for reference or checks)
    # C = np.exp(-A**(1.0 / xi))
    
    # Exact expansion of the standard Gumbel PDF
    term_C = np.exp(-A**(1.0 / xi))
    term_uv = u * v
    term_A1 = A**(1.0 / xi + -2.0)
    term_xy = (x * y)**(xi - 1.0)
    term_bracket = A**(1.0 / xi) + xi - 1.0
    
    pdf = (term_C / term_uv) * term_A1 * term_xy * term_bracket
    return pdf


def gumbel_conditional_inverse(u, probability, xi, iterations=64):
    """Invert ``P(V<=v | U=u)=dC(u,v)/du`` in a stable log domain."""
    if xi < 1:
        raise ValueError("xi must be at least one")
    if xi == 1.0:
        return np.asarray(probability, dtype=float)
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1 - 1e-15)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    x = -np.log(u)
    log_x = np.log(x)
    log_target = np.log(probability)

    def log_conditional_cdf(y):
        log_y = np.log(y)
        high = np.maximum(xi * log_x, xi * log_y)
        log_a = high + np.log(
            np.exp(xi * log_x - high) + np.exp(xi * log_y - high)
        )
        return (
            -np.exp(log_a / xi)
            + x
            + (xi - 1.0) * log_x
            + (1.0 / xi - 1.0) * log_a
        )

    lower = np.full(np.broadcast_shapes(u.shape, probability.shape), 1e-14)
    upper = np.full_like(lower, 800.0)
    for _ in range(iterations):
        midpoint = np.sqrt(lower * upper)
        move_right = log_conditional_cdf(midpoint) > log_target
        lower = np.where(move_right, midpoint, lower)
        upper = np.where(move_right, upper, midpoint)
    return np.exp(-np.sqrt(lower * upper))
