#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:23 2026

@author: jaydenwang

Portfolio loss model with the MGF of Beta LGD.
"""

import numpy as np
from scipy.special import hyp1f1, poch
from scipy.interpolate import CubicSpline

class PortfolioLoss:
    """
    Portfolio loss model with a spline-based Beta MGF and exact values near the mean.
    """
    def __init__(self, u, a, b, s_min=-25.0, s_max=25.0, grid_size=1501, exact_mgf_radius=0.15):
        self.u = np.asarray(u, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        if not (self.u.ndim == self.a.ndim == self.b.ndim == 1):
            raise ValueError("u, a and b must be one-dimensional arrays")
        if not (len(self.u) == len(self.a) == len(self.b)):
            raise ValueError("u, a and b must have the same length")
        if np.any(self.u <= 0) or np.any(self.a <= 0) or np.any(self.b <= 0):
            raise ValueError("exposures and beta shape parameters must be positive")
        if not (s_min < 0 < s_max) or grid_size < 100:
            raise ValueError("the interpolation grid must contain zero and be sufficiently dense")
        

        self.s_grid = np.linspace(float(s_min), float(s_max), int(grid_size))

        self.s_grid[np.argmin(np.abs(self.s_grid))] = 0.0
        
        # Precompute confluent hypergeometric functions for all firms for CGF orders 0 to 4 used by spa.py.
        # All result matrices have shape (1000, N), keeping each firm's own exposure and parameters.
        hyp_k0 = np.array([hyp1f1(self.a, self.a + self.b, s * self.u) for s in self.s_grid])
        self.spline_k0 = CubicSpline(self.s_grid, hyp_k0, axis=0, extrapolate=False)
        
        hyp_k1 = np.array([hyp1f1(self.a + 1, self.a + self.b + 1, s * self.u) for s in self.s_grid])
        self.spline_k1 = CubicSpline(self.s_grid, hyp_k1, axis=0, extrapolate=False)
        
        hyp_k2 = np.array([hyp1f1(self.a + 2, self.a + self.b + 2, s * self.u) for s in self.s_grid])
        self.spline_k2 = CubicSpline(self.s_grid, hyp_k2, axis=0, extrapolate=False)
        
        hyp_k3 = np.array([hyp1f1(self.a + 3, self.a + self.b + 3, s * self.u) for s in self.s_grid])
        self.spline_k3 = CubicSpline(self.s_grid, hyp_k3, axis=0, extrapolate=False)
        
        hyp_k4 = np.array([hyp1f1(self.a + 4, self.a + self.b + 4, s * self.u) for s in self.s_grid])
        self.spline_k4 = CubicSpline(self.s_grid, hyp_k4, axis=0, extrapolate=False)
        
        self.exact_mgf_radius = float(exact_mgf_radius)
        if not np.isfinite(self.exact_mgf_radius) or self.exact_mgf_radius <= 0:
            raise ValueError("exact_mgf_radius must be finite and positive")
        self._max_exposure = float(np.max(self.u))
        self.expected_epsilon = self.a / (self.a + self.b)
        # Parameters are fixed for this model context. Keep multiplication order
        # unchanged: coefficient * spline value * exposure power.
        self._derivative_coef = {k: poch(self.a, k) / poch(self.a + self.b, k)
                                 for k in (1, 2, 3, 4)}
        self._exposure_power = {k: self.u ** k for k in (1, 2, 3, 4)}
        
    def mgf_epsilon(self, s):
        """Compute the MGF; call hyp1f1 for scalar saddlepoints near the mean."""
        self._validate_s(s)

        if np.ndim(s) == 0 and abs(s) * self._max_exposure < self.exact_mgf_radius:
            return hyp1f1(self.a, self.a + self.b, s * self.u)
        return self.spline_k0(s)
        
    def mgf_epsilon_deriv(self, s, k=1):
        """Compute derivatives using the same method as the MGF, with the exposure chain rule."""
        self._validate_s(s)
        if k not in (1, 2, 3, 4):
            raise ValueError("only derivatives of order 1 through 4 are supported")
        coef = self._derivative_coef[k]
        if np.ndim(s) == 0 and abs(s) * self._max_exposure < self.exact_mgf_radius:
            return coef * hyp1f1(self.a + k, self.a + self.b + k, s * self.u) * self._exposure_power[k]
        if k == 1:
            return coef * self.spline_k1(s) * self.u
        elif k == 2:
            return coef * self.spline_k2(s) * self._exposure_power[2]
        elif k == 3:
            return coef * self.spline_k3(s) * self._exposure_power[3]
        elif k == 4:
            return coef * self.spline_k4(s) * self._exposure_power[4]
        raise AssertionError("unreachable derivative order")

    def _validate_s(self, s):
        values = np.asarray(s)
        if np.any(values < self.s_grid[0]) or np.any(values > self.s_grid[-1]):
            raise ValueError(
                f"s={s} lies outside the validated interpolation interval "
                f"[{self.s_grid[0]}, {self.s_grid[-1]}]"
            )

    def A_j(self, s, p_cond):

        return 1 - p_cond + p_cond * self.mgf_epsilon(s)
        
    def A_j_deriv(self, s, p_cond, k):
 
        return p_cond * self.mgf_epsilon_deriv(s, k)
