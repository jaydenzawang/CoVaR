#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional saddlepoint approximations for portfolio CDFs and stop-loss moments.
"""

import numpy as np
from scipy.stats import norm, beta as beta_dist
from scipy.optimize import brentq

class SaddlepointApproximation:
    def __init__(
        self,
        portfolio,
        target_i=None,
        saddle_bound=20.0,
        root_tol=1e-10,
        mean_region=1e-2,
        tail_bound_tol=1e-12,
    ):
        self.portfolio = portfolio
        self.target_i = target_i
        self.saddle_bound = float(saddle_bound)
        self.root_tol = float(root_tol)
        self.mean_region = float(mean_region)
        self.tail_bound_tol = float(tail_bound_tol)
        if not (0 < self.saddle_bound <= min(abs(portfolio.s_grid[0]), portfolio.s_grid[-1])):
            raise ValueError("saddle_bound must lie inside the portfolio interpolation grid")
        if not 0 < self.mean_region <= 0.1:
            raise ValueError("mean_region must lie in (0, 0.1]")
        if not 0 < self.tail_bound_tol < 1e-4:
            raise ValueError("tail_bound_tol must lie in (0, 1e-4)")

        self.mask = np.ones(len(self.portfolio.u), dtype=bool)
        if self.target_i is not None:
            if not 0 <= self.target_i < len(self.mask):
                raise IndexError("target_i is outside the portfolio")
            self.mask[self.target_i] = False
        self.support_max = float(np.sum(self.portfolio.u[self.mask]))
        self._cgf_terms_cache = None

    # ------------------------------------------------------------------
    def cgf_deriv(self, s, p_cond, k):
        """Use vector operations to compute CGF derivatives fast"""
        p_cond = np.asarray(p_cond, dtype=float)
        if p_cond.shape != self.portfolio.u.shape:
            raise ValueError("p_cond must have one entry per obligor")
        if np.any((p_cond < 0) | (p_cond > 1)):
            raise ValueError("conditional default probabilities must lie in [0, 1]")
        with np.errstate(over='ignore', invalid='ignore'):
            # Consecutive cumulants at the same saddle share A and A^(j).
            # Exact keys only; copying p detects callers mutating an array.
            cache = self._cgf_terms_cache
            if cache is None or cache[0] != s or not np.array_equal(cache[1], p_cond):
                cache = (s, p_cond.copy(), [self.portfolio.A_j(s, p_cond)[self.mask]])
                self._cgf_terms_cache = cache
            terms = cache[2]
            for order in range(len(terms), k + 1):
                terms.append(self.portfolio.A_j_deriv(s, p_cond, order)[self.mask])
            A = terms[0]

            if k == 0:
                return np.sum(np.log(A))

            A1 = terms[1]
            if k == 1:
                return np.sum(A1 / A)

            A2 = terms[2]
            if k == 2:
                return np.sum(A2 / A - (A1 / A)**2)

            A3 = terms[3]
            if k == 3:
                return np.sum(
                    A3 / A
                    - 3 * A2 * A1 / (A**2)
                    + 2 * (A1 / A)**3
                )

            A4 = terms[4]
            if k == 4:
                return np.sum(
                    A4 / A
                    - 4 * A3 * A1 / (A**2)
                    - 3 * (A2 / A)**2
                    + 12 * A2 * (A1**2) / (A**3)
                    - 6 * (A1 / A)**4
                )

    # ------------------------------------------------------------------
    def find_saddlepoint(self, y, p_cond):
        """Solve K'(s)=y with a bracketed method and a checked residual.

        Returns ``None`` only when the requested point is so deep in a tail that
        the validated interpolation interval cannot bracket its saddlepoint.
        """
        mu0 = self.cgf_deriv(0.0, p_cond, 1)
        if abs(y - mu0) <= self.root_tol:
            return 0.0
        bound = self.saddle_bound
        lo, hi = (-bound, 0.0) if y < mu0 else (0.0, bound)
        f_lo = self.cgf_deriv(lo, p_cond, 1) - y
        f_hi = self.cgf_deriv(hi, p_cond, 1) - y
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return None
        root = brentq(
            lambda s: self.cgf_deriv(s, p_cond, 1) - y,
            lo,
            hi,
            xtol=self.root_tol,
            rtol=4 * np.finfo(float).eps,
            maxiter=100,
        )
        residual = abs(self.cgf_deriv(root, p_cond, 1) - y)
        if residual > max(1e-8, self.root_tol * max(1.0, abs(y))):
            raise RuntimeError(f"saddlepoint residual is too large: {residual:g}")
        return root

    def _chernoff_tail_bound(self, y, p_cond):
        """Bound the missing tail when the finite saddle bracket is exhausted."""
        mean = self.cgf_deriv(0.0, p_cond, 1)
        s = self.saddle_bound if y > mean else -self.saddle_bound
        log_bound = self.cgf_deriv(s, p_cond, 0) - s * y
        return min(float(np.exp(min(log_bound, 0.0))), 1.0)

    # ------------------------------------------------------------------
    def prob_L_minus_i(self, y, p_cond):
        """Compute the conditional CDF via SPA (with an asymptotic form for the far tail)"""
        if y < 0:
            return 0.0
        if y == 0:
            return np.prod(1.0 - p_cond[self.mask])
        if y >= self.support_max:
            return 1.0

        s_hat = self.find_saddlepoint(y, p_cond)
        if s_hat is None:
            bound = self._chernoff_tail_bound(y, p_cond)
            mean = self.cgf_deriv(0.0, p_cond, 1)
            if bound <= self.tail_bound_tol:
                return 1.0 if y > mean else 0.0
            raise RuntimeError(
                "the saddlepoint lies outside the validated interval and the "
                f"Chernoff remainder bound {bound:.3g} exceeds tolerance; "
                "increase saddle_bound and the portfolio interpolation range"
            )

        # ---------------------------------------------------------------
        # Mean-region Taylor evaluation avoids cancellation in the LR terms.
        #
        # The exact limit of the LR bracket as ŝ→0 is λ₃(0)/6, where
        #   λ₃(0) = K'''(0) / K''(0)^{3/2}
        # Thus the Taylor form is:
        #   F(y) ≈ Φ(z) + φ(z) · λ₃(0)/6，  z = (y - μ) / √κ₂
        # The gap from the full formula at the boundary is only O(ŝ), for a smooth join.
        # ---------------------------------------------------------------
        if abs(s_hat) < self.mean_region:
            K2_0 = self.cgf_deriv(0, p_cond, 2)
            if K2_0 <= 1e-12 or np.isnan(K2_0):
                return 0.5
            K3_0   = self.cgf_deriv(0, p_cond, 3)
            mean   = self.cgf_deriv(0, p_cond, 1)
            z      = (y - mean) / np.sqrt(K2_0)
            lam3_0 = K3_0 / (K2_0 ** 1.5)
            return np.clip(norm.cdf(z) + norm.pdf(z) * (lam3_0 / 6.0), 0.0, 1.0)

        K0 = self.cgf_deriv(s_hat, p_cond, 0)
        K2 = self.cgf_deriv(s_hat, p_cond, 2)

        if K2 <= 1e-12 or np.isnan(K2):
            return 1.0 if y > self.cgf_deriv(0, p_cond, 1) else 0.0

        w_hat = np.sign(s_hat) * np.sqrt(2 * max(s_hat * y - K0, 0))
        v_hat = s_hat * np.sqrt(K2)

        if abs(w_hat) < 1e-6 or abs(v_hat) < 1e-6:
            return 1.0 if y > self.cgf_deriv(0, p_cond, 1) else 0.0

        K3   = self.cgf_deriv(s_hat, p_cond, 3)
        K4   = self.cgf_deriv(s_hat, p_cond, 4)
        lam3 = K3 / (K2 ** 1.5)
        lam4 = K4 / (K2 ** 2.0)

        # Fourth-order SPA tail formula in the paper (eq. SPA F)
        bracket = (1.0 / w_hat - 1.0 / v_hat
                   - 1.0 / w_hat**3 + 1.0 / v_hat**3
                   + lam3 / (2.0 * v_hat**2)
                   - (1.0 / v_hat) * (lam4 / 8.0 - 5.0 * lam3**2 / 24.0))
        return np.clip(norm.cdf(w_hat) + norm.pdf(w_hat) * bracket, 0.0, 1.0)

    def density_upper_bound(self, y, p_cond, s):
        """
        Exponential bound on the continuous density (beta shapes >= 1).
        """
        a, b, u = (x[self.mask] for x in (self.portfolio.a, self.portfolio.b, self.portfolio.u))
        if np.any(a < 1) or np.any(b < 1):
            return np.inf
        lo, hi = np.zeros_like(u), u.copy()
        for _ in range(60):
            x = (lo + hi) / 2
            left = np.divide(a-1, x, out=np.zeros_like(x), where=a != 1)
            right = np.divide(b-1, u-x, out=np.zeros_like(x), where=b != 1)
            slope = s + left - right
            lo = np.where(slope > 0, x, lo)
            hi = np.where(slope > 0, hi, x)
        x = (lo + hi) / 2
        log_sup = s*x + beta_dist.logpdf(x/u, a, b) - np.log(u)
        # Include endpoints for beta shapes equal to one.
        log_sup = np.maximum(log_sup, beta_dist.logpdf(0, a, b)-np.log(u))
        log_sup = np.maximum(log_sup, s*u+beta_dist.logpdf(1, a, b)-np.log(u))
        p = p_cond[self.mask]
        A = self.portfolio.A_j(s, p_cond)[self.mask]
        from scipy.special import logsumexp
        with np.errstate(divide="ignore"):
            log_bound = self.cgf_deriv(s,p_cond,0)-s*y + logsumexp(np.log(p)+log_sup-np.log(A))
        return float(np.exp(min(log_bound, 700)))

    def density(self, y, p_cond):
        """
        Second-order Daniels density for the continuous positive-loss part.
        """
        if y <= 0 or y >= self.support_max:
            return 0.0
        s = self.find_saddlepoint(y, p_cond)
        if s is None:
            mean = self.cgf_deriv(0, p_cond, 1)
            edge = self.saddle_bound if y > mean else -self.saddle_bound
            if self.density_upper_bound(y, p_cond, edge) <= self.tail_bound_tol:
                return 0.0
            raise RuntimeError("density saddle outside interpolation range; increase saddle_bound and MGF range")
        k2 = self.cgf_deriv(s, p_cond, 2)
        k3 = self.cgf_deriv(s, p_cond, 3)
        k4 = self.cgf_deriv(s, p_cond, 4)
        correction = 1 + k4 / (8*k2**2) - 5*k3**2 / (24*k2**3)
        value = np.exp(self.cgf_deriv(s, p_cond, 0) - s*y) * correction / np.sqrt(2*np.pi*k2)
        if not np.isfinite(value) or value < 0:
            raise FloatingPointError("invalid second-order Daniels density")
        return float(value)

    # ------------------------------------------------------------------
    def stop_loss_L_minus_i(self, y, p_cond):
        """
        Approximate E[(L_{-i}-y)^+ | Theta] directly.
        """
        p_cond = np.asarray(p_cond, dtype=float)
        mu_cond = np.sum(
            (self.portfolio.u * self.portfolio.expected_epsilon * p_cond)[self.mask]
        )
        if y < 0:
            return float(mu_cond - y)
        if y == 0:
            return float(mu_cond)
        if y >= self.support_max:
            return 0.0

        s_hat = self.find_saddlepoint(y, p_cond)
        if s_hat is None:
            bound = self._chernoff_tail_bound(y, p_cond)
            if y > mu_cond and (self.support_max - y) * bound <= self.tail_bound_tol:
                return 0.0
            if y < mu_cond and y * bound <= self.tail_bound_tol:
                return max(float(mu_cond - y), 0.0)
            raise RuntimeError(
                "the saddlepoint lies outside the validated interval and the "
                f"stop-loss remainder bound based on {bound:.3g} exceeds "
                "tolerance; increase saddle_bound and the interpolation range"
            )

        if abs(s_hat) < self.mean_region:
            k2 = self.cgf_deriv(0.0, p_cond, 2)
            if k2 <= 1e-12 or not np.isfinite(k2):
                raise RuntimeError("conditional variance is non-positive near the mean")
            k3 = self.cgf_deriv(0.0, p_cond, 3)
            k4 = self.cgf_deriv(0.0, p_cond, 4)
            mean_limit = norm.pdf(0.0) * (
                np.sqrt(k2)
                - k4 / (24.0 * k2**1.5)
                + k3**2 / (24.0 * k2**2.5)
            )
            return max(float(mean_limit), 0.0)

        k0 = self.cgf_deriv(s_hat, p_cond, 0)
        k2 = self.cgf_deriv(s_hat, p_cond, 2)
        if k2 <= 1e-12 or not np.isfinite(k2):
            raise RuntimeError("conditional tilted variance is non-positive")

        w_hat = np.sign(s_hat) * np.sqrt(2.0 * max(s_hat * y - k0, 0.0))
        v_hat = s_hat * np.sqrt(k2)
        if abs(w_hat) < 1e-8 or abs(v_hat) < 1e-8:
            raise RuntimeError("unstable stop-loss evaluation outside the mean branch")
        stop_loss = (mu_cond - y) * (1.0 - norm.cdf(w_hat)) + norm.pdf(
            w_hat
        ) * (
            (mu_cond - y) * (w_hat**-3 - w_hat**-1)
            + 1.0 / (v_hat * s_hat)
        )
        return max(float(stop_loss), 0.0)
