"""SPA for the marked law L | L_i <= y in the draft's exposure formulas."""

import numpy as np
from scipy.special import roots_jacobi, betainc

from spa import SaddlepointApproximation


def marked_lgd_rule(portfolio, i, y, order):
    """Unnormalized moments on [0,y], with the beta singularity integrated out.

    Gauss-Jacobi absorbs x**(a-1). Normalization uses the exact beta CDF;
    increasing order controls the remaining smooth integrand error.
    """
    a, b, u = portfolio.a[i], portfolio.b[i], portfolio.u[i]
    mass = float(betainc(a, b, y / u))
    if y == 0:
        return np.zeros(1), np.zeros(1)
    t, w = roots_jacobi(order, 0.0, a - 1.0)
    x = y * (t + 1.0) / 2.0
    w = w * (1.0 - x / u) ** (b - 1.0)
    return x, mass * w / np.sum(w)


class MarkedSaddlepoint(SaddlepointApproximation):
    def __init__(self, base, y, nodes, weights):
        # Share the immutable portfolio and SPA options with the minus-i engine.
        self.__dict__.update(base.__dict__)
        self.nodes, self.weights = nodes, weights
        self.mass = float(np.sum(weights))
        self.support_max = base.support_max + y

    def prob_L_minus_i(self, y, p_cond):
        if y == 0:
            p = p_cond[self.target_i]
            fi = 1-p+p*self.mass
            return float(np.prod(1-p_cond[self.mask])*(1-p)/fi)
        return super().prob_L_minus_i(y, p_cond)

    def cgf_deriv(self, s, p_cond, k):
        p = p_cond[self.target_i]
        fi = 1.0 - p + p * self.mass
        if fi <= 0:
            raise FloatingPointError("marked event has zero probability")
        terms = p * self.weights * np.exp(s * self.nodes)
        h = 1.0 - p + np.sum(terms)
        if k == 0:
            marked = np.log(h / fi)
        else:
            m = [None] + [np.dot(terms, self.nodes**j) / h for j in range(1, k + 1)]
            if k == 1:
                marked = m[1]
            elif k == 2:
                marked = m[2] - m[1]**2
            elif k == 3:
                marked = m[3] - 3*m[2]*m[1] + 2*m[1]**3
            else:
                marked = m[4] - 4*m[3]*m[1] - 3*m[2]**2 + 12*m[2]*m[1]**2 - 6*m[1]**4
        return super().cgf_deriv(s, p_cond, k) + marked

    def density_upper_bound(self, y, p_cond, s):
        # Joint marked density is bounded by the unmarked density; normalize
        # by P(L_i<=y_mark | Theta) to bound the marked conditional law.
        full = SaddlepointApproximation(self.portfolio, saddle_bound=self.saddle_bound)
        fi = 1-p_cond[self.target_i]+p_cond[self.target_i]*self.mass
        return full.density_upper_bound(y, p_cond, s) / fi
