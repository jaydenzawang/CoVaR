"""Plain Monte Carlo benchmark matching the paper's conditioning estimators."""

import numpy as np
import warnings
from scipy.stats import chi2, norm, skewnorm
from tqdm.auto import tqdm

import config
from gumbel_interplay import gumbel_conditional_inverse


class PlainMonteCarlo:
    def __init__(self, context, mode, bandwidth, direction="system", min_effective_tail_samples=None):
        if direction not in ("system", "exposure"):
            raise ValueError("direction must be system or exposure")
        self.min_effective_tail_samples = (config.BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES
                                           if min_effective_tail_samples is None else min_effective_tail_samples)
        self.direction = direction
        if mode not in ("equality", "inequality"):
            raise ValueError("mode must be 'equality' or 'inequality'")
        self.context = context
        self.mode = mode
        self.is_t_like = hasattr(context.copula, "nu")
        self.is_skew = hasattr(context.copula, "gamma")
        self.xi = float(getattr(context.copula, "xi", 1.0))
        self.bandwidth = float(bandwidth)
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

    def _sample_factor(self, n, rng):
        u = np.clip(rng.random(n), np.finfo(float).eps, 1 - np.finfo(float).eps)
        if self.is_skew:
            z = skewnorm.ppf(u, self.context.copula.gamma)
        else:
            z = norm.ppf(u)
        lambda_ = None
        if self.is_t_like:
            second = np.clip(
                rng.random(n), np.finfo(float).eps, 1 - np.finfo(float).eps
            )
            if self.xi > 1:
                survival = gumbel_conditional_inverse(u, second, self.xi)
                lambda_ = chi2.isf(survival, self.context.copula.nu)
            else:
                lambda_ = chi2.ppf(second, self.context.copula.nu)
        return z, lambda_

    def _conditioning_weight(self, target_loss, a):
        if self.mode == "equality":
            if a == 0:
                return (target_loss == 0).astype(float)
            return (
                np.abs(target_loss - a) <= self.bandwidth
            ).astype(float) / (2.0 * self.bandwidth)
        if a == 0:
            return np.ones_like(target_loss)
        return (target_loss >= a).astype(float)

    def estimate(
        self,
        n,
        a,
        beta,
        batch_size,
        seed,
        show_progress=False,
        progress_desc=None,
        progress_position=0,
        compute_es=True,
    ):
        if int(n) < 1 or int(batch_size) < 1 or not 0 < beta < 1:
            raise ValueError("n and batch_size must be positive and beta in (0,1)")
        rng = np.random.default_rng(seed)
        portfolio = self.context.portfolio
        losses, weights = [], []
        remaining = int(n)
        with tqdm(
            total=int(n),
            desc=progress_desc or f"PMC {self.mode} paths",
            unit="path",
            unit_scale=True,
            disable=not show_progress,
            leave=False,
            position=progress_position,
        ) as progress:
            while remaining:
                size = min(remaining, int(batch_size))
                z, lambda_ = self._sample_factor(size, rng)
                if not self.is_t_like:
                    p_cond = self.context.copula.conditional_default_prob(z)
                else:
                    p_cond = self.context.copula.conditional_default_prob(z, lambda_)
                defaults = rng.random(p_cond.shape) <= p_cond
                lgd = rng.beta(portfolio.a, portfolio.b, size=p_cond.shape)
                # Preserve RNG draws and arithmetic order, reuse the LGD buffer.
                lgd *= defaults
                lgd *= portfolio.u
                individual = lgd
                total = np.sum(individual, axis=1)
                selected = individual[:, self.context.target_i]
                if self.direction == "exposure":
                    total, selected = selected, total
                losses.append(total)
                weights.append(
                    self._conditioning_weight(
                        selected, a
                    )
                )
                remaining -= size
                progress.update(size)

        loss = np.concatenate(losses)
        weight = np.concatenate(weights)
        total_weight = np.sum(weight)
        if total_weight <= 0:
            raise RuntimeError("PMC produced no effective conditioning samples")
        # Zero conditioning weights never contribute to the quantile. Keep
        # normalization and ES sums on the original arrays, but sort only the
        # effective paths (all positive PMC weights are the same constant).
        active = np.flatnonzero(weight > 0)
        order = active[np.argsort(loss[active])]
        cumulative = np.cumsum(weight[order])
        index = np.searchsorted(cumulative, beta * total_weight, side="left")
        covar = float(loss[order[min(index, len(order) - 1)]])
        coes = None
        if compute_es:
            stop_loss = np.sum(weight * np.maximum(loss - covar, 0.0)) / total_weight
            coes = float(covar + stop_loss / (1.0 - beta))
        ess = total_weight**2 / np.sum(weight**2)
        if (
            ess * (1.0 - beta)
            < self.min_effective_tail_samples
        ):
            warnings.warn(
                "effective tail sample size is below the configured minimum; "
                "increase PMC_NUM_SIMS or use the result only as a smoke test",
                RuntimeWarning,
                stacklevel=2,
            )
        return covar, coes, float(ess)
