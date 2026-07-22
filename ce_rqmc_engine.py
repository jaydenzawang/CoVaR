"""Faithful CE-RQMC benchmark for conditional CoVaR and robust CoES.

Unlike the Hybrid-SPA engine, this benchmark simulates the target obligor and
uses the conditioning weights stated in the paper.  Equality conditioning at a
positive loss uses a uniform kernel; equality at zero uses the exact non-default
atom; inequality at zero is the unconditional portfolio.
"""

import numpy as np
import warnings
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, norm, skewnorm
from scipy.stats.qmc import Sobol
from tqdm.auto import tqdm

import config
from gumbel_interplay import gumbel_conditional_inverse


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


class IterativeCE_RQMC:
    def __init__(
        self,
        portfolio,
        copula,
        target_i,
        mode,
        bandwidth,
        calibration_bandwidth=None,
    ):
        if mode not in ("equality", "inequality"):
            raise ValueError("mode must be 'equality' or 'inequality'")
        self.portfolio = portfolio
        self.copula = copula
        self.target_i = int(target_i)
        self.mode = mode
        self.is_t_like = hasattr(copula, "nu")
        self.is_skew = hasattr(copula, "gamma")
        self.xi = float(getattr(copula, "xi", 1.0))
        self.nu = float(getattr(copula, "nu", np.nan))
        self.gamma = float(getattr(copula, "gamma", 0.0))
        self.n_obligors = len(portfolio.u)
        # ``bandwidth`` is the single final-estimation bandwidth.  Calibration
        # may use a wider band to obtain a stable proposal without changing the
        # final self-normalized estimator's target.
        self.bandwidth = float(bandwidth)
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if calibration_bandwidth is None:
            calibration_bandwidth = self.bandwidth
        self.calibration_bandwidth = float(calibration_bandwidth)
        if self.calibration_bandwidth < self.bandwidth:
            raise ValueError(
                "calibration_bandwidth must be at least the final bandwidth"
            )
        self.factor_dimension = 1 + int(self.is_t_like)
        self.dimension = self.factor_dimension + 2 * self.n_obligors

    @staticmethod
    def _defensive_likelihood_ratio(log_pure_lr, defensive_probability):
        """Return f / [delta f + (1-delta) g] without overflow."""
        delta = float(defensive_probability)
        if not 0.0 < delta < 1.0:
            raise ValueError("defensive_probability must lie in (0, 1)")
        log_pure_lr = np.asarray(log_pure_lr, dtype=float)
        log_mixture_over_nominal = np.logaddexp(
            np.log(delta), np.log1p(-delta) - log_pure_lr
        )
        log_defensive_lr = np.clip(
            -log_mixture_over_nominal, -745.0, np.log(1.0 / delta)
        )
        return np.exp(log_defensive_lr)

    def _proposal_paths(
        self, n, mu, kappa, seed, defensive_probability=None
    ):
        """Generate pure-CE pilot paths or defensive-mixture final paths."""
        if not _is_power_of_two(int(n)):
            raise ValueError("RQMC sample size must be a power of two")
        if not (0 < kappa <= 1):
            raise ValueError("kappa must lie in (0, 1]")
        defensive = defensive_probability is not None
        if defensive:
            delta = float(defensive_probability)
            if not 0.0 < delta < 1.0:
                raise ValueError("defensive_probability must lie in (0, 1)")
        else:
            delta = None
        rqmc_dimension = self.dimension + int(defensive)
        points = Sobol(rqmc_dimension, scramble=True, seed=seed).random_base2(
            int(np.log2(n))
        )
        points = np.clip(points, np.finfo(float).eps, 1 - np.finfo(float).eps)

        cursor = 0
        if defensive:
            nominal_component = points[:, cursor] < delta
            cursor += 1
        else:
            nominal_component = np.zeros(int(n), dtype=bool)
        base_u = points[:, cursor]
        cursor += 1
        if self.is_skew:
            z_nominal_base = skewnorm.ppf(base_u, self.gamma)
        else:
            z_nominal_base = norm.ppf(base_u)
        z = z_nominal_base + np.where(nominal_component, 0.0, mu)

        lambda_ = None
        if self.is_t_like:
            lambda_u = points[:, cursor]
            cursor += 1
            if self.xi == 1.0:
                proposal_scale = np.where(nominal_component, 1.0, kappa)
                lambda_ = proposal_scale * chi2.ppf(lambda_u, self.nu)
            else:
                nominal_u = (
                    skewnorm.cdf(z, self.gamma)
                    if self.is_skew
                    else norm.cdf(z)
                )
                survival_u = gumbel_conditional_inverse(
                    nominal_u, lambda_u, self.xi
                )
                lambda_ = chi2.isf(survival_u, self.nu)

        if not self.is_t_like:
            p_cond = self.copula.conditional_default_prob(z)
        else:
            p_cond = self.copula.conditional_default_prob(z, lambda_)

        default_u = points[:, cursor : cursor + self.n_obligors]
        cursor += self.n_obligors
        lgd_u = points[:, cursor : cursor + self.n_obligors]
        defaults = default_u <= p_cond
        lgd = beta_dist.ppf(lgd_u, self.portfolio.a, self.portfolio.b)
        individual_losses = defaults * lgd * self.portfolio.u
        total_loss = np.sum(individual_losses, axis=1)

        log_lr_z = -mu * z + 0.5 * mu**2
        if self.is_skew:
            numerator = np.clip(ndtr(self.gamma * z), 1e-300, None)
            denominator = np.clip(
                ndtr(self.gamma * (z - mu)), 1e-300, None
            )
            log_lr_z += np.log(numerator) - np.log(denominator)
        log_lr = log_lr_z
        if self.is_t_like and self.xi == 1.0:
            shape = self.nu / 2.0
            log_lr += shape * np.log(kappa) + lambda_ * (
                1.0 / (2.0 * kappa) - 0.5
            )
        if defensive:
            likelihood_ratio = self._defensive_likelihood_ratio(log_lr, delta)
        else:
            likelihood_ratio = np.exp(np.clip(log_lr, -745, 700))
        return (
            total_loss,
            individual_losses[:, self.target_i],
            likelihood_ratio,
            z,
            lambda_,
        )

    def _conditioning_weight(self, target_loss, a_val, bandwidth=None):
        if self.mode == "equality":
            if a_val == 0:
                return (target_loss == 0).astype(float)
            if bandwidth is None:
                bandwidth = self.bandwidth
            bandwidth = float(bandwidth)
            if bandwidth <= 0:
                raise ValueError("positive-equality bandwidth must be positive")
            distance = np.abs(target_loss - a_val)
            return (distance <= bandwidth).astype(float) / (2.0 * bandwidth)
        if a_val == 0:
            return np.ones_like(target_loss)
        return (target_loss >= a_val).astype(float)

    def _skew_location_update(self, z, weights, current_mu):
        gamma = self.gamma

        def score(mu):
            argument = gamma * (z - mu)
            mills = np.exp(
                np.minimum(norm.logpdf(argument) - norm.logcdf(argument), 700.0)
            )
            return np.sum(weights * ((z - mu) - gamma * mills))

        center = np.sum(weights * z) / np.sum(weights)
        lo, hi = center - 10.0, center + 10.0
        if score(lo) * score(hi) > 0:
            return current_mu
        return float(brentq(score, lo, hi, xtol=1e-10))

    def calibrate_and_optimize(
        self,
        a_val,
        q,
        n0,
        T_max,
        l0,
        initial_mu,
        initial_kappa,
        min_kappa,
        pilot_base_seed,
        convergence_tolerance,
        post_target_updates,
        min_calibration_ess,
        min_elite_ess,
        max_pilot_retries,
        show_progress=False,
    ):
        if not 0 < q < 1:
            raise ValueError("q must lie in (0, 1)")
        if not np.isfinite(l0) or l0 <= 0:
            raise ValueError("l0 must be a positive finite loss threshold")
        if not 0 < min_kappa <= initial_kappa <= 1:
            raise ValueError("require 0 < min_kappa <= initial_kappa <= 1")
        if not 1 <= int(T_max) <= 5:
            raise ValueError("T_max must be between one and five")
        if float(convergence_tolerance) <= 0:
            raise ValueError("convergence_tolerance must be positive")
        if not 0 <= int(post_target_updates) < int(T_max):
            raise ValueError("post_target_updates must lie in [0, T_max)")
        if float(min_calibration_ess) <= 0 or float(min_elite_ess) <= 0:
            raise ValueError("calibration and elite ESS thresholds must be positive")
        if int(max_pilot_retries) < 0:
            raise ValueError("max_pilot_retries must be nonnegative")
        mu, kappa = float(initial_mu), float(initial_kappa)
        iterations = tqdm(
            range(int(T_max)),
            desc=f"CE {self.mode} calibration",
            unit="iter",
            disable=not show_progress,
            leave=False,
        )
        max_retries = int(max_pilot_retries)
        target_reached = False
        target_level_updates = 0
        valid_updates = 0
        last_zeta = np.nan
        last_parameter_change = np.inf
        last_calibration_ess = 0.0
        last_elite_ess = 0.0
        used_calibration_bandwidth = self.calibration_bandwidth
        for iteration in iterations:
            for attempt in range(max_retries + 1):
                if self.mode == "equality" and a_val > 0:
                    # Widen only the pilot band.  The final estimator continues
                    # to use self.bandwidth.
                    attempt_size = int(n0)
                    pilot_bandwidth = self.calibration_bandwidth * (2**attempt)
                else:
                    attempt_size = int(n0) * (2**attempt)
                    pilot_bandwidth = self.bandwidth
                loss, target_loss, lr, z, lambda_ = self._proposal_paths(
                    attempt_size,
                    mu,
                    kappa,
                    seed=int(pilot_base_seed) + 100 * iteration + attempt,
                )
                conditioning = self._conditioning_weight(
                    target_loss, a_val, bandwidth=pilot_bandwidth
                )
                conditioning_total = np.sum(conditioning)
                conditioning_square_total = np.sum(conditioning**2)
                calibration_ess = (
                    conditioning_total**2 / conditioning_square_total
                    if conditioning_square_total > 0
                    else 0.0
                )
                if calibration_ess < float(min_calibration_ess):
                    continue
                if target_reached:
                    zeta = float(l0)
                else:
                    zeta = min(
                        float(l0),
                        self._weighted_quantile(loss, conditioning, 1.0 - q),
                    )
                omega = lr * conditioning * (loss >= zeta)
                total = np.sum(omega)
                omega_square_total = np.sum(omega**2)
                elite_ess = (
                    total**2 / omega_square_total
                    if omega_square_total > 0 and np.isfinite(total)
                    else 0.0
                )
                if (
                    np.isfinite(total)
                    and total > 0
                    and elite_ess >= float(min_elite_ess)
                ):
                    break
            else:
                warnings.warn(
                    "CE calibration exhausted its bounded pilot fallback; "
                    "using the latest valid tilting parameter",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
            previous_mu, previous_kappa = mu, kappa
            if self.is_skew:
                mu = self._skew_location_update(z, omega, mu)
            else:
                mu = float(np.sum(omega * z) / total)
            if self.is_t_like and self.xi == 1.0:
                kappa = float(np.sum(omega * lambda_) / (self.nu * total))
                kappa = float(np.clip(kappa, min_kappa, 1.0))
            mu_change = abs(mu - previous_mu) / (1.0 + abs(previous_mu))
            if self.is_t_like and self.xi == 1.0:
                kappa_change = abs(np.log(kappa) - np.log(previous_kappa))
            else:
                kappa_change = 0.0
            parameter_change = max(mu_change, kappa_change)
            valid_updates += 1
            last_zeta = float(zeta)
            last_parameter_change = float(parameter_change)
            last_calibration_ess = float(calibration_ess)
            last_elite_ess = float(elite_ess)
            used_calibration_bandwidth = float(pilot_bandwidth)
            iterations.set_postfix(
                zeta=f"{zeta:.3f}",
                mu=f"{mu:.3f}",
                kappa=f"{kappa:.3f}",
                ESS=f"{elite_ess:.1f}",
            )
            if zeta >= l0:
                target_reached = True
                target_level_updates += 1
                if (
                    parameter_change <= float(convergence_tolerance)
                    or target_level_updates >= int(post_target_updates) + 1
                ):
                    break
        self.last_calibration = {
            "iterations": valid_updates,
            "target_reached": target_reached,
            "zeta": last_zeta,
            "parameter_change": last_parameter_change,
            "conditioning_ess": last_calibration_ess,
            "elite_ess": last_elite_ess,
            "calibration_bandwidth": used_calibration_bandwidth,
        }
        return mu, kappa

    @staticmethod
    def _weighted_quantile(values, weights, probability):
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie in (0, 1)")
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.shape != weights.shape:
            raise ValueError("values and weights must have the same shape")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("quantile weights must be finite and nonnegative")
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cumulative = np.cumsum(weights)
        if cumulative[-1] <= 0 or not np.isfinite(cumulative[-1]):
            raise RuntimeError("final estimator has zero or invalid total weight")
        index = np.searchsorted(cumulative, probability * cumulative[-1], side="left")
        return float(values[min(index, len(values) - 1)])

    def estimate_preliminary_threshold(
        self, a_val, beta, size, replicas, base_seed
    ):
        """Estimate conditional l0 from independent nominal RQMC samples."""
        if not _is_power_of_two(int(size)):
            raise ValueError("preliminary RQMC size must be a power of two")
        if int(replicas) < 1:
            raise ValueError("preliminary replicas must be positive")
        losses = []
        weights = []
        for replication in range(int(replicas)):
            loss, target_loss, _, _, _ = self._proposal_paths(
                int(size),
                mu=0.0,
                kappa=1.0,
                seed=int(base_seed) + replication,
            )
            losses.append(loss)
            # A moderately wider pilot band gives a stable, inexpensive l0 for
            # proposal calibration.  The reported estimator still uses the
            # single final bandwidth self.bandwidth.
            weights.append(
                self._conditioning_weight(
                    target_loss,
                    a_val,
                    bandwidth=self.calibration_bandwidth,
                )
            )
        losses = np.concatenate(losses)
        weights = np.concatenate(weights)
        total_weight = np.sum(weights)
        self.last_preliminary_ess = float(
            total_weight**2 / np.sum(weights**2)
        )
        if (
            self.last_preliminary_ess * (1.0 - beta)
            < config.BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES
        ):
            warnings.warn(
                "preliminary conditional quantile has too few effective tail "
                "samples; increase CE_RQMC_PRELIMINARY_REPLICAS/SIZE or the "
                "positive-equality calibration bandwidth",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._weighted_quantile(losses, weights, beta)

    def estimate_risk_measures(
        self,
        n,
        mu_star,
        kappa_star,
        a_val,
        beta,
        seed=None,
        defensive_probability=None,
    ):
        if defensive_probability is None:
            defensive_probability = config.CE_RQMC_DEFENSIVE_PROBABILITY
        loss, target_loss, lr, _, _ = self._proposal_paths(
            n,
            mu_star,
            kappa_star,
            seed,
            defensive_probability=defensive_probability,
        )
        weights = lr * self._conditioning_weight(target_loss, a_val)
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        covar = self._weighted_quantile(loss, weights, beta)
        total_weight = np.sum(weights)
        stop_loss = np.sum(weights * np.maximum(loss - covar, 0.0)) / total_weight
        coes = covar + stop_loss / (1.0 - beta)
        ess = total_weight**2 / np.sum(weights**2)
        if (
            ess * (1.0 - beta)
            < config.BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES
        ):
            warnings.warn(
                "effective tail sample size is below the configured minimum; "
                "increase CE_RQMC_FINAL_SIZE or "
                "improve the CE proposal/defensive probability before "
                "reporting this estimate",
                RuntimeWarning,
                stacklevel=2,
            )
        return covar, float(coes), float(ess)
