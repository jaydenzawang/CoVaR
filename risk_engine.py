"""Hybrid-SPA estimators matching the definitions in ``CoVaR_draft.tex``."""

from dataclasses import dataclass
import multiprocessing as mp
import traceback
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist

from spa import SaddlepointApproximation
from var_calc import calculate_individual_var


def _evaluate_factor_operation(spa_minus, spa_all, p, operation, arguments):
    if operation == "cdf_all":
        return spa_all.prob_L_minus_i(arguments[0], p)
    if operation == "cdf_minus":
        return spa_minus.prob_L_minus_i(arguments[0], p)
    if operation == "cdf_inequality":
        loss, x_nodes, x_weights = arguments
        conditional = np.fromiter(
            (spa_minus.prob_L_minus_i(loss - x, p) for x in x_nodes),
            dtype=float,
            count=len(x_nodes),
        )
        return np.dot(x_weights, conditional)
    if operation == "stop_all":
        theta = arguments[0]
        return spa_all.stop_loss_L_minus_i(theta, p)
    if operation == "stop_minus":
        y = arguments[0]
        return spa_minus.stop_loss_L_minus_i(y, p)
    if operation == "stop_inequality":
        theta, x_nodes, x_weights = arguments
        stop_losses = np.empty(len(x_nodes))
        for index, x in enumerate(x_nodes):
            y = theta - x
            stop_losses[index] = spa_minus.stop_loss_L_minus_i(y, p)
        return np.dot(x_weights, stop_losses)
    raise ValueError(f"unknown factor operation: {operation}")


def _factor_worker_loop(connection, spa_minus, spa_all, p_cond):
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            start, end, operation, arguments = task
            try:
                cpu_started = time.process_time()
                values = np.empty(end - start)
                for local_index, factor_index in enumerate(range(start, end)):
                    values[local_index] = _evaluate_factor_operation(
                        spa_minus,
                        spa_all,
                        p_cond[factor_index],
                        operation,
                        arguments,
                    )
                cpu_elapsed = time.process_time() - cpu_started
                connection.send((True, start, values, cpu_elapsed))
            except BaseException:
                connection.send((False, start, traceback.format_exc(), 0.0))
    finally:
        connection.close()


@dataclass(frozen=True)
class RiskEstimate:
    covar: float
    coes: float
    distress_threshold: float
    conditioning_probability: float | None


class HybridSPARiskEngine:
    """Compute conditional generalized quantiles and robust conditional ES.

    ``mode='equality'`` implements regular conditioning for positive ``a`` and
    the non-default atom when ``a=0``.  ``mode='inequality'`` implements the
    positive truncated-LGD event and reduces exactly to the unconditional loss
    distribution when ``a=0``.
    """

    def __init__(
        self,
        context,
        lgd_order=32,
        quantile_tol=1e-5,
        n_jobs=None,
        saddle_bound=20.0,
        mean_region=1e-2,
        tail_bound_tol=1e-12,
    ):
        self.context = context
        self.portfolio = context.portfolio
        self.factors = context.factors
        self.target_i = context.target_i
        self.quantile_tol = float(quantile_tol)
        self.n_jobs = int(
            getattr(context, "spa_n_jobs", 1) if n_jobs is None else n_jobs
        )
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
        if lgd_order < 8:
            raise ValueError("lgd_order is too small")
        self._legendre_nodes, self._legendre_weights = leggauss(int(lgd_order))
        spa_options = dict(
            saddle_bound=saddle_bound,
            mean_region=mean_region,
            tail_bound_tol=tail_bound_tol,
        )
        self.spa_minus = SaddlepointApproximation(
            self.portfolio, self.target_i, **spa_options
        )
        self.spa_all = SaddlepointApproximation(
            self.portfolio, None, **spa_options
        )
        self._workers = []
        self._worker_cpu_time = 0.0

    @property
    def target_pd(self):
        return float(self.context.copula.p[self.target_i])

    def individual_var(self, alpha=None):
        alpha = self.context.alpha if alpha is None else float(alpha)
        return calculate_individual_var(
            self.portfolio, self.context.copula, self.target_i, alpha
        )

    def _checked_probability(self, value):
        value = float(value)
        if not np.isfinite(value) or value < -5e-7 or value > 1 + 5e-7:
            raise FloatingPointError(f"conditional CDF is invalid before rounding: {value}")
        return float(np.clip(value, 0.0, 1.0))

    def _factor_event_weights(self, mode, a):
        p_i_theta = self.factors.p_cond[:, self.target_i]
        if mode == "equality":
            likelihood = p_i_theta if a > 0 else 1.0 - p_i_theta
        elif mode == "inequality":
            likelihood = np.ones_like(p_i_theta) if a == 0 else p_i_theta
        else:
            raise ValueError("mode must be 'equality' or 'inequality'")
        raw = self.factors.weights * likelihood
        probability = float(np.sum(raw))
        if probability <= 0 or not np.isfinite(probability):
            raise FloatingPointError("conditioning event has invalid numerical probability")
        return raw / probability, probability

    def _truncated_lgd_rule(self, a):
        i = self.target_i
        u_i = self.portfolio.u[i]
        if not 0 < a < u_i:
            raise ValueError("positive distress threshold must lie in (0, u_i)")
        shape_a = self.portfolio.a[i]
        shape_b = self.portfolio.b[i]
        lower_probability = beta_dist.cdf(a / u_i, shape_a, shape_b)
        survival = beta_dist.sf(a / u_i, shape_a, shape_b)
        uniform_nodes = 0.5 * (self._legendre_nodes + 1.0)
        # Smooth the beta-PPF upper-endpoint singularity.  For beta shape b>1,
        # q=1-(1-t)^b makes the conditional quantile asymptotically linear in
        # t near one; for b<=1 the identity map is already regular enough.
        endpoint_power = max(float(shape_b), 1.0)
        conditional_probabilities = 1.0 - (1.0 - uniform_nodes) ** endpoint_power
        probabilities = lower_probability + survival * conditional_probabilities
        nodes = u_i * beta_dist.ppf(probabilities, shape_a, shape_b)
        weights = (
            0.5
            * self._legendre_weights
            * endpoint_power
            * (1.0 - uniform_nodes) ** (endpoint_power - 1.0)
        )
        weights /= np.sum(weights)
        return nodes, weights, survival

    def _ensure_workers(self):
        if not self._workers:
            available_methods = mp.get_all_start_methods()
            start_method = "fork" if "fork" in available_methods else "spawn"
            context = mp.get_context(start_method)
            worker_count = min(self.n_jobs, len(self.factors.weights))
            for _ in range(worker_count):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_factor_worker_loop,
                    args=(
                        child_connection,
                        self.spa_minus,
                        self.spa_all,
                        self.factors.p_cond,
                    ),
                )
                process.start()
                child_connection.close()
                self._workers.append((process, parent_connection))
        return self._workers

    def close(self):
        for _, connection in self._workers:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for process, connection in self._workers:
            process.join()
            connection.close()
        self._workers = []

    def cpu_time(self):
        """Parent plus accumulated worker-process CPU time in seconds."""
        return time.process_time() + self._worker_cpu_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_object):
        self.close()

    def _map_factor_nodes(self, operation, *arguments):
        """Evaluate factor-node blocks in processes and restore original order."""
        if self.n_jobs == 1:
            return np.fromiter(
                (
                    _evaluate_factor_operation(
                        self.spa_minus,
                        self.spa_all,
                        p,
                        operation,
                        arguments,
                    )
                    for p in self.factors.p_cond
                ),
                dtype=float,
                count=len(self.factors.weights),
            )
        node_count = len(self.factors.weights)
        workers = self._ensure_workers()
        boundaries = np.linspace(0, node_count, len(workers) + 1, dtype=int)
        for index, (_, connection) in enumerate(workers):
            connection.send(
                (
                    int(boundaries[index]),
                    int(boundaries[index + 1]),
                    operation,
                    arguments,
                )
            )
        values = np.empty(node_count)
        for _, connection in workers:
            success, start, payload, worker_cpu = connection.recv()
            if not success:
                raise RuntimeError(f"factor worker failed:\n{payload}")
            self._worker_cpu_time += worker_cpu
            chunk = payload
            values[start : start + len(chunk)] = chunk
        return values

    def cdf(self, loss, mode, a):
        loss = float(loss)
        a = float(a)
        factor_weights, _ = self._factor_event_weights(mode, a)

        if mode == "inequality" and a == 0:
            values = self._map_factor_nodes("cdf_all", loss)
        elif mode == "equality":
            values = self._map_factor_nodes("cdf_minus", loss - a)
        else:
            x_nodes, x_weights, _ = self._truncated_lgd_rule(a)
            values = self._map_factor_nodes(
                "cdf_inequality", loss, x_nodes, x_weights
            )
        return self._checked_probability(np.dot(factor_weights, values))

    def covar(self, beta, mode, a):
        beta = float(beta)
        if not 0 < beta < 1:
            raise ValueError("beta must lie strictly between zero and one")
        if self.cdf(0.0, mode, a) >= beta:
            return 0.0
        support = float(np.sum(self.portfolio.u))
        objective = lambda loss: self.cdf(loss, mode, a) - beta
        if objective(support) < 0:
            raise RuntimeError("conditional CDF does not reach beta on the loss support")
        return float(brentq(objective, 0.0, support, xtol=self.quantile_tol, rtol=1e-12))

    def _stop_loss(self, theta, mode, a):
        factor_weights, _ = self._factor_event_weights(mode, a)

        if mode == "inequality" and a == 0:
            values = self._map_factor_nodes("stop_all", theta)
        elif mode == "equality":
            y = theta - a
            values = self._map_factor_nodes("stop_minus", y)
        else:
            x_nodes, x_weights, _ = self._truncated_lgd_rule(a)
            values = self._map_factor_nodes(
                "stop_inequality", theta, x_nodes, x_weights
            )

        result = float(np.dot(factor_weights, values))
        if result < -1e-7 or not np.isfinite(result):
            raise FloatingPointError(f"invalid stop-loss expectation: {result}")
        return max(result, 0.0)

    def stop_loss(self, theta, mode, a):
        """Public stop-loss expectation E[(L-theta)^+ | conditioning event]."""
        return self._stop_loss(float(theta), mode, float(a))

    def conditioning_probability(self, mode, a):
        """Probability of a positive-probability event; ``None`` for L_i=a>0."""
        _, factor_normalizer = self._factor_event_weights(mode, a)
        if mode == "equality":
            return factor_normalizer if a == 0 else None
        if a == 0:
            return 1.0
        _, _, survival = self._truncated_lgd_rule(a)
        return factor_normalizer * survival

    def coes(self, beta, mode, a, theta=None):
        beta = float(beta)
        theta = self.covar(beta, mode, a) if theta is None else float(theta)
        stop_loss = self.stop_loss(theta, mode, a)
        return theta + stop_loss / (1.0 - beta)

    def estimate(self, mode, alpha=None, beta=None):
        alpha = self.context.alpha if alpha is None else float(alpha)
        beta = self.context.beta if beta is None else float(beta)
        a = self.individual_var(alpha)
        theta = self.covar(beta, mode, a)
        coes = self.coes(beta, mode, a, theta)
        probability = self.conditioning_probability(mode, a)
        return RiskEstimate(theta, coes, a, probability)
