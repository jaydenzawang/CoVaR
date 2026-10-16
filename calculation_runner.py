"""
Shared execution and timing for the method/measure main files.
Numerical formulas belong to the engines.
"""

from dataclasses import dataclass, field
import time

import numpy as np
from tqdm.auto import tqdm

import config
from model_factory import build_model_context, build_spa_engine, format_model_summary
from pmc_engine import PlainMonteCarlo
from ce_rqmc_engine import IterativeCE_RQMC
from var_calc import calculate_individual_var


MEASURES = {
    "covar": ("system", False, "CoVaR", "CoES"),
    "coes": ("system", True, "CoVaR", "CoES"),
    "ecovar": ("exposure", False, "E-CoVaR", "E-CoES"),
    "ecoes": ("exposure", True, "E-CoVaR", "E-CoES"),
}
METHOD_LABELS = {"spa": "SPA", "pmc": "PMC", "ce_rqmc": "CE-RQMC"}


@dataclass(frozen=True)
class Runtime:
    wall_seconds: float
    cpu_seconds: float


@dataclass(frozen=True)
class CalculationResult:
    method: str
    measure: str
    mode: str
    direction: str
    distress_threshold: float
    threshold_source: str
    values: dict[str, float]
    standard_errors: dict[str, float]
    timings: dict[str, Runtime]
    model_summary: str
    replications: int = 0
    mean_ess: float | None = None
    bandwidth: float | None = None
    diagnostics: dict = field(default_factory=dict)
    # Columns correspond to values.keys(); ESS is stored separately.
    replication_values: np.ndarray | None = None


def _timed(function, cpu_clock=time.process_time):
    wall_start, cpu_start = time.perf_counter(), cpu_clock()
    value = function()
    return value, Runtime(time.perf_counter()-wall_start, cpu_clock()-cpu_start)


def _integer(cfg, name, minimum=1):
    value = getattr(cfg, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(cfg, name):
    value = float(getattr(cfg, name))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_run(cfg, method, measure, mode):
    if method not in METHOD_LABELS:
        raise ValueError(f"unknown method: {method}")
    if measure not in MEASURES:
        raise ValueError(f"unknown measure: {measure}")
    if mode not in ("equality", "inequality"):
        raise ValueError("CONDITION_MODE must be 'equality' or 'inequality'")
    for name, minimum in (("SPA_N_JOBS", 1), ("SPA_LGD_ORDER", 8),
                          ("SPA_MGF_GRID_SIZE", 100), ("SPA_SCALE_ORDER", 8)):
        _integer(cfg, name, minimum)
    _integer(cfg, "SPA_GAUSSIAN_FACTOR_ORDER" if cfg.COPULA_TYPE == "Gaussian" else "SPA_T_FACTOR_ORDER", 8)
    for name in ("SPA_QUANTILE_TOL", "SPA_ROOT_TOL", "SPA_SADDLE_BOUND", "SPA_MEAN_REGION", "SPA_TAIL_BOUND_TOL"):
        _positive(cfg, name)
    if not 0 < cfg.SPA_MEAN_REGION <= .1 or not 0 < cfg.SPA_TAIL_BOUND_TOL < 1e-4:
        raise ValueError("invalid SPA mean-region or tail tolerance")
    if not np.isfinite(cfg.SPA_MGF_S_MIN) or not np.isfinite(cfg.SPA_MGF_S_MAX) or not (
        cfg.SPA_MGF_S_MIN <= -cfg.SPA_SADDLE_BOUND < 0 < cfg.SPA_SADDLE_BOUND <= cfg.SPA_MGF_S_MAX
    ):
        raise ValueError("SPA MGF range must contain both saddle bounds")
    direction = MEASURES[measure][0]
    threshold = getattr(cfg, f"{direction.upper()}_DISTRESS_THRESHOLD")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("fixed distress threshold must be finite and nonnegative")
    if method == "spa":
        return
    _positive(cfg, "BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES")
    prefix = "PMC" if method == "pmc" else "CE_RQMC"
    ratio_name = prefix + ("_EXPOSURE" if direction == "exposure" else "") + "_EQUALITY_BANDWIDTH_RATIO"
    _positive(cfg, ratio_name)
    if method == "pmc":
        _integer(cfg, "PMC_NUM_RUNS", 2)
        _integer(cfg, "PMC_NUM_SIMS")
        _integer(cfg, "PMC_BATCH_SIZE")
        _integer(cfg, "PMC_BASE_SEED", 0)
        return
    _integer(cfg, "CE_RQMC_REPLICAS", 2)
    _integer(cfg, "CE_RQMC_PRELIMINARY_REPLICAS")
    for name in ("CE_RQMC_PRELIMINARY_SIZE", "CE_RQMC_PILOT_SIZE", "CE_RQMC_FINAL_SIZE"):
        n = _integer(cfg, name)
        if n & (n-1):
            raise ValueError(f"{name} must be a power of two")
    iterations = _integer(cfg, "CE_RQMC_MAX_ITERATIONS")
    updates = _integer(cfg, "CE_RQMC_POST_TARGET_UPDATES", 0)
    retries = _integer(cfg, "CE_RQMC_MAX_PILOT_RETRIES", 0)
    if iterations > 5 or updates >= iterations or retries >= 100:
        raise ValueError("CE requires iterations <= 5, post-target updates < iterations, and retries < 100")
    for name in ("CE_RQMC_ELITE_FRACTION", "CE_RQMC_DEFENSIVE_PROBABILITY"):
        if not 0 < float(getattr(cfg, name)) < 1:
            raise ValueError(f"{name} must lie in (0,1)")
    for name in ("CE_RQMC_MIN_CALIBRATION_ESS", "CE_RQMC_MIN_ELITE_ESS", "CE_RQMC_PARAMETER_TOLERANCE"):
        _positive(cfg, name)
    if not np.isfinite(cfg.CE_RQMC_INITIAL_MU) or not 0 < cfg.CE_RQMC_MIN_KAPPA <= cfg.CE_RQMC_INITIAL_KAPPA <= 1:
        raise ValueError("invalid initial CE location or scale parameters")
    if not np.isfinite(cfg.CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER) or cfg.CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER < 1:
        raise ValueError("CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER must be >= 1")
    # The final randomizations must be independent of all proposal training.
    pre = _integer(cfg, "CE_RQMC_PRELIMINARY_BASE_SEED", 0)
    pilot = _integer(cfg, "CE_RQMC_PILOT_BASE_SEED", 0)
    final = _integer(cfg, "CE_RQMC_BASE_SEED", 0)
    pre_seeds = set(range(pre, pre+cfg.CE_RQMC_PRELIMINARY_REPLICAS))
    pilot_seeds = {pilot+100*i+j for i in range(iterations) for j in range(retries+1)}
    final_seeds = set(range(final, final+cfg.CE_RQMC_REPLICAS))
    if pre_seeds & pilot_seeds or pre_seeds & final_seeds or pilot_seeds & final_seeds:
        raise ValueError("CE preliminary, pilot and final seed ranges must be disjoint")


def summarize_replications(estimates, labels):
    """SE of the replication mean, using sample SD (ddof=1), not path SD."""
    samples = np.asarray(estimates, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != len(labels):
        raise ValueError("at least two replications and one column per measure are required")
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("non-finite benchmark result; no partial mean is reported")
    means = samples.mean(axis=0)
    errors = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    return dict(zip(labels, map(float, means))), dict(zip(labels, map(float, errors)))


def run_calculation(method, measure, cfg=config, *, mode=None, report=True):

    mode = cfg.CONDITION_MODE if mode is None else mode
    _validate_run(cfg, method, measure, mode)
    direction, with_es, q_label, es_label = MEASURES[measure]
    labels = [q_label, es_label] if with_es else [q_label]
    total_wall, total_cpu = time.perf_counter(), time.process_time()
    timings, diagnostics = {}, {}
    spa = None
    context, timings["setup"] = _timed(lambda: build_model_context(cfg))
    summary = format_model_summary(context, ("L_i" if direction == "system" else "L") + ("=threshold" if mode == "equality" else ">=threshold"))
    try:
        fixed = getattr(cfg, f"{direction.upper()}_DISTRESS_THRESHOLD")
        if method == "spa" or (direction == "exposure" and fixed is None):
            spa, engine_time = _timed(lambda: build_spa_engine(context))
            t = timings["setup"]
            timings["setup"] = Runtime(t.wall_seconds+engine_time.wall_seconds, t.cpu_seconds+engine_time.cpu_seconds)
        clock = spa.cpu_time if spa is not None else time.process_time
        if fixed is not None:
            threshold, source = float(fixed), "config: fixed loss"
            timings["threshold"] = Runtime(0., 0.)
        elif direction == "system":
            threshold, timings["threshold"] = _timed(lambda: calculate_individual_var(context.portfolio, context.copula, context.target_i, context.alpha))
            source = "analytic individual VaR(alpha)"
        else:
            threshold, timings["threshold"] = _timed(lambda: spa.portfolio_var(), clock)
            source = "SPA portfolio VaR(alpha)"
        support = context.portfolio.u[context.target_i] if direction == "system" else np.sum(context.portfolio.u)
        if not 0 <= threshold < support:
            raise ValueError("distress threshold must be >= 0 and below the conditioning-loss support")
        values, errors = {}, {}
        bandwidth = mean_ess = replication_values = None
        replications = 0
        if method == "spa":
            q_function = spa.covar if direction == "system" else spa.ecovar
            es_function = spa.coes if direction == "system" else spa.ecoes
            q, timings["quantile"] = _timed(lambda: q_function(context.beta, mode, threshold), clock)
            values[q_label] = q
            if with_es:
                values[es_label], timings["es_incremental"] = _timed(lambda: es_function(context.beta, mode, threshold, theta=q), clock)
        else:
            prefix = "PMC" if method == "pmc" else "CE_RQMC"
            ratio_name = prefix + ("_EXPOSURE" if direction == "exposure" else "") + "_EQUALITY_BANDWIDTH_RATIO"
            ratio = float(getattr(cfg, ratio_name))
            bandwidth = ratio * (max(threshold, 1.) if direction == "exposure" else context.portfolio.u[context.target_i])
            common = dict(mode=mode, bandwidth=bandwidth, direction=direction,
                          min_effective_tail_samples=cfg.BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES)
            if method == "pmc":
                engine = PlainMonteCarlo(context, **common)
                replications = cfg.PMC_NUM_RUNS
                def estimate(r):
                    return engine.estimate(cfg.PMC_NUM_SIMS, threshold, context.beta,
                                           batch_size=cfg.PMC_BATCH_SIZE, seed=cfg.PMC_BASE_SEED+r,
                                           show_progress=cfg.SHOW_PROGRESS,
                                           progress_desc=f"PMC {mode} run {r+1}/{replications}",
                                           progress_position=1, compute_es=with_es)
            else:
                engine = IterativeCE_RQMC(context.portfolio, context.copula, context.target_i,
                                          calibration_bandwidth=bandwidth*cfg.CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER,
                                          **common)
                l0, timings["preliminary"] = _timed(lambda: engine.estimate_preliminary_threshold(
                    threshold, context.beta, cfg.CE_RQMC_PRELIMINARY_SIZE,
                    cfg.CE_RQMC_PRELIMINARY_REPLICAS, cfg.CE_RQMC_PRELIMINARY_BASE_SEED))
                (mu, kappa), timings["calibration"] = _timed(lambda: engine.calibrate_and_optimize(
                    threshold, q=cfg.CE_RQMC_ELITE_FRACTION, n0=cfg.CE_RQMC_PILOT_SIZE,
                    T_max=cfg.CE_RQMC_MAX_ITERATIONS, l0=l0, initial_mu=cfg.CE_RQMC_INITIAL_MU,
                    initial_kappa=cfg.CE_RQMC_INITIAL_KAPPA, min_kappa=cfg.CE_RQMC_MIN_KAPPA,
                    pilot_base_seed=cfg.CE_RQMC_PILOT_BASE_SEED,
                    convergence_tolerance=cfg.CE_RQMC_PARAMETER_TOLERANCE,
                    post_target_updates=cfg.CE_RQMC_POST_TARGET_UPDATES,
                    min_calibration_ess=cfg.CE_RQMC_MIN_CALIBRATION_ESS,
                    min_elite_ess=cfg.CE_RQMC_MIN_ELITE_ESS,
                    max_pilot_retries=cfg.CE_RQMC_MAX_PILOT_RETRIES, show_progress=cfg.SHOW_PROGRESS))
                diagnostics.update(l0=l0, preliminary_ess=engine.last_preliminary_ess,
                                   mu=mu, kappa=kappa, calibration=engine.last_calibration.copy())
                replications = cfg.CE_RQMC_REPLICAS
                def estimate(r):
                    return engine.estimate_risk_measures(
                        cfg.CE_RQMC_FINAL_SIZE, mu, kappa, threshold, context.beta,
                        seed=cfg.CE_RQMC_BASE_SEED+r,
                        defensive_probability=cfg.CE_RQMC_DEFENSIVE_PROBABILITY, compute_es=with_es)
            def simulate():
                rows, effective = [], []
                with tqdm(range(replications), desc=f"{METHOD_LABELS[method]} {mode}", unit="run", disable=not cfg.SHOW_PROGRESS) as progress:
                    for r in progress:
                        q, es, ess = estimate(r)
                        rows.append([q, es] if with_es else [q])
                        effective.append(ess)
                        progress.set_postfix({labels[-1]: f"{rows[-1][-1]:.6f}", "ESS": f"{ess:.0f}"})
                return np.asarray(rows), float(np.mean(effective))
            (replication_values, mean_ess), timings["estimation"] = _timed(simulate)
            values, errors = summarize_replications(replication_values, labels)
    finally:
        worker_cpu = 0. if spa is None else spa.cpu_time()-time.process_time()
        if spa is not None:
            spa.close()
    timings["total"] = Runtime(time.perf_counter()-total_wall, time.process_time()-total_cpu+max(worker_cpu, 0.))
    result = CalculationResult(method, measure, mode, direction, threshold, source,
                               values, errors, timings, summary, replications,
                               mean_ess, bandwidth, diagnostics, replication_values)
    if report:
        print_result(result)
    return result


def print_result(result):
    
    condition = "Equality condition" if result.mode == "equality" else "Inequality condition"
    print(f"{METHOD_LABELS[result.method]}({condition})")
    for label, value in result.values.items():
        suffix = f"    SE = {result.standard_errors[label]:.8g}" if label in result.standard_errors else ""
        print(f"{label} = {value:.10f}{suffix}")

    def show_time(label, *phases):
        cpu = sum(result.timings[p].cpu_seconds for p in phases)
        wall = sum(result.timings[p].wall_seconds for p in phases)
        print(f"{label}:CPU = {cpu:.6f} s; wall = {wall:.6f} s")

    print()
    show_time("Setup (model and distress threshold)", "setup", "threshold")
    labels = list(result.values)
    if result.method == "spa":
        show_time(labels[0], "quantile")
        if "es_incremental" in result.timings:
            show_time(f"{labels[1]} (extra time, given {labels[0]})", "es_incremental")
    else:
        if result.method == "ce_rqmc":
            show_time("CE initial estimate and fit", "preliminary", "calibration")
        # Both measures use the same simulated paths; these are joint runtimes,
        # not separate per-measure times or the average time of one replication.
        show_time(" + ".join(labels) + " (total time for all runs)", "estimation")
    show_time("Total (setup and all steps)", "total")
    cpu = result.timings["total"].cpu_seconds - sum(
        result.timings[p].cpu_seconds for p in ("setup", "threshold")
    )
    wall = result.timings["total"].wall_seconds - sum(
        result.timings[p].wall_seconds for p in ("setup", "threshold")
    )
    print(f"Total (without setup): CPU = {cpu:.6f} s; wall = {wall:.6f} s")
    print()
