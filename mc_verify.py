#!/usr/bin/env python3
"""Replication driver for the plain-Monte-Carlo benchmark."""

import time
import numpy as np
from tqdm.auto import tqdm

import config
from model_factory import build_model_context
from pmc_engine import PlainMonteCarlo
from var_calc import calculate_individual_var


def evaluate_benchmark_with_se(
    num_sims,
    num_runs,
    copula=None,
    mode="equality", # equality inequality
    show_progress=True,
    batch_size=None,
    bandwidth_ratio=None,
    base_seed=None,
):
    """Retained public entry point; ``copula`` is accepted for compatibility."""
    if int(num_runs) < 2:
        raise ValueError("num_runs must be at least two to estimate SE")
    if int(num_sims) < 1:
        raise ValueError("PMC_NUM_SIMS must be positive")
    context = build_model_context()
    if copula is not None and type(copula) is not type(context.copula):
        raise ValueError("the supplied copula does not match config.COPULA_TYPE")
    a_val = calculate_individual_var(
        context.portfolio, context.copula, context.target_i, context.alpha
    )
    batch_size = config.PMC_BATCH_SIZE if batch_size is None else int(batch_size)
    bandwidth_ratio = (
        config.PMC_EQUALITY_BANDWIDTH_RATIO
        if bandwidth_ratio is None
        else float(bandwidth_ratio)
    )
    base_seed = config.PMC_BASE_SEED if base_seed is None else int(base_seed)
    if batch_size < 1:
        raise ValueError("PMC_BATCH_SIZE must be positive")
    if bandwidth_ratio <= 0:
        raise ValueError("PMC_EQUALITY_BANDWIDTH_RATIO must be positive")
    bandwidth = bandwidth_ratio * context.portfolio.u[context.target_i]
    engine = PlainMonteCarlo(context, mode=mode, bandwidth=bandwidth)
    started = time.perf_counter()
    estimates = []
    replications = tqdm(
        range(int(num_runs)),
        desc=f"PMC {mode} replications",
        unit="run",
        disable=not show_progress,
        position=0,
    )
    for run in replications:
        estimate = engine.estimate(
            num_sims,
            a_val,
            context.beta,
            batch_size=batch_size,
            seed=base_seed + run,
            show_progress=show_progress,
            progress_desc=f"PMC {mode} run {run + 1}/{num_runs}",
            progress_position=1,
        )
        estimates.append(estimate)
        replications.set_postfix(
            CoVaR=f"{estimate[0]:.4f}", CoES=f"{estimate[1]:.4f}"
        )
    estimates = np.asarray(estimates)
    mean = estimates.mean(axis=0)
    se = estimates[:, :2].std(axis=0, ddof=1) / np.sqrt(num_runs)
    print(
        f"PMC {mode} (h/u_i={bandwidth_ratio:.3g}): "
        f"CoVaR={mean[0]:.8f} (SE={se[0]:.3g}), "
        f"CoES={mean[1]:.8f} (SE={se[1]:.3g}), mean ESS={mean[2]:.1f}, "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    return mean[0], se[0]


if __name__ == "__main__":
    evaluate_benchmark_with_se(
        config.PMC_NUM_SIMS,
        config.PMC_NUM_RUNS,
        mode="equality",
    )
    evaluate_benchmark_with_se(
        config.PMC_NUM_SIMS, config.PMC_NUM_RUNS, mode="inequality"
    )
