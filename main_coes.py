#!/usr/bin/env python3
"""Command-line entry point for equality-conditioned robust CoES."""

import time

import config
from main import get_engine
from model_factory import format_model_summary


def evaluate_coes_numerator_fixed(theta, a_val):
    """Return E[(L-theta)^+ | L_i=a], not a raw tail first moment."""
    return get_engine().stop_loss(theta, "equality", a_val)


def calculate_coes(beta, a_val, theta=None, engine=None):
    engine = get_engine() if engine is None else engine
    return engine.coes(beta, "equality", a_val, theta)


def calculate_spa_result(
    progress_callback=None, engine=None, a_val=None, return_wall_time=False
):
    """Return ``(CoVaR^=, CoES^=)`` for the current configuration."""
    engine = get_engine() if engine is None else engine
    a_val = engine.individual_var() if a_val is None else float(a_val)
    if progress_callback is not None:
        progress_callback("model ready")
    covar_started = time.perf_counter()
    covar = engine.covar(config.beta, "equality", a_val)
    covar_wall = time.perf_counter() - covar_started
    if progress_callback is not None:
        progress_callback("CoVaR complete")
    coes_started = time.perf_counter()
    coes = engine.coes(config.beta, "equality", a_val, theta=covar)
    coes_wall = time.perf_counter() - coes_started
    if progress_callback is not None:
        progress_callback("CoES complete")
    if return_wall_time:
        return covar, coes, covar_wall + coes_wall
    return covar, coes


def main():
    setup_wall_start = time.perf_counter()
    setup_cpu_start = time.process_time()
    engine = get_engine()
    a_val = engine.individual_var()
    setup_wall = time.perf_counter() - setup_wall_start
    setup_cpu = time.process_time() - setup_cpu_start

    covar_wall_start = time.perf_counter()
    covar_cpu_start = engine.cpu_time()
    covar = engine.covar(config.beta, "equality", a_val)
    covar_wall = time.perf_counter() - covar_wall_start
    covar_cpu = engine.cpu_time() - covar_cpu_start

    coes_wall_start = time.perf_counter()
    coes_cpu_start = engine.cpu_time()
    coes = engine.coes(config.beta, "equality", a_val, theta=covar)
    coes_incremental_wall = time.perf_counter() - coes_wall_start
    coes_incremental_cpu = engine.cpu_time() - coes_cpu_start
    engine.close()

    print(format_model_summary(engine.context, "L_i=a"))
    print(f"a={a_val:.8f}")
    print(f"CoVaR={covar:.8f}")
    print(f"CoES={coes:.8f}")
    print(f"setup CPU time={setup_cpu:.3f}s, wall time={setup_wall:.3f}s")
    print(f"CoVaR CPU time={covar_cpu:.3f}s, wall time={covar_wall:.3f}s")
    print(
        "CoES incremental CPU time="
        f"{coes_incremental_cpu:.3f}s, wall time={coes_incremental_wall:.3f}s"
    )
    print(
        "CoVaR+CoES CPU time="
        f"{covar_cpu + coes_incremental_cpu:.3f}s, "
        f"wall time={covar_wall + coes_incremental_wall:.3f}s"
    )


if __name__ == "__main__":
    main()
