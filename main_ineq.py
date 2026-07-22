#!/usr/bin/env python3
"""Command-line entry point for inequality-conditioned Hybrid-SPA CoVaR."""

from functools import lru_cache
import time

import config
from model_factory import build_model_context, format_model_summary
from risk_engine import HybridSPARiskEngine


@lru_cache(maxsize=1)
def get_engine():
    return HybridSPARiskEngine(build_model_context())


def evaluate_conditional_cdf_ineq_fixed(loss, a_val):
    return get_engine().cdf(loss, "inequality", a_val)


def evaluate_conditional_cdf_ineq_adaptive(loss, a_val):
    raise NotImplementedError(
        "adaptive integration was removed from the production path; use "
        "quadrature-order convergence checks instead"
    )


def calculate_covar_ineq(beta, a_val, engine=None):
    engine = get_engine() if engine is None else engine
    return engine.covar(beta, "inequality", a_val)


def main():
    started = time.perf_counter()
    engine = get_engine()
    estimate = engine.estimate("inequality")
    engine.close()
    print(format_model_summary(engine.context, "L_i>=a"))
    print(f"a={estimate.distress_threshold:.8f}")
    print(f"CoVaR={estimate.covar:.8f}")
    print(f"conditioning probability={estimate.conditioning_probability:.8f}")
    print(f"elapsed={time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
