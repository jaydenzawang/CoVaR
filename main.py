#!/usr/bin/env python3
"""Command-line entry point for equality-conditioned Hybrid-SPA CoVaR."""

from functools import lru_cache
import time

import config
from model_factory import build_model_context, format_model_summary
from risk_engine import HybridSPARiskEngine


@lru_cache(maxsize=1)
def get_engine():
    """Construct the expensive model lazily and exactly once."""
    return HybridSPARiskEngine(build_model_context())


def evaluate_conditional_cdf_fixed(loss, a):
    return get_engine().cdf(loss, "equality", a)


def evaluate_conditional_cdf_adaptive(loss, a):
    raise NotImplementedError(
        "adaptive integration was removed from the production path; use "
        "quadrature-order convergence checks instead"
    )


def calculate_covar(beta, a_val, engine=None):
    engine = get_engine() if engine is None else engine
    return engine.covar(beta, "equality", a_val)


def main():
    started = time.perf_counter()
    engine = get_engine()
    estimate = engine.estimate("equality")
    engine.close()
    print(format_model_summary(engine.context, "L_i=a"))
    print(f"a={estimate.distress_threshold:.8f}")
    print(f"CoVaR={estimate.covar:.8f}")
    if estimate.conditioning_probability is None:
        print("conditioning probability=0 (regular conditional distribution)")
    else:
        print(f"conditioning probability={estimate.conditioning_probability:.8f}")
    print(f"elapsed={time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
