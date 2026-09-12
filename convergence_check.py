#!/usr/bin/env python3
"""Check factor/LGD quadrature convergence before producing paper tables."""

import numpy as np

import config
from model_factory import build_model_context
from risk_engine import HybridSPARiskEngine


def run_convergence_check(mode="equality"):
    if config.COPULA_TYPE == "Gaussian":
        orders = ((256, 8), (512, 8), (1024, 8))
    else:
        orders = ((32, 32), (48, 48), (64, 64))
    results = []
    for factor_order, shock_order in orders:
        context = build_model_context(n_gh=factor_order, n_gl=shock_order)
        engine = HybridSPARiskEngine(context, lgd_order=32)
        a = engine.individual_var()
        covar = engine.covar(context.beta, mode, a)
        coes = engine.coes(context.beta, mode, a, covar)
        engine.close()
        results.append((factor_order, shock_order, covar, coes))
        print(
            f"orders=({factor_order},{shock_order}), "
            f"CoVaR={covar:.10f}, CoES={coes:.10f}"
        )
    changes = []
    for index, label in ((2, "CoVaR"), (3, "CoES")):
        previous, final = results[-2][index], results[-1][index]
        absolute = abs(final - previous)
        relative = absolute / max(abs(final), np.finfo(float).eps)
        changes.append(relative)
        print(
            f"{label} last refinement: absolute={absolute:.6g}, "
            f"relative={relative:.3%}"
        )
    if max(changes) > 1e-3:
        raise RuntimeError(
            "factor quadrature has not converged to 0.1%; increase the orders"
        )
    if mode == "inequality":
        context = build_model_context(n_gh=orders[-1][0], n_gl=orders[-1][1])
        lgd_results = []
        for lgd_order in (16, 32, 64):
            engine = HybridSPARiskEngine(context, lgd_order=lgd_order)
            a = engine.individual_var()
            covar = engine.covar(context.beta, mode, a)
            coes = engine.coes(context.beta, mode, a, covar)
            engine.close()
            lgd_results.append((lgd_order, covar, coes))
            print(
                f"LGD order={lgd_order}, CoVaR={covar:.10f}, CoES={coes:.10f}"
            )
        lgd_changes = [
            abs(lgd_results[-1][index] - lgd_results[-2][index])
            / max(abs(lgd_results[-1][index]), np.finfo(float).eps)
            for index in (1, 2)
        ]
        if max(lgd_changes) > 1e-3:
            raise RuntimeError(
                "truncated-LGD quadrature has not converged to 0.1%; "
                "increase lgd_order"
            )
    return results


if __name__ == "__main__":
    run_convergence_check("equality")
    run_convergence_check("inequality")
