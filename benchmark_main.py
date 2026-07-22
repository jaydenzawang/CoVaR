#!/usr/bin/env python3
"""Run the faithful CE-RQMC benchmark with replication-based uncertainty."""

import time
import numpy as np
from tqdm.auto import tqdm

import config
from ce_rqmc_engine import IterativeCE_RQMC, _is_power_of_two
from model_factory import build_model_context
from var_calc import calculate_individual_var

def main():
    if config.CE_RQMC_REPLICAS < 2:
        raise ValueError("CE_RQMC_REPLICAS must be at least two to estimate SE")
    if not 0 < config.CE_RQMC_ELITE_FRACTION < 1:
        raise ValueError("CE_RQMC_ELITE_FRACTION must lie in (0, 1)")
    if not 1 <= config.CE_RQMC_MAX_ITERATIONS <= 5:
        raise ValueError("CE_RQMC_MAX_ITERATIONS must be between one and five")
    if config.CE_RQMC_EQUALITY_BANDWIDTH_RATIO <= 0:
        raise ValueError("CE_RQMC_EQUALITY_BANDWIDTH_RATIO must be positive")
    if config.CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER < 1:
        raise ValueError(
            "CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER must be at least one"
        )
    if config.CE_RQMC_MAX_PILOT_RETRIES < 0:
        raise ValueError("CE_RQMC_MAX_PILOT_RETRIES must be nonnegative")
    if config.CE_RQMC_MIN_CALIBRATION_ESS <= 0:
        raise ValueError("CE_RQMC_MIN_CALIBRATION_ESS must be positive")
    if config.CE_RQMC_MIN_ELITE_ESS <= 0:
        raise ValueError("CE_RQMC_MIN_ELITE_ESS must be positive")
    if config.CE_RQMC_PARAMETER_TOLERANCE <= 0:
        raise ValueError("CE_RQMC_PARAMETER_TOLERANCE must be positive")
    if not 0 <= config.CE_RQMC_POST_TARGET_UPDATES < config.CE_RQMC_MAX_ITERATIONS:
        raise ValueError(
            "CE_RQMC_POST_TARGET_UPDATES must lie in [0, CE_RQMC_MAX_ITERATIONS)"
        )
    if not 0 < config.CE_RQMC_DEFENSIVE_PROBABILITY < 1:
        raise ValueError("CE_RQMC_DEFENSIVE_PROBABILITY must lie in (0, 1)")
    if not _is_power_of_two(config.CE_RQMC_PILOT_SIZE):
        raise ValueError("CE_RQMC_PILOT_SIZE must be a power of two")
    if not _is_power_of_two(config.CE_RQMC_FINAL_SIZE):
        raise ValueError("CE_RQMC_FINAL_SIZE must be a power of two")
    if not _is_power_of_two(config.CE_RQMC_PRELIMINARY_SIZE):
        raise ValueError("CE_RQMC_PRELIMINARY_SIZE must be a power of two")
    if config.CE_RQMC_PRELIMINARY_REPLICAS < 1:
        raise ValueError("CE_RQMC_PRELIMINARY_REPLICAS must be positive")
    context = build_model_context()
    a_val = calculate_individual_var(
        context.portfolio, context.copula, context.target_i, context.alpha
    )
    print(f"model={config.COPULA_TYPE}, xi={config.XI}, a={a_val:.8f}")
    ratio = float(config.CE_RQMC_EQUALITY_BANDWIDTH_RATIO)
    final_bandwidth = ratio * context.portfolio.u[context.target_i]
    calibration_bandwidth = (
        config.CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER * final_bandwidth
    )
    for mode in ("equality", "inequality"):
        engine = IterativeCE_RQMC(
            context.portfolio,
            context.copula,
            context.target_i,
            mode=mode,
            bandwidth=final_bandwidth,
            calibration_bandwidth=calibration_bandwidth,
        )
        started = time.perf_counter()
        l0 = engine.estimate_preliminary_threshold(
            a_val,
            context.beta,
            size=config.CE_RQMC_PRELIMINARY_SIZE,
            replicas=config.CE_RQMC_PRELIMINARY_REPLICAS,
            base_seed=config.CE_RQMC_PRELIMINARY_BASE_SEED,
        )
        print(
            f"{mode} (final h/u_i={ratio:.3g}): "
            f"independent preliminary l0={l0:.8f}, "
            f"ESS={engine.last_preliminary_ess:.1f}"
        )
        mu, kappa = engine.calibrate_and_optimize(
            a_val,
            q=config.CE_RQMC_ELITE_FRACTION,
            n0=config.CE_RQMC_PILOT_SIZE,
            T_max=config.CE_RQMC_MAX_ITERATIONS,
            l0=l0,
            initial_mu=config.CE_RQMC_INITIAL_MU,
            initial_kappa=config.CE_RQMC_INITIAL_KAPPA,
            min_kappa=config.CE_RQMC_MIN_KAPPA,
            pilot_base_seed=config.CE_RQMC_PILOT_BASE_SEED,
            convergence_tolerance=config.CE_RQMC_PARAMETER_TOLERANCE,
            post_target_updates=config.CE_RQMC_POST_TARGET_UPDATES,
            min_calibration_ess=config.CE_RQMC_MIN_CALIBRATION_ESS,
            min_elite_ess=config.CE_RQMC_MIN_ELITE_ESS,
            max_pilot_retries=config.CE_RQMC_MAX_PILOT_RETRIES,
            show_progress=True,
        )
        calibration = engine.last_calibration
        print(
            f"calibration: iterations={calibration['iterations']}, "
            f"target_reached={calibration['target_reached']}, "
            f"d={calibration['parameter_change']:.3g}, "
            f"conditioning ESS={calibration['conditioning_ess']:.1f}, "
            f"elite ESS={calibration['elite_ess']:.1f}, "
            f"h_cal/u_i={calibration['calibration_bandwidth'] / context.portfolio.u[context.target_i]:.3g}"
        )
        estimates = []
        replications = tqdm(
            range(config.CE_RQMC_REPLICAS),
            desc=f"CE-RQMC {mode} replications",
            unit="run",
        )
        for replication in replications:
            estimate = engine.estimate_risk_measures(
                config.CE_RQMC_FINAL_SIZE,
                mu,
                kappa,
                a_val,
                context.beta,
                seed=config.CE_RQMC_BASE_SEED + replication,
                defensive_probability=config.CE_RQMC_DEFENSIVE_PROBABILITY,
            )
            estimates.append(estimate)
            replications.set_postfix(
                CoVaR=f"{estimate[0]:.4f}",
                CoES=f"{estimate[1]:.4f}",
                ESS=f"{estimate[2]:.0f}",
            )
        estimates = np.asarray(estimates)
        means = estimates.mean(axis=0)
        standard_errors = estimates[:, :2].std(axis=0, ddof=1) / np.sqrt(
            config.CE_RQMC_REPLICAS
        )
        print(
            f"{mode} (final h/u_i={ratio:.3g}): "
            f"CoVaR={means[0]:.8f} (SE={standard_errors[0]:.3g}), "
            f"CoES={means[1]:.8f} (SE={standard_errors[1]:.3g}), "
            f"mean ESS={means[2]:.1f}, mu={mu:.4f}, kappa={kappa:.4f}, "
            f"delta={config.CE_RQMC_DEFENSIVE_PROBABILITY:.2f}, "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )


if __name__ == "__main__":
    main()
