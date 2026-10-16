#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:20:58 2026

@author: jaydenwang
"""

import numpy as np
rng = np.random.default_rng(42)


# 1: Portfolio parameters

N = 500  # Number of financial institution / obligor

# u_i: Exposure of obligor i
u = np.ones(N)

# a_i, b_i: Beta distribution parameters of LGD
a = np.full(N, 2.0)
b = np.full(N, 5.0)

# p_i: Unconditional default probability
p = rng.uniform(0.01, 0.05, N)
p = p[np.argsort(p)]

# rho_i: Factor loadings of systematic factor Z
# rho_i^2=0.50
rho = np.full(N, np.sqrt(0.50))



# 2: Copula models parameters

# Options: 'Gaussian', 't' or 'Skew-t'

COPULA_TYPE = 't'
# Gumbel dependence parameter
# XI = 1.0 independence; XI > 1.0 dependence
XI = 1.0  

#  t and Skew-t copula
# Freedom degree of Lambda
nu = 5.0  

# Skew-t copula
# Skewness or shape parameter of Z
gamma = 0.5



# 3: Main computation, distress scenarios, and confidence levels

# main.py parameters
CONDITION_MODE = "inequality"  # "equality" or "inequality"
SHOW_PROGRESS = False  

# None: Compute VaR from alpha; a number: Use the same fixed distress threshold for all methods.
# SYSTEM uses firm loss L_i; EXPOSURE uses portfolio loss L.
SYSTEM_DISTRESS_THRESHOLD = None
EXPOSURE_DISTRESS_THRESHOLD = None
# All auto exposure thresholds use SPA; benchmark SE excludes threshold error.

target_i = 200  # Zero-based index: 200 is the 201st firm
# print(p[target_i])

alpha = 0.99  # Distress confidence level: CoVaR uses firm VaR; E-CoVaR uses portfolio VaR
beta = 0.99   # Conditional quantile confidence level for both CoVaR / E-CoVaR

# 4: SPA
SPA_GAUSSIAN_FACTOR_ORDER = 512
SPA_T_FACTOR_ORDER = 64       # First factor quadrature order for t / Skew-t
SPA_SCALE_ORDER = 64         # Scale factor quadrature order for t / Skew-t
SPA_LGD_ORDER = 32
SPA_QUANTILE_TOL = 1e-5
SPA_SADDLE_BOUND = 20.0       # Must be within the MGF grid bounds below
SPA_ROOT_TOL = 1e-10
SPA_MEAN_REGION = 1e-2

SPA_EXACT_MGF_RADIUS = 0.15
SPA_TAIL_BOUND_TOL = 1e-12


SPA_N_JOBS = 4
SPA_MGF_GRID_SIZE = 1_501
SPA_MGF_S_MIN = -25.0
SPA_MGF_S_MAX = 25.0


# 5: PMC benchmark settings

PMC_NUM_SIMS = 5_000_000 # 1_000_000     # Paths per independent run
PMC_NUM_RUNS = 10            # Independent runs used to compute standard error
PMC_BATCH_SIZE = 4_096       # Generate paths in batches to limit memory use
PMC_EQUALITY_BANDWIDTH_RATIO = 0.03  # CoVaR/CoES: h = ratio * u_i
PMC_EXPOSURE_EQUALITY_BANDWIDTH_RATIO = 0.03  # E-CoVaR/E-CoES: h = ratio * max(v, 1)
PMC_BASE_SEED = 42           # Run r uses BASE_SEED + r


# 6: CE-RQMC benchmark settings

# The number of Sobol points must be a power of 2.
CE_RQMC_REPLICAS = 32
CE_RQMC_PILOT_SIZE = 262_144 # 8_192
CE_RQMC_FINAL_SIZE = 1_048_576 # 32_768 # 16_384

CE_RQMC_ELITE_FRACTION = 0.10
CE_RQMC_MAX_ITERATIONS = 5  # Hui Wang: iterative CE uses at most five updates
# Use one final equality bandwidth ratio; do not loop over several ratios.
CE_RQMC_EQUALITY_BANDWIDTH_RATIO = 0.03  # CoVaR/CoES: h = ratio * u_i
CE_RQMC_EXPOSURE_EQUALITY_BANDWIDTH_RATIO = 0.03  # Exposure: h = ratio * max(v, 1)

CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER = 2.0
# Independent preliminary RQMC randomizations used to estimate conditional l0.
CE_RQMC_PRELIMINARY_SIZE = 8_192
CE_RQMC_PRELIMINARY_REPLICAS = 64
CE_RQMC_PRELIMINARY_BASE_SEED = 30_000
CE_RQMC_MAX_PILOT_RETRIES = 2
CE_RQMC_MIN_CALIBRATION_ESS = 64
CE_RQMC_MIN_ELITE_ESS = 5
CE_RQMC_PARAMETER_TOLERANCE = 0.02
CE_RQMC_POST_TARGET_UPDATES = 1
CE_RQMC_INITIAL_MU = 0.0
CE_RQMC_INITIAL_KAPPA = 1.0
CE_RQMC_MIN_KAPPA = 0.05

# The final estimate uses delta*f_nominal + (1-delta)*g_CE; fitting still uses pure CE.
# Larger delta is more robust but makes tilting less efficient. Start with 0.10, then test 0.05/0.10/0.20.
CE_RQMC_DEFENSIVE_PROBABILITY = 0.10
CE_RQMC_PILOT_BASE_SEED = 10_000
CE_RQMC_BASE_SEED = 1_000

# PMC/CE-RQMC warns of low accuracy if ESS × (1-beta) is below this value.
BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES = 20
