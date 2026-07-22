# CoVaR computational modules

This directory implements the model and definitions in `CoVaR_draft.tex`.
Production calculations and simulation benchmarks are deliberately separated.

## Module map

- `config.py`: portfolio and experiment inputs only.
- `model_factory.py`: validates configuration and builds one reusable model context.
- `copula.py`: conditional default probabilities and calibrated default thresholds.
- `factor_integration.py`: probability-space Gauss–Legendre/Rosenblatt integration.
- `loss_model.py`: beta-LGD MGF and its first four derivatives.
- `spa.py`: conditional CDF and direct stop-loss saddlepoint approximations.
- `risk_engine.py`: equality/inequality CoVaR and robust stop-loss CoES.
- `pmc_engine.py`: direct plain-Monte-Carlo benchmark.
- `ce_rqmc_engine.py`: direct CE-RQMC benchmark with paper-consistent scale tilting.
- `main*.py`: small compatibility/command-line entry points.
- `convergence_check.py`: mandatory factor-quadrature refinement check.
- `tests/test_core.py`: formula, boundary and support tests.

## Mathematical conventions

- Positive equality conditioning `L_i=a` is regular conditioning through the
  conditional density; its probability is zero and is reported as `None`.
- Equality at `a=0` means the exact non-default atom `D_i=0`.
- Inequality at `a=0` is the whole sample space and therefore gives
  unconditional portfolio VaR/ES.
- CoES always uses
  `CoVaR + E[(L-CoVaR)^+ | condition] / (1-beta)`.
- The SPA engine is used only by `risk_engine.py`. PMC and CE-RQMC simulate the
  target obligor and use indicator/kernel conditioning weights, so they remain
  independent benchmarks.

## Validation workflow

Run the following before producing numerical tables:

```bash
python -m pytest -q CoVaR_codes/tests/test_core.py
python CoVaR_codes/convergence_check.py
python CoVaR_codes/main_coes.py
python CoVaR_codes/main_ineq_coes.py
```

The two SPA CoES entry points report setup time, CoVaR time, incremental CoES
time conditional on an already known CoVaR, and their combined time. Both CPU
time (`process_time`) and wall time (`perf_counter`) are shown.

For positive equality conditioning, PMC and CE-RQMC each use one configured
final bandwidth ratio. CE-RQMC pilot calibration may temporarily widen its band
when the calibration ESS is too small, but every reported estimate uses its
single configured final bandwidth. Both benchmark engines warn when the implied
effective tail sample size is below 20.

RQMC sample sizes must be powers of two. The current configuration uses 8192
pilot points and 16384 final points per scrambling, with 32 independent
scramblings.

PMC displays both replication and within-replication path progress. CE-RQMC
displays calibration iterations and final scrambled-replication progress using
`tqdm`. Programmatic PMC calls can suppress the bars with
`show_progress=False`.

The Gaussian production grid uses 512 probability-space nodes. The convergence
check compares 256, 512 and 1024 nodes and, for inequality conditioning, also
compares truncated-LGD orders 16, 32 and 64. It fails if the final CoVaR or CoES
refinement exceeds 0.1%.

For Gumbel-dependent t and skew-t models, default thresholds are calibrated on
the same probability-space Rosenblatt rule used by the production factor grid.
This keeps the integrated conditional default probabilities aligned with the
configured marginal PDs, including for strong dependence such as `XI = 3.0`.

CE calibration obtains `l0` from independent nominal RQMC preliminary
randomizations using the calibration band. Each adaptive pilot threshold is a
conditioning-weighted quantile under the current proposal; likelihood ratios
enter the CE parameter update, not the intermediate quantile. Calibration is
capped at five iterations, includes at most the configured number of
post-target stabilization updates, and uses disjoint preliminary, pilot and
final seed ranges.

All user-facing PMC and CE-RQMC tuning parameters now live in `config.py`.
`mc_verify.py` reads the `PMC_*` settings; `benchmark_main.py` reads the
`CE_RQMC_*` settings. The engine modules contain the algorithms but no hidden
simulation-budget, bandwidth, CE-initialization or seed defaults.

For inequality SPA, the truncated beta-LGD integral is evaluated after an exact
conditional-probability transform with endpoint smoothing. CoES evaluates the
stop-loss moment directly rather than subtracting two SPA approximations. Factor nodes are distributed in
deterministic blocks across persistent worker processes; configure the worker
count with `SPA_N_JOBS` in `config.py`, or set it to `1` for a serial audit.
