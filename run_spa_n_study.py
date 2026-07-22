#!/usr/bin/env python3
"""Measure Hybrid-SPA CoVaR+CoES wall time as portfolio size N changes."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

import config


# Change only this string to switch the conditioning case.
CONDITION_CASE = "inequality"   # inequality

N_VECTOR = (
    # 50,
    100,
    # 150,
    200,
    # 250,
    300,
    # 350,
    400,
    # 450,
    500,
    # 550,
    600,
    # 650,
    700,
    # 750,
    800,
    # 850,
    900,
    # 950,
    1000,
)
COPULA_TYPES = ("Gaussian", "t", "Skew-t")
FIELDNAMES = ("copula_type", "n", "xi", "covar_coes_wall_seconds")


def _entry_module(condition_case: str):
    if condition_case == "equality":
        return importlib.import_module("main_coes")
    if condition_case == "inequality":
        return importlib.import_module("main_ineq_coes")
    raise ValueError('CONDITION_CASE must be either "equality" or "inequality"')


def _base_portfolio() -> dict:
    """Take an immutable snapshot of the configured maximum-size portfolio."""
    return {
        name: np.asarray(getattr(config, name)).copy()
        for name in ("u", "a", "b", "p", "rho")
    }


def _configure_portfolio(base: dict, n: int) -> None:
    """Use an evenly spaced subset so each N preserves the baseline PD range."""
    maximum_n = len(base["p"])
    if not 1 <= n <= maximum_n:
        raise ValueError(f"N={n} must be between 1 and {maximum_n}")
    indices = np.rint(np.linspace(0, maximum_n - 1, n)).astype(int)
    config.N = int(n)
    for name, values in base.items():
        setattr(config, name, values[indices].copy())
    config.target_i = int(n // 2)


def _run_one(entry, base: dict, copula_type: str, n: int, report_stage):
    config.COPULA_TYPE = copula_type
    _configure_portfolio(base, n)
    entry.get_engine.cache_clear()
    engine = None
    try:
        engine = entry.get_engine()
        a_val = engine.individual_var()
        _, _, wall_seconds = entry.calculate_spa_result(
            progress_callback=report_stage,
            engine=engine,
            a_val=a_val,
            return_wall_time=True,
        )
        return {
            "copula_type": copula_type,
            "n": int(n),
            "xi": float(config.XI),
            "covar_coes_wall_seconds": wall_seconds,
        }
    finally:
        if engine is not None:
            engine.close()
        entry.get_engine.cache_clear()


def _write_results(path: Path, rows: list[dict]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "copula_type": row["copula_type"],
                "n": row["n"],
                "xi": f"{row['xi']:.2f}",
                "covar_coes_wall_seconds": (
                    f"{row['covar_coes_wall_seconds']:.6f}"
                ),
            }
            for row in rows
        )
    temporary_path.replace(path)


def _restore_config(snapshot: dict) -> None:
    for name, value in snapshot.items():
        setattr(config, name, value)


def main() -> None:
    condition_case = CONDITION_CASE.strip().lower()
    entry = _entry_module(condition_case)
    output_path = Path(__file__).with_name(
        f"spa_n_{condition_case}_wall_time.csv"
    )
    base = _base_portfolio()
    config_snapshot = {
        "N": config.N,
        "u": config.u,
        "a": config.a,
        "b": config.b,
        "p": config.p,
        "rho": config.rho,
        "target_i": config.target_i,
        "COPULA_TYPE": config.COPULA_TYPE,
    }
    experiments = [
        (copula_type, n)
        for copula_type in COPULA_TYPES
        for n in N_VECTOR
    ]
    rows = []

    print(
        f"SPA N study: case={condition_case}, entry={entry.__name__}.py, "
        f"experiments={len(experiments)}"
    )
    try:
        with tqdm(
            total=len(experiments) * 3,
            desc=f"SPA {condition_case}",
            unit="stage",
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
        ) as progress:
            for case_number, (copula_type, n) in enumerate(experiments, start=1):
                def report_stage(stage, *, _case=case_number, _model=copula_type, _n=n):
                    progress.update(1)
                    progress.set_postfix_str(
                        f"case={_case}/{len(experiments)}, model={_model}, "
                        f"N={_n}, {stage}"
                    )
                    progress.refresh()
                    tqdm.write(
                        f"[{_case}/{len(experiments)}] model={_model}, "
                        f"N={_n}: {stage}",
                        file=sys.stdout,
                    )

                tqdm.write(
                    f"[{case_number}/{len(experiments)}] "
                    f"model={copula_type}, N={n}: starting",
                    file=sys.stdout,
                )
                row = _run_one(entry, base, copula_type, n, report_stage)
                rows.append(row)
                _write_results(output_path, rows)
                tqdm.write(
                    f"[{case_number}/{len(experiments)}] model={copula_type}, "
                    f"N={n}: {row['covar_coes_wall_seconds']:.3f}s, saved",
                    file=sys.stdout,
                )
    finally:
        _restore_config(config_snapshot)

    print("\nFinal CoVaR+CoES wall times")
    print(f"{'model':<10} {'N':>6} {'wall time (s)':>15}")
    for row in rows:
        print(
            f"{row['copula_type']:<10} {row['n']:>6d} "
            f"{row['covar_coes_wall_seconds']:>15.6f}"
        )
    print(f"\nCSV result: {output_path.resolve()}")


if __name__ == "__main__":
    main()
