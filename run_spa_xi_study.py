#!/usr/bin/env python3
"""Run the SPA CoVaR/CoES xi study for the t and skew-t copulas.

Change only ``CONDITION_CASE`` below to switch between the equality and
inequality experiments. Results are checkpointed after every xi value in a
plotting-ready long-format CSV with columns ``copula_type,xi,covar,coes``.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
import sys

from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# User control: use "equality" or "inequality".
# ---------------------------------------------------------------------------
CONDITION_CASE = "inequality"

XI_VECTOR = (
    1.00,
    1.02,
    1.04,
    1.06,
    1.08,
    1.10,
    1.20,
    1.40,
    1.60,
    1.80,
    2.00,
    2.20,
    2.40,
    2.60,
    2.80,
    3.00,
)
COPULA_TYPES = ("t", "Skew-t")


FIELDNAMES = (
    "copula_type",
    "xi",
    "covar",
    "coes",
)


def _entry_module(condition_case: str):
    """Select the same engine entry point as the corresponding main script."""
    if condition_case == "equality":
        return importlib.import_module("main_coes")
    if condition_case == "inequality":
        return importlib.import_module("main_ineq_coes")
    raise ValueError('CONDITION_CASE must be either "equality" or "inequality"')


def _run_one(entry, copula_type: str, xi: float, progress_callback=None):
    config.COPULA_TYPE = copula_type
    config.XI = float(xi)

    # main.py/main_ineq.py cache their expensive engine.  Clearing the cache is
    # essential because every (model, xi) pair needs a newly calibrated model.
    entry.get_engine.cache_clear()
    engine = None
    try:
        covar, coes = entry.calculate_spa_result(progress_callback)
        engine = entry.get_engine()
        return {
            "copula_type": copula_type,
            "xi": float(xi),
            "covar": covar,
            "coes": coes,
        }
    finally:
        if engine is not None:
            engine.close()
        entry.get_engine.cache_clear()


def _write_results(path: Path, rows: list[dict]) -> None:
    """Checkpoint a consistent, plotting-ready long-format CSV."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "copula_type": row["copula_type"],
                "xi": f"{row['xi']:.2f}",
                "covar": f"{row['covar']:.10f}",
                "coes": f"{row['coes']:.10f}",
            }
            for row in rows
        )
    temporary_path.replace(path)


def _print_summary(rows: list[dict], output_path: Path) -> None:
    covar_label = "CoVaR^=" if CONDITION_CASE == "equality" else "CoVaR"
    coes_label = "CoES^=" if CONDITION_CASE == "equality" else "CoES"
    print("\nFinal SPA results")
    print(f"{'model':<8} {'xi':>5} {covar_label:>12} {coes_label:>12}")
    for row in rows:
        print(
            f"{row['copula_type']:<8} {row['xi']:>5.2f} "
            f"{row['covar']:>12.8f} {row['coes']:>12.8f}"
        )
    print(f"\nCSV result: {output_path.resolve()}")


def main() -> None:
    condition_case = CONDITION_CASE.strip().lower()
    entry = _entry_module(condition_case)
    output_path = Path(__file__).with_name(
        f"spa_xi_{condition_case}_results.csv"
    )
    original_copula_type = config.COPULA_TYPE
    original_xi = config.XI
    rows = []
    experiments = [
        (copula_type, xi)
        for copula_type in COPULA_TYPES
        for xi in XI_VECTOR
    ]

    print(
        f"SPA xi study: case={condition_case}, "
        f"entry={entry.__name__}.py, experiments={len(experiments)}"
    )
    try:
        with tqdm(
            total=len(experiments) * 3,
            desc=f"SPA {condition_case}",
            unit="stage",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.1,
            leave=True,
        ) as progress:
            for case_number, (copula_type, xi) in enumerate(experiments, start=1):
                def report_stage(stage, *, _model=copula_type, _xi=xi, _case=case_number):
                    progress.update(1)
                    progress.set_postfix_str(
                        f"case={_case}/{len(experiments)}, "
                        f"model={_model}, xi={_xi:.2f}, {stage}"
                    )
                    progress.refresh()
                    tqdm.write(
                        f"[{_case}/{len(experiments)}] "
                        f"model={_model}, xi={_xi:.2f}: {stage}",
                        file=sys.stdout,
                    )

                tqdm.write(
                    f"[{case_number}/{len(experiments)}] "
                    f"model={copula_type}, xi={xi:.2f}: starting",
                    file=sys.stdout,
                )
                progress.set_postfix_str(
                    f"case={case_number}/{len(experiments)}, "
                    f"model={copula_type}, xi={xi:.2f}, building model"
                )
                progress.refresh()
                row = _run_one(entry, copula_type, xi, report_stage)
                rows.append(row)
                _write_results(output_path, rows)
                tqdm.write(
                    f"[{case_number}/{len(experiments)}] "
                    f"model={copula_type}, xi={xi:.2f}: saved to CSV",
                    file=sys.stdout,
                )
    finally:
        config.COPULA_TYPE = original_copula_type
        config.XI = original_xi

    _print_summary(rows, output_path)


if __name__ == "__main__":
    main()
