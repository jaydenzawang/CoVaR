#!/usr/bin/env python3
"""Run the SPA CoVaR/CoES target-PD sensitivity study.

Change only ``CONDITION_CASE`` below to switch between equality and
inequality conditioning.  Results are checkpointed after every target
obligor in a plotting-ready long-format CSV.

The labels 100, ..., 900 reproduce the zero-based ``target_i`` convention
used by the existing code and by the supplied first two table entries.  The
label 1000 denotes the final obligor and therefore maps to array index 999.
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

OBLIGOR_LABELS = tuple(range(100, 1001, 100))
COPULA_TYPES = ("Gaussian", "t", "Skew-t")
FIELDNAMES = (
    "copula_type",
    "i",
    "target_index",
    "p_i_percent",
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


def _target_index(obligor_label: int, portfolio_size: int) -> int:
    """Translate table labels to valid indices, including N -> N - 1."""
    if not 0 < obligor_label <= portfolio_size:
        raise ValueError(
            f"obligor label i={obligor_label} must be in [1, {portfolio_size}]"
        )
    return min(int(obligor_label), portfolio_size - 1)


def _run_one(entry, copula_type: str, obligor_label: int, report_stage):
    target_index = _target_index(obligor_label, int(config.N))
    config.COPULA_TYPE = copula_type
    config.target_i = target_index

    # Every target and copula needs a newly calibrated/cached engine.
    entry.get_engine.cache_clear()
    engine = None
    try:
        engine = entry.get_engine()
        a_val = engine.individual_var()
        covar, coes = entry.calculate_spa_result(
            progress_callback=report_stage,
            engine=engine,
            a_val=a_val,
        )
        return {
            "copula_type": copula_type,
            "i": int(obligor_label),
            "target_index": target_index,
            "p_i_percent": 100.0 * float(config.p[target_index]),
            "xi": float(config.XI),
            "covar": covar,
            "coes": coes,
        }
    finally:
        if engine is not None:
            engine.close()
        entry.get_engine.cache_clear()


def _write_results(path: Path, rows: list[dict]) -> None:
    """Checkpoint a complete and internally consistent CSV."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {
                "copula_type": row["copula_type"],
                "i": row["i"],
                "target_index": row["target_index"],
                "p_i_percent": f"{row['p_i_percent']:.10f}",
                "xi": f"{row['xi']:.2f}",
                "covar": f"{row['covar']:.10f}",
                "coes": f"{row['coes']:.10f}",
            }
            for row in rows
        )
    temporary_path.replace(path)


def _print_summary(rows: list[dict], output_path: Path, condition_case: str) -> None:
    covar_label = "CoVaR^=" if condition_case == "equality" else "CoVaR"
    coes_label = "CoES^=" if condition_case == "equality" else "CoES"
    print("\nFinal SPA target-PD sensitivity results")
    print(
        f"{'model':<10} {'i':>5} {'index':>6} {'p_i (%)':>10} "
        f"{covar_label:>12} {coes_label:>12}"
    )
    for row in rows:
        print(
            f"{row['copula_type']:<10} {row['i']:>5d} "
            f"{row['target_index']:>6d} {row['p_i_percent']:>10.4f} "
            f"{row['covar']:>12.4f} {row['coes']:>12.4f}"
        )
    print(f"\nCSV result: {output_path.resolve()}")


def main() -> None:
    condition_case = CONDITION_CASE.strip().lower()
    entry = _entry_module(condition_case)
    output_path = Path(__file__).with_name(
        f"spa_p_{condition_case}_results.csv"
    )
    config_snapshot = {
        "COPULA_TYPE": config.COPULA_TYPE,
        "target_i": config.target_i,
    }
    experiments = [
        (copula_type, obligor_label)
        for copula_type in COPULA_TYPES
        for obligor_label in OBLIGOR_LABELS
    ]
    rows = []

    print(
        f"SPA p_i study: case={condition_case}, entry={entry.__name__}.py, "
        f"experiments={len(experiments)}, N={config.N}, xi={config.XI:.2f}"
    )
    try:
        with tqdm(
            total=len(experiments) * 3,
            desc=f"SPA p_i {condition_case}",
            unit="stage",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.1,
            leave=True,
        ) as progress:
            for case_number, (copula_type, obligor_label) in enumerate(
                experiments, start=1
            ):
                target_index = _target_index(obligor_label, int(config.N))

                def report_stage(
                    stage,
                    *,
                    _case=case_number,
                    _model=copula_type,
                    _label=obligor_label,
                ):
                    progress.update(1)
                    progress.set_postfix_str(
                        f"case={_case}/{len(experiments)}, model={_model}, "
                        f"i={_label}, {stage}"
                    )
                    progress.refresh()
                    tqdm.write(
                        f"[{_case}/{len(experiments)}] model={_model}, "
                        f"i={_label}: {stage}",
                        file=sys.stdout,
                    )

                tqdm.write(
                    f"[{case_number}/{len(experiments)}] model={copula_type}, "
                    f"i={obligor_label}, target_index={target_index}, "
                    f"p_i={100.0 * config.p[target_index]:.4f}%: starting",
                    file=sys.stdout,
                )
                row = _run_one(
                    entry, copula_type, obligor_label, report_stage
                )
                rows.append(row)
                _write_results(output_path, rows)
                tqdm.write(
                    f"[{case_number}/{len(experiments)}] model={copula_type}, "
                    f"i={obligor_label}: saved to CSV",
                    file=sys.stdout,
                )
    finally:
        config.COPULA_TYPE = config_snapshot["COPULA_TYPE"]
        config.target_i = config_snapshot["target_i"]

    _print_summary(rows, output_path, condition_case)


if __name__ == "__main__":
    main()
