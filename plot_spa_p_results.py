#!/usr/bin/env python3
"""Plot equality- and inequality-conditioned SPA target-PD results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = {
    "equality": "spa_p_equality_results.csv",
    "inequality": "spa_p_inequality_results.csv",
}
OUTPUT_PNG = "spa_p_covar_coes.png"
OUTPUT_PDF = "spa_p_covar_coes.pdf"
REQUIRED_COLUMNS = {
    "copula_type",
    "i",
    "target_index",
    "p_i_percent",
    "covar",
    "coes",
}
MODEL_ORDER = ("Gaussian", "t", "Skew-t")


def _read_result(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing))}"
        )

    columns = [
        "copula_type",
        "i",
        "target_index",
        "p_i_percent",
        "covar",
        "coes",
    ]
    data = data.loc[:, columns].copy()
    for column in ("i", "target_index", "p_i_percent", "covar", "coes"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.empty:
        raise ValueError(f"{path.name} contains no results")
    if data.iloc[:, 1:].isna().any().any():
        raise ValueError(f"{path.name} contains missing numeric values")
    if data.duplicated(["copula_type", "i"]).any():
        raise ValueError(f"{path.name} contains duplicate copula_type/i rows")

    # p_i is a property of the target and must agree across copulas.
    pd_counts = data.groupby("i")["p_i_percent"].nunique()
    if (pd_counts > 1).any():
        raise ValueError(f"{path.name} has inconsistent p_i values across copulas")
    return data.sort_values(["copula_type", "p_i_percent"])


def _plot_measure(axis, data: pd.DataFrame, measure: str, case: str) -> None:
    styles = {
        "Gaussian": {"marker": "^", "linestyle": ":"},
        "t": {"marker": "o", "linestyle": "-"},
        "Skew-t": {"marker": "s", "linestyle": "--"},
    }
    present_models = list(dict.fromkeys(data["copula_type"]))
    models = [model for model in MODEL_ORDER if model in present_models]
    models.extend(model for model in present_models if model not in models)

    for model in models:
        subset = data.loc[data["copula_type"] == model].sort_values(
            "p_i_percent"
        )
        axis.plot(
            subset["p_i_percent"],
            subset[measure],
            linewidth=1.8,
            markersize=4.5,
            label=model,
            **styles.get(model, {"marker": "D", "linestyle": "-."}),
        )

    case_symbol = "=" if case == "equality" else r"\geq"
    measure_label = "CoVaR" if measure == "covar" else "CoES"
    axis.set_title(f"{case.capitalize()}: {measure_label}")
    axis.set_xlabel(r"Unconditional default probability $p_i$ (\%)")
    axis.set_ylabel(rf"$\widehat{{\mathrm{{{measure_label}}}}}^{{{case_symbol}}}$")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False)


def make_plots(directory: Path = BASE_DIR) -> tuple[Path, Path]:
    """Draw all available equality/inequality panels and return output paths."""
    available = []
    for case, filename in INPUT_FILES.items():
        path = directory / filename
        if path.exists():
            available.append((case, _read_result(path)))

    if not available:
        expected = ", ".join(INPUT_FILES.values())
        raise FileNotFoundError(
            f"No result CSV was found in {directory}. Expected: {expected}"
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        nrows=len(available),
        ncols=2,
        figsize=(12, 4.5 * len(available)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (case, data) in enumerate(available):
        _plot_measure(axes[row_index, 0], data, "covar", case)
        _plot_measure(axes[row_index, 1], data, "coes", case)

    png_path = directory / OUTPUT_PNG
    pdf_path = directory / OUTPUT_PDF
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = make_plots()
    print(f"PNG figure: {png_path}")
    print(f"PDF figure: {pdf_path}")


if __name__ == "__main__":
    main()
