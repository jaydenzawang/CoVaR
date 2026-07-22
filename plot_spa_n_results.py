#!/usr/bin/env python3
"""Plot Hybrid-SPA CoVaR+CoES wall time against portfolio size N."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = {
    "equality": "spa_n_equality_wall_time.csv",
    "inequality": "spa_n_inequality_wall_time.csv",
}
OUTPUT_PNG = "spa_n_computational_speed.png"
OUTPUT_PDF = "spa_n_computational_speed.pdf"
REQUIRED_COLUMNS = {"copula_type", "n", "covar_coes_wall_seconds"}
MODEL_ORDER = ("Gaussian", "t", "Skew-t")


def _read_result(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(sorted(missing))}"
        )
    data = data.copy()
    data["n"] = pd.to_numeric(data["n"], errors="raise")
    data["covar_coes_wall_seconds"] = pd.to_numeric(
        data["covar_coes_wall_seconds"], errors="raise"
    )
    if data.empty:
        raise ValueError(f"{path.name} contains no results")
    if data[["n", "covar_coes_wall_seconds"]].isna().any().any():
        raise ValueError(f"{path.name} contains missing numeric values")
    if data.duplicated(["copula_type", "n"]).any():
        raise ValueError(f"{path.name} contains duplicate copula_type/N rows")
    return data.sort_values(["copula_type", "n"])


def _plot_case(axis, data: pd.DataFrame, case: str) -> None:
    styles = {
        "Gaussian": {"marker": "^", "linestyle": ":"},
        "t": {"marker": "o", "linestyle": "-"},
        "Skew-t": {"marker": "s", "linestyle": "--"},
    }
    present = list(dict.fromkeys(data["copula_type"]))
    models = [model for model in MODEL_ORDER if model in present]
    models.extend(model for model in present if model not in models)
    for model in models:
        subset = data.loc[data["copula_type"] == model].sort_values("n")
        axis.plot(
            subset["n"],
            subset["covar_coes_wall_seconds"],
            linewidth=1.8,
            markersize=4.5,
            label=model,
            **styles.get(model, {"marker": "D", "linestyle": "-."}),
        )
    condition = r"$\mathcal{C}_i^= = \{L_i=a\}$" if case == "equality" else (
        r"$\mathcal{C}_i = \{L_i\geq a\}$"
    )
    axis.set_title(f"Computational speed: {condition}")
    axis.set_xlabel(r"Number of obligors $N$")
    axis.set_ylabel("CoVaR + CoES wall time (seconds)")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False)


def make_plots(directory: Path = BASE_DIR) -> tuple[Path, Path]:
    available = []
    for case, filename in INPUT_FILES.items():
        path = directory / filename
        if path.exists():
            available.append((case, _read_result(path)))
    if not available:
        expected = ", ".join(INPUT_FILES.values())
        raise FileNotFoundError(
            f"No timing CSV was found in {directory}. Expected: {expected}"
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        1,
        len(available),
        figsize=(7 * len(available), 5),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (case, data) in enumerate(available):
        _plot_case(axes[0, column], data, case)

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
