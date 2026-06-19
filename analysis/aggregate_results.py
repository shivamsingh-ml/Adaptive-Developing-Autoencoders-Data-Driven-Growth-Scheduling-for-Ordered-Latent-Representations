import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results/raw")


def load_runs():
    rows = []

    for file in RESULTS_DIR.glob("*.json"):
        with open(file, "r") as f:
            run = json.load(f)

        rows.append(
            {
                "dataset": run["dataset"],
                "experiment": run["experiment"],
                "seed": run["seed"],
                "linear_probe_acc": run.get("linear_probe_acc"),
                "knn5_acc": run.get("knn5_acc"),
                "final_intrinsic_dim": run.get("final_intrinsic_dim"),
                "final_intrinsic_dim_Facco": run.get("final_intrinsic_dim_Facco"),
                "final_latent_dim": run.get("final_latent_dim"),
                "n_growth_events": run.get("n_growth_events"),
            }
        )

    return pd.DataFrame(rows)


def mean_std(series):
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"


def main():
    df = load_runs()

    print("\nLoaded runs:")
    print(df)

    grouped = []

    for (dataset, experiment), group in df.groupby(
        ["dataset", "experiment"]
    ):
        grouped.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "n_runs": len(group),
                "linear_probe": mean_std(group["linear_probe_acc"]),
                "knn5": mean_std(group["knn5_acc"]),
                "intrinsic_dim": mean_std(group["final_intrinsic_dim"]),
                "intrinsic_dim_Facco": mean_std(group["final_intrinsic_dim_Facco"]),
            }
        )

    summary = pd.DataFrame(grouped)

    print("\nSummary:")
    print(summary)

    output_dir = Path("results/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "baseline_summary.csv",
        index=False,
    )

    print(
        "\nSaved:",
        output_dir / "baseline_summary.csv"
    )
if __name__ == "__main__":
    main()