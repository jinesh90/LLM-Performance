#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_single_metric(df, metric, log_scale=False, save_path=None):
    plt.figure()

    models = sorted(df["model"].unique())

    for model in models:
        subset = df[df["model"] == model].sort_values("concurrency")

        plt.plot(
            subset["concurrency"],
            subset[metric],
            marker="o",
            label=model
        )

    plt.xlabel("Concurrency Level")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Concurrency")

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--log", action="store_true")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "ttft_ms",
            "e2e_ms",
            "tpot_ms",
            "itl_mean_ms",
            "itl_p95_ms"
        ]
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Basic validation
    required_cols = ["model", "concurrency"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Warning: {metric} not found in CSV, skipping.")
            continue

        save_path = None
        if args.outdir:
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            suffix = "_log" if args.log else ""
            save_path = f"{args.outdir}/{metric}{suffix}.png"

        plot_single_metric(
            df,
            metric=metric,
            log_scale=args.log,
            save_path=save_path
        )


if __name__ == "__main__":
    main()
