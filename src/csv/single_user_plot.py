#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_single_user(df, metric, save_path=None):
    """
    Plot a bar comparison of models for C=1 only.
    """

    # Filter single-user
    df_single = df[df["concurrency"] == 1]

    if df_single.empty:
        raise ValueError("No rows found with concurrency == 1")

    df_single = df_single.sort_values(metric)

    plt.figure()
    plt.bar(df_single["model"], df_single[metric])

    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison (Single User)")
    plt.xticks(rotation=45)
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ttft_ms", "e2e_ms", "tpot_ms", "itl_mean_ms", "itl_p95_ms"]
    )
    parser.add_argument("--outdir", default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Skipping {metric} (not found in CSV)")
            continue

        save_path = None
        if args.outdir:
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            save_path = f"{args.outdir}/{metric}_single_user.png"

        plot_single_user(df, metric, save_path)


if __name__ == "__main__":
    main()
