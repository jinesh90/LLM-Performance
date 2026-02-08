#!/usr/bin/env python3

import argparse
import pandas as pd
import glob
import re
from pathlib import Path


def extract_model_and_concurrency(filename):
    """
    Extract model name and concurrency from filename.
    Expected format: modelname_cX.csv
    Example: mistral7b_c8.csv
    """
    pattern = re.compile(r"(.+)_c(\d+)\.csv$")
    match = pattern.search(filename)

    if not match:
        return None, None

    model = match.group(1)
    concurrency = int(match.group(2))

    return model, concurrency


def aggregate_csvs(input_dir, output_path):
    """
    Aggregate all *_cX.csv files into a single averaged CSV.
    """

    files = glob.glob(f"{input_dir}/*_c*.csv")

    if not files:
        raise ValueError("No matching CSV files found.")

    data_frames = []

    for file in files:
        model, concurrency = extract_model_and_concurrency(Path(file).name)

        if model is None:
            continue

        df = pd.read_csv(file)

        df["model"] = model
        df["concurrency"] = concurrency

        data_frames.append(df)

    combined = pd.concat(data_frames, ignore_index=True)

    # Metrics to average
    metrics = [
        "ttft_ms",
        "e2e_ms",
        "tpot_ms",
        "itl_mean_ms",
        "itl_p95_ms",
        "prefill_ms",
        "tps",
        "prompt_tokens",
        "output_tokens"
    ]

    # Keep only columns that exist
    metrics = [m for m in metrics if m in combined.columns]

    # Aggregate
    summary = (
        combined
        .groupby(["model", "concurrency"])[metrics]
        .mean()
        .reset_index()
    )

    summary = summary.sort_values(["model", "concurrency"])

    summary.to_csv(output_path, index=False)

    print(f"Final averaged CSV saved to: {output_path}")
    print(f"Models found: {summary['model'].unique()}")
    print(f"Concurrency levels: {sorted(summary['concurrency'].unique())}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM benchmark CSV files")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing model_cX.csv files"
    )
    parser.add_argument(
        "--output",
        default="final_averaged_results.csv",
        help="Output aggregated CSV file"
    )

    args = parser.parse_args()

    aggregate_csvs(args.input_dir, args.output)


if __name__ == "__main__":
    main()
