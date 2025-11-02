#!/usr/bin/env python3
import json
import re
import argparse
import statistics as stats
from pathlib import Path

def last_number(s):
    """Extract the last number (integer/float) from a string."""
    if not s:
        return None
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None

def score_gsm8k(file_path):
    N = 0
    C = 0
    latencies = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            y_pred = last_number(ex.get("pred", ""))
            y_ref = last_number(ex.get("ref", ""))
            if y_pred is not None and y_ref is not None:
                if y_pred == y_ref:
                    C += 1
                N += 1
            if "e2e_ms" in ex:
                latencies.append(ex["e2e_ms"])

    acc = C / N if N else 0
    print("\n📊 GSM8K Evaluation Results")
    print("---------------------------")
    print(f"File          : {file_path}")
    print(f"Examples      : {N}")
    print(f"Correct       : {C}")
    print(f"Accuracy      : {acc*100:.2f}%")

    if latencies:
        print(f"Avg E2E (ms)  : {stats.mean(latencies):.2f}")
        print(f"p95 E2E (ms)  : {stats.quantiles(latencies, n=100)[94]:.2f}")
    print("---------------------------\n")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Score GSM8K predictions (exact numeric match).")
    parser.add_argument("--input", required=True, help="Input JSONL file, e.g. pred_gsm8k_mistral_7b.jsonl")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    score_gsm8k(path)

if __name__ == "__main__":
    main()
