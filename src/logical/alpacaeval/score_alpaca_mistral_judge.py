#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
import requests

# ========================
# CONFIGURATION
# ========================
# Example:
# export MISTRAL_API_KEY=sk-xxxxxx
API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    raise SystemExit("❌ Please set MISTRAL_API_KEY environment variable")

API_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ========================
# FUNCTIONS
# ========================
def judge_pair(prompt, ans_a, ans_b, model="mistral-large-latest"):
    """Send A vs B comparison to Mistral judge model."""
    question = f"""You are a strict evaluator.
Compare two model answers to the same instruction below.

INSTRUCTION:
{prompt}

=== Response A ===
{ans_a}

=== Response B ===
{ans_b}

Which response is more helpful, accurate, and concise?
Reply ONLY with "A", "B", or "Tie"."""
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": question}]
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:300]}")
    out = resp.json()
    res = out["choices"][0]["message"]["content"].strip()
    # normalize
    if "A" in res and "B" in res:
        res = "Tie"
    if res not in ["A", "B", "Tie"]:
        res = "Tie"
    return res


def compare_files(file_a, file_b, model="mistral-large-latest"):
    """Compare predictions from two model outputs JSONL files."""
    a = [json.loads(x) for x in open(file_a)]
    b = [json.loads(x) for x in open(file_b)]
    assert len(a) == len(b), "Files must have same number of prompts"

    wins_a = wins_b = ties = 0
    results = []

    for ex_a, ex_b in tqdm(zip(a, b), total=len(a), desc="Judging"):
        prompt = ex_a.get("prompt") or ex_a.get("instruction", "")
        res = judge_pair(prompt, ex_a["pred"], ex_b["pred"], model=model)
        results.append({"id": ex_a.get("id"), "judge": res})
        if res == "A": wins_a += 1
        elif res == "B": wins_b += 1
        else: ties += 1

    total = wins_a + wins_b + ties
    win_rate = (wins_a + 0.5 * ties) / total * 100 if total else 0

    print("\n===============================")
    print(f"🎯 Mistral Judge Model: {model}")
    print(f"Judged {total} prompts")
    print(f"A wins: {wins_a}, B wins: {wins_b}, Ties: {ties}")
    print(f"Win-rate (A): {win_rate:.2f}%")
    print("===============================")

    return results


def main():
    ap = argparse.ArgumentParser(description="Pairwise judgment using Mistral API")
    ap.add_argument("--a", required=True, help="Path to first model's predictions JSONL")
    ap.add_argument("--b", required=True, help="Path to second model's predictions JSONL")
    ap.add_argument("--judge-model", default="mistral-large-latest", help="Judge model (default: mistral-large-latest)")
    ap.add_argument("--save", help="Optional output JSONL with judge results")
    args = ap.parse_args()

    results = compare_files(args.a, args.b, model=args.judge_model)
    if args.save:
        with open(args.save, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"💾 Saved detailed results to {args.save}")


if __name__ == "__main__":
    main()
