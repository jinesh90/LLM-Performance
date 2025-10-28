#!/usr/bin/env python3
import json
import time
import requests
import argparse
import os
from pathlib import Path

API = "http://127.0.0.1:11434/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}


def chat(model, messages, t=0.0, max_tokens=512, seed=7):
    """Send a chat completion request to Ollama API."""
    t0 = time.time()
    r = requests.post(
        API,
        headers=HEADERS,
        json={
            "model": model,
            "messages": messages,
            "temperature": t,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": False,
        },
    )
    t1 = time.time()
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
    out = r.json()
    txt = out["choices"][0]["message"]["content"]
    return {"text": txt, "e2e_ms": (t1 - t0) * 1000}


def ask(model, prompt):
    """Wrap a single-turn user prompt."""
    msgs = [{"role": "user", "content": prompt}]
    return chat(model, msgs)


def run_file(model, input_path, output_path):
    """Run evaluation for a JSONL dataset."""
    outs = []
    total = sum(1 for _ in open(input_path))
    print(f"\n🔹 Running {model} on {input_path} ({total} examples)...\n")

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            ex = json.loads(line)
            prompt = ex.get("question") or ex.get("prompt") or ex.get("input")
            if not prompt:
                print(f"[WARN] Missing prompt in line {i}")
                continue

            try:
                y = ask(model, prompt)
            except Exception as e:
                print(f"[ERROR] Line {i}: {e}")
                continue

            rec = {
                "id": ex.get("id", i),
                "prompt": prompt,
                "pred": y["text"],
                "ref": ex.get("answer"),
                "e2e_ms": y["e2e_ms"],
            }
            fout.write(json.dumps(rec) + "\n")

            if i % 5 == 0 or i == total:
                print(f"  ✅ Completed {i}/{total}")

    print(f"\n✅ Output saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses on a dataset.")
    parser.add_argument("--model", required=True, help="Model name, e.g., mistral:7b or llama3.1:8b")
    parser.add_argument("--input", required=True, help="Input JSONL file (e.g., gsm8k.jsonl)")
    parser.add_argument("--output", help="Output file path (optional)")
    args = parser.parse_args()

    model_sanitized = args.model.replace(":", "_").replace("/", "_")
    dataset_name = Path(args.input).stem
    output_file = args.output or f"pred_{dataset_name}_{model_sanitized}.jsonl"

    os.makedirs(Path(output_file).parent, exist_ok=True)
    run_file(args.model, args.input, output_file)


if __name__ == "__main__":
    main()