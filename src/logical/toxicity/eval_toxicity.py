#!/usr/bin/env python3
import json, time, requests, argparse, os
from pathlib import Path

API = "http://127.0.0.1:11434/v1/chat/completions"
HEADERS = {"Content-Type":"application/json"}

def chat(model, messages, temperature=0.7, max_tokens=256, seed=7):
    t0 = time.time()
    r = requests.post(API, headers=HEADERS, json={
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False
    })
    t1 = time.time()
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
    resp = r.json()
    text = resp["choices"][0]["message"]["content"]
    return {"text": text, "e2e_ms": (t1-t0)*1000}

def run_rtp(model, input_path, output_path, temperature=0.7):
    total = sum(1 for _ in open(input_path))
    print(f"Running {model} on {input_path} ({total} prompts)...")
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            ex = json.loads(line)
            prompt = ex["prompt"]
            messages = [{"role": "user", "content": prompt}]
            try:
                res = chat(model, messages, temperature=temperature)
            except Exception as e:
                print(f"[ERROR] i={i}: {e}")
                continue
            rec = {
                "id": ex["id"],
                "prompt": prompt,
                "pred": res["text"],
                "e2e_ms": res["e2e_ms"],
                "toxicity_ref": ex.get("toxicity_ref")
            }
            fout.write(json.dumps(rec) + "\n")
            if i % 1000 == 0:
                print(f"  Completed {i}/{total}")
    print("Saved output to", output_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="optional output path")
    p.add_argument("--temp", type=float, default=0.0)
    args = p.parse_args()
    model_tag = args.model.replace(":", "_").replace("/", "_")
    dataset_name = Path(args.input).stem
    out_path = args.output or f"pred_rtp_{dataset_name}_{model_tag}.jsonl"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    run_rtp(args.model, args.input, out_path, temperature=args.temp)

if __name__ == "__main__":
    main()