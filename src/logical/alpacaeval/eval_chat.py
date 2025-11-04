#!/usr/bin/env python3
import json, time, requests, argparse, os
from pathlib import Path

API = "http://127.0.0.1:11434/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def chat(model, messages, temperature=0.0, max_tokens=512, seed=7):
    t0 = time.time()
    r = requests.post(API, headers=HEADERS, json={
        "model": model,
        "messages": [{"role": "system", "content": "Answer the user's question directly without any reasoning or thinking steps. Respond concisely with the final answer only."}] +messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False
    })
    t1 = time.time()
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
    out = r.json()
    return {"text": out["choices"][0]["message"]["content"], "e2e_ms": (t1 - t0) * 1000}

def run_file(model, input_path, output_path):
    total = sum(1 for _ in open(input_path))
    print(f"\n🔹 Running {model} on {input_path} ({total} prompts)...\n")
    with open(input_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin, 1):
            ex = json.loads(line)
            instr = ex.get("instruction") or ex.get("prompt")
            user_input = ex.get("input", "")
            prompt = instr if not user_input else f"{instr}\n{user_input}"
            messages = [{"role": "user", "content": prompt}]
            try:
                res = chat(model, messages)
            except Exception as e:
                print(f"[ERROR] {i}: {e}")
                continue
            rec = {"id": ex.get("id", i), "prompt": prompt, "pred": res["text"], "ref": ex.get("output"), "e2e_ms": res["e2e_ms"]}
            fout.write(json.dumps(rec) + "\n")
            if i % 10 == 0 or i == total:
                print(f"  ✅ {i}/{total}")
    print(f"✅ Output saved → {output_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="optional custom output path")
    args = p.parse_args()
    model_tag = args.model.replace(":", "_").replace("/", "_")
    dataset_name = Path(args.input).stem
    out = args.output or f"pred_{dataset_name}_{model_tag}.jsonl"
    os.makedirs(Path(out).parent, exist_ok=True)
    run_file(args.model, args.input, out)

if __name__ == "__main__":
    main()
