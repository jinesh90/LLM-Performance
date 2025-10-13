#!/usr/bin/env python3
import asyncio, httpx, time, json, csv, argparse, random, statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

def pct(vs, p):
    if not vs: return None
    vs = sorted(vs)
    k = (len(vs)-1) * p/100
    f = int(k); c = min(f+1, len(vs)-1)
    if f == c: return vs[f]
    return vs[f] + (vs[c]-vs[f])*(k-f)

def ns_to_ms(ns): return ns/1e6
def ns_to_s(ns):  return ns/1e9

async def stream_once(client: httpx.AsyncClient, host: str, model: str, prompt: str, no_think: bool=True, timeout=600) -> Dict[str, Any]:
    url = host.rstrip('/') + "/api/generate"
    payload = {
        "model": model,
        "prompt": f"Reasoning: low\n\n{prompt}",
        "stream": True,
        "think": (False if no_think else True),
        "options": {"temperature": 0, "top_p": 0.9, "num_predict": 256}
    }
    start = time.time()
    first_t = None
    prev_chunk_t = None
    itl_ms: List[float] = []
    final = None
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                now = time.time()
                try:
                    d = json.loads(line)
                except Exception:
                    # ignore non-JSON keepalives if any
                    continue
                if d.get("response") and first_t is None:
                    first_t = now
                if prev_chunk_t is not None:
                    itl_ms.append((now - prev_chunk_t)*1000.0)
                prev_chunk_t = now
                if d.get("done"):
                    final = d
                    break
    except Exception as e:
        return {"ok": False, "error": str(e), "model": model}

    end = time.time()
    ttft_ms = (first_t - start)*1000.0 if first_t else None
    e2e_ms  = (end - start)*1000.0
    # From Ollama final payload
    eval_count = (final or {}).get("eval_count", 0)
    eval_duration_ns = (final or {}).get("eval_duration", 0)  # ns
    prompt_eval_count = (final or {}).get("prompt_eval_count", 0)
    prompt_eval_duration_ns = (final or {}).get("prompt_eval_duration", 0)

    tps = (eval_count / ns_to_s(eval_duration_ns)) if eval_duration_ns else None
    tpot_ms = (1000.0 / tps) if tps and tps > 0 else None
    prefill_ms = ns_to_ms(prompt_eval_duration_ns) if prompt_eval_duration_ns else None

    return {
        "ok": True,
        "model": model,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "prompt_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "tps": tps,
        "tpot_ms": tpot_ms,
        "prefill_ms": prefill_ms,
        "itl_mean_ms": (statistics.mean(itl_ms) if itl_ms else None),
        "itl_p95_ms": (pct(itl_ms, 95) if itl_ms else None),
        "itl_count": len(itl_ms)
    }

async def worker(name: int, prompts: List[Dict[str,str]], host: str, model: str, results: List[Dict[str,Any]], stop_at: float, semaphore: asyncio.Semaphore, no_think: bool):
    async with httpx.AsyncClient() as client:
        while time.time() < stop_at:
            rec = random.choice(prompts)
            prompt = rec["prompt"]
            async with semaphore:
                res = await stream_once(client, host, model, prompt, no_think=no_think)
            res["label"] = rec.get("label","")
            res["ts"] = time.time()
            results.append(res)

async def run_test(host: str, model: str, prompts: List[Dict[str,str]], duration_s: int, concurrency: int, no_think: bool):
    results: List[Dict[str,Any]] = []
    sem = asyncio.Semaphore(concurrency)
    test_start = time.time()
    stop_at = test_start + duration_s
    tasks = [asyncio.create_task(worker(i, prompts, host, model, results, stop_at, sem, no_think)) for i in range(concurrency)]
    await asyncio.gather(*tasks)
    test_end = time.time()
    wall = test_end - test_start
    return results, wall

def aggregate(rows: List[Dict[str,Any]], wall_time_s: float):
    ok = [r for r in rows if r.get("ok")]
    errs = [r for r in rows if not r.get("ok")]
    def take(key):
        xs = [r[key] for r in ok if r.get(key) is not None]
        return {
            "mean": (statistics.mean(xs) if xs else None),
            "p50": (pct(xs, 50) if xs else None),
            "p95": (pct(xs, 95) if xs else None),
            "p99": (pct(xs, 99) if xs else None),
            "count": len(xs)
        }
    # System-level
    total_tokens = sum((r.get("output_tokens") or 0) for r in ok)
    system_tps = total_tokens / wall_time_s if wall_time_s > 0 else None
    rps = (len(ok) / wall_time_s) if wall_time_s > 0 else None

    return {
        "n_requests": len(rows),
        "n_ok": len(ok),
        "n_errors": len(errs),
        "rps": rps,
        "system_tps": system_tps,
        "ttft_ms": take("ttft_ms"),
        "e2e_ms": take("e2e_ms"),
        "tps": take("tps"),
        "tpot_ms": take("tpot_ms"),
        "prefill_ms": take("prefill_ms"),
        "itl_mean_ms": take("itl_mean_ms"),
        "itl_p95_ms": take("itl_p95_ms")
    }

def write_csv(path: Path, rows: List[Dict[str,Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = ["ts","model","label","ok","error","ttft_ms","e2e_ms","prompt_tokens","output_tokens","tps","tpot_ms","prefill_ms","itl_mean_ms","itl_p95_ms","itl_count"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

def print_summary(title: str, agg: Dict[str,Any]):
    def fmt(d):
        if d["mean"] is None: return "–"
        return f"mean={d['mean']:.1f}  p50={d['p50']:.1f}  p95={d['p95']:.1f}  p99={d['p99']:.1f}"
    print(f"\n=== {title} ===")
    print(f"Requests OK/Total: {agg['n_ok']}/{agg['n_requests']}  |  Errors: {agg['n_errors']}")
    print(f"RPS: {agg['rps']:.2f}   System TPS: {agg['system_tps']:.2f}" if agg['rps'] else "RPS: –   System TPS: –")
    for k in ["ttft_ms","e2e_ms","tps","tpot_ms","prefill_ms","itl_mean_ms","itl_p95_ms"]:
        print(f"{k}: {fmt(agg[k])}")

def load_prompts(path: Path):
    ps = []
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            ps.append(json.loads(line))
    return ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--duration", type=int, default=60, help="test length in seconds")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no_think", action="store_true", help="disable chain-of-thought/thinking")
    ap.add_argument("--out", default="csv/combined.csv")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    results, wall = asyncio.run(run_test(args.host, args.model, prompts, args.duration, args.concurrency, args.no_think))
    write_csv(Path(args.out), results)
    agg = aggregate(results, wall)
    print_summary(f"{args.model}  (dur={args.duration}s, conc={args.concurrency})", agg)

if __name__ == "__main__":
    main()
