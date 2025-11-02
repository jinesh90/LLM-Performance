#!/usr/bin/env python3
import json, openai, argparse
from tqdm import tqdm

openai.api_base = "https://api.openai.com/v1"
# export OPENAI_API_KEY=your_key  (or adjust for Claude etc.)

def judge(prompt, ans_a, ans_b):
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
    completion = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
    )
    res = completion.choices[0].message.content.strip()
    if "A" in res and "B" in res: res="Tie"
    if res not in ["A","B","Tie"]: res="Tie"
    return res

def compare_files(file_a, file_b):
    a = [json.loads(x) for x in open(file_a)]
    b = [json.loads(x) for x in open(file_b)]
    assert len(a)==len(b)
    wins_a=wins_b=tie=0
    for ex_a, ex_b in tqdm(zip(a,b), total=len(a)):
        p = ex_a["prompt"]
        res = judge(p, ex_a["pred"], ex_b["pred"])
        if res=="A": wins_a+=1
        elif res=="B": wins_b+=1
        else: tie+=1
    total=wins_a+wins_b+tie
    winrate = (wins_a + 0.5*tie)/total
    print(f"\nModelA vs ModelB results ({len(a)} prompts):")
    print(f"A wins={wins_a}, B wins={wins_b}, ties={tie}")
    print(f"Win-rate for A = {winrate*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="pred file A")
    ap.add_argument("--b", required=True, help="pred file B")
    args=ap.parse_args()
    compare_files(args.a, args.b)