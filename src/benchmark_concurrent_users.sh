#!/bin/bash

echo "Starting Concurrent Requests to LLMs"

for U in 1 2 4 8 16 32 64; do
  echo "gpt-oss:20b  conc=$U"
  python3 benchmark.py --no_think --model gpt-oss:20b --duration 120 --concurrency $U --out csv/gptoss20b_c${U}.csv
done


for U in 1 2 4 8 16 32 64; do
  echo "yi:34b  conc=$U"
  python3 benchmark.py --no_think --model yi:34b --duration 120 --concurrency $U --out csv/yi34b_c${U}.csv
done


for U in 1 2 4 8 16 32 64; do
  echo "llama3.1:8b  conc=$U"
  python3 benchmark.py --no_think --model llama3.1:8b --duration 120 --concurrency $U --out csv/llama318b_c${U}.csv
done

for U in 1 2 4 8 16 32 64; do
  echo "mistral:7b  conc=$U"
  python3 benchmark.py --no_think --model mistral:7b --duration 120 --concurrency $U --out csv/mistral7b_c${U}.csv
done


for U in 1 2 4 8 16 32 64; do
  echo "gemma2:27b   conc=$U"
  python3 benchmark.py --no_think --model gemma2:27b --duration 120 --concurrency $U --out csv/gemma2:27b_c${U}.csv
done


for U in 1 2 4 8 16 32 64; do
  echo "qwen2.5:14b   conc=$U"
  python3 benchmark.py --no_think --model qwen2.5:14b --duration 120 --concurrency $U --out csv/qwen2514b_c${U}.csv
done

echo "Completed Concurrent Requests to LLMs !"




