#!/bin/bash

python3 benchmark.py --no_think --model gpt-oss:20b --duration 60 --concurrency 1 --out csv/gptoss20b_c1.csv
python3 benchmark.py --no_think --model yi:34b --duration 60 --concurrency 1 --out csv/yi34b_c1.csv
python3 benchmark.py --no_think --model llama3.1:8b --duration 60 --concurrency 1 --out csv/llama318b_c1.csv
python3 benchmark.py --no_think --model mistral:7b --duration 60 --concurrency 1 --out csv/mistral7b_c1.csv
python3 benchmark.py --no_think --model gemma2:27b --duration 60 --concurrency 1 --out csv/gemma227b_c1.csv
python3 benchmark.py --no_think --model qwen2.5:14b --duration 60 --concurrency 1 --out csv/qwen2514b_c1.csv
