# LLM-Performance
Compare Performance on Various Open Source LLM base models

# OSS-20B and other open source base models / Open WebUI  on AWS g5.12xlarge

This guide describes how to set up a complete local-LLM environment on an **AWS g5.12xlarge (4×A10G)** instance using **Ollama**, **Open WebUI**, and **MCPO** (Model Context Protocol OpenAPI Proxy).

---

## Prerequisites

### EC2 Configuration
| Setting | Recommended Value |
|----------|------------------|
| **Instance Type** | `g5.12xlarge` (4×A10G GPUs) |
| **AMI** | Ubuntu 22.04 LTS (x86_64) |
| **Storage** | 300 – 500 GB gp3 EBS |
| **Security Groups** | 22 (SSH), 11434 (Ollama), 3000 (Open WebUI), 3333 (MCPO) |
| **Key Pair** | SSH access key |

---

## Step 0 – Base System Prep

```bash
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get -y install build-essential wget curl git htop tmux unzip jq pkg-config
sudo hostnamectl set-hostname g5-ollama
```

## Step 1 (OPTIONAL: In case you have selected AMI that has already installed NVIDIA, DEEP LEARNING PACAKGE)– NVIDIA Drivers & CUDA Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get -y install nvidia-driver-535
sudo reboot
```

After reboot:

```nvidia-smi   # should list 4x A10G GPUs```

## Step 2 – Install Docker & NVIDIA Container Toolkit
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

sudo apt-get -y install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Step 3 – Install Ollama (GPU-enabled)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Allow access
```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama -p Environment
sudo ss -lntp | grep 11434
curl -s http://0.0.0.0:11434/v1/models
```

Manual launch alternative
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OLLAMA_HOST=0.0.0.0 \
OLLAMA_ORIGINS='*' \
ollama serve
```

Install Other Base Model for Performance than OSS 20B
```bash
ollama pull gpt-oss:20b
ollama pull qwen2.5:14b
ollama pull gemma2:27b
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull yi:34b   # may need a quantized variant to fit 24GB VRAM
```

Check models 
```bash
ollama list # this will display all models
```

## Step 5 – Run Open WebUI (Docker)
```bash
docker run -d --name open-webui -p 3000:8080 --gpus=all \
  --add-host=host.docker.internal:host-gateway \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

Visit http://<EC2_PUBLIC_IP>:3000 # make sure security groups allows access of port 3000

Connect Model:

Base URL: http://<EC2_PUBLIC_IP>:11434

API Path: /v1


## Step 6 - Validate
```bash
sudo ss -lntp | grep 11434

curl -s http://127.0.0.1:11434/v1/models | jq
curl -s -X POST http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-oss:20b","messages":[{"role":"user","content":"Hello!"}]}'
```


# What to Measure for LLM Performance

## ***TTFT***

What is TTFT?
```
Time to First Token
TTFT ≈ (queueing) 
     + (request parse + auth + rate-limit) 
     + (prompt preprocess: tokenization, truncation, routing) 
     + (KV-cache allocation/fill or retrieval from cache) 
     + (model graph warmup / compile / load-once costs if cold) 
     + (first forward pass latency up to token #1)
```

What is “snappiness”?

“Snappiness” is the perceived responsiveness of a chat/app. It’s mostly determined by:

TTFT (how quickly the first token appears), and Inter-token latency (ITL)/Time-per-output-token (TPOT) once streaming starts (how smoothly tokens arrive), plus
UI behavior (does text stream immediately, are partials flushed frequently), and Tail behavior (does it stall near the end due to long logits sampling or safety/rerank passes).

You can think of snappiness as:
```
snappiness ∝ 1 / (TTFT + jitter_in_ITL + UI_flush_delay + tail_stall)
```

Example timeline (interactive request)
```
t0        send()
t0+20ms   request parsed, auth checked
t0+40ms   tokenization complete
t0+90ms   scheduler admits to GPU; KV allocated
t0+140ms  first forward done; sampler picks token
t0+160ms  first token chunk flushed  <-- TTFT
t0+175ms  token #2
t0+190ms  token #3
...
t0+1200ms last token, stream closes  <-- E2E latency
```

Example Reporting template 

Prompt len: 512 tokens, Max output: 128, Sampling: greedy
Concurrency: 8 users, continuous batching on
TTFT: p50=220 ms, p95=380 ms
TPOT: p50=22 ms, p95=45 ms
E2E: p50=2.9 s
Throughput: 120 tok/s (system), 45 tok/s (per user)
Notes: prompt-cache hit-rate 72%, speculative decoding ON


## **ITL / TPOT (inter-token latency / time-per-output-token)**
Average time between consecutive output tokens once streaming has started. It measures how smoothly tokens arrive after the first token.

How to compute (client-side)?

Record the timestamp for each token chunk you receive after the first token.

Compute the deltas between adjacent tokens: Δᵢ = tᵢ − tᵢ₋₁.

ITL/TPOT = mean(Δᵢ). Also track p50/p95 to capture jitter.

Relation to other metrics:

TTFT = time until the first token.

ITL/TPOT = cadence after streaming begins.

E2E latency ≈ TTFT + (ITL × #tokens_out) + tail work.


## **E2E (END to END Latency)** 

E2E latency is the total wall-clock time from when your client sends a request until the full response is received (stream closed or final chunk delivered).

Streaming vs non-streaming

Streaming E2E: from send() → first byte (TTFB) → first token (TTFT) → tokens arrive → last token → stream closes.

Non-streaming E2E: from send() → single response body received.

```
E2E ≈ TTFT + (ITL × N_out) + tail_overhead
```

## **SYSTEM TOKEN PER SECOND**

the aggregate token generation rate of your whole serving stack (one host or a fleet).

```
system_tps = (total output tokens emitted by all requests) / (measurement window in seconds)
```
Why it’s useful

Capacity planning: “How many tokens/sec can the cluster sustain at target latency?”

**Cost/efficiency: tokens/sec per GPU or per dollar.**

SLO alarms: sudden drops signal contention, paging, or bad deployments.

How it differs from user TPS

User TPS (per-stream TPS) ≈ 1 / ITL (time-per-output-token) that a single user experiences.

system_tps sums across all concurrent streams. It rises with concurrency and batching—up to the point latency/SLOs suffer.

Quick relationships

In steady state:

system_tps ≈ RPS × E[output_tokens_per_request]

user_tps ≈ 1 / ITL

Upper bound intuition (decode phase):

system_tps ≤ (Σ GPUs) × (decode_tps_per_GPU) × (utilization) × (batch_efficiency)


## RPS ##
RPS = Requests Per Second — the rate at which your system completes requests.
