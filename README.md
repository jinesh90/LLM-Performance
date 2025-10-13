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

