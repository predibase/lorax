# 🚀 LoRAX Deployment Playbook

Welcome to the **LoRAX Deployment Playbook**! This guide is designed for **first-time operators** setting up a **LoRAX server** on a fresh **Ubuntu 22.04** GPU host with **sudo** access. We'll walk you through each step, explain *why* it matters, and provide quick fixes for common issues. Let's get your **LoRAX server** up and running! 🎉

> **Goal:** Deploy a working **LoRAX server** with a chosen model, understand the process, and troubleshoot issues fast.

---

## 📋 Overview

To deploy **LoRAX**, you need these components in order:

1. **GPU Driver** – Verify `nvidia-smi` works on the host.
2. **Docker Engine** – Ensure the user is in the `docker` group.
3. **NVIDIA Container Runtime** – Make GPUs accessible inside containers.
4. **LoRAX Container** – Pull or build the container image.
5. **Model Files** – Download or cache model files.
6. **API** – Confirm the server is listening and passes a basic inference test.

> **Quick Sanity Check:** Stop at the first failure in this sequence:
> - **A.** Run `nvidia-smi` on the host.
> - **B.** Test GPU access in a container: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`.
> - **C.** Launch **LoRAX** with `MODEL_ID=gpt2`.
> - **D.** Test the API with `curl`.
> - **E.** Scale up to a larger model.

---

## Phase 1: Host Setup

### 1. Verify NVIDIA Driver ✅

Ensure your **NVIDIA driver** is working correctly.

```bash
nvidia-smi
```

**Success:** Displays a table with the driver version and GPU details.  
**Common Failures:**
- *`command not found`* → Driver not installed or PATH issue.
- *“NVIDIA-SMI has failed”* → Kernel module mismatch or Secure Boot blocking.

> **Fix:** Reinstall the NVIDIA driver or disable/enroll MOK for Secure Boot.

---

### 2. Install Docker 🐳

Set up **Docker** to run containers on **Ubuntu 22.04**.

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**What This Does:**
- Updates package metadata.
- Installs tools for HTTPS repositories.
- Sets up Docker’s GPG key and repository.
- Installs **Docker Engine**, CLI, and plugins.

**Success:** Run `docker --version` and `systemctl status docker` (should show *active (running)*).  
**Common Failures:**
- GPG/repo errors (“NO_PUBKEY”, “Unsigned”) → Key issue; redo key setup.
- Architecture mismatch on non-x86 hosts.

> **Fix:** Re-run key download steps and `apt update`.

---

### 3. Install NVIDIA Container Toolkit 🔧

Enable GPU access inside **Docker containers**.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**What This Does:**
- Adds the NVIDIA Container Toolkit repository.
- Installs the toolkit and configures Docker to use NVIDIA GPUs.

**Success:** Check `/etc/docker/daemon.json` for `runtimes.nvidia`. Test with a CUDA container (Step 5).  
**Common Failures:**
- `nvidia-ctk: command not found` → Installation failed; redo apt steps.
- “Could not select device driver” → Runtime misconfigured; re-run configure and restart.

> **Fix:** Re-run the toolkit installation and configuration steps.

---

### 4. Add User to Docker Group 👤

Allow running **Docker** commands without `sudo`.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Success:** `groups` shows `docker`; `docker ps` works without `sudo`.  
**Common Failure:** Commands still require `sudo` → Log out and back in.

> **Tip:** Log out and log back in to apply group changes.

---

### 5. Verify GPU in Container 🖥️

Confirm GPUs are accessible inside a **Docker container**.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Success:** Displays a table similar to `nvidia-smi` on the host.  
**Common Failures:**
- “Unknown runtime specified nvidia” → Toolkit setup incomplete (redo Step 3).
- “CUDA driver version insufficient” → Host driver outdated; update it.
- “Could not select device driver” → Runtime misconfigured; redo Step 3.

> **Fix:** Revisit NVIDIA Container Toolkit setup or update the host driver.

---

## Phase 2: Deploy LoRAX

Choose one deployment path:
- **(A) Pre-built Image** – Fastest option, recommended for most users.
- **(B) Build from Source** – Only for custom changes or unreleased patches.

### Option A: Pre-built Image 🎉

#### 1. Pull the LoRAX Image

```bash
docker pull ghcr.io/predibase/lorax:main
```


**Success:** Image downloads successfully.  
**Common Failure:** Network timeout → Retry or check connectivity.

> **Tip:** This is a public image, so no authentication issues are expected.

---

#### 2. Choose Your Model 📊

Start with **`gpt2`** for a quick test (it’s small and fast). Larger models require careful **VRAM** planning to avoid `CUDA out of memory` errors.

| **Model** | **Params** | **VRAM (FP16/BF16)** | **Notes** |
|-----------|------------|-----------------------|-----------|
| `gpt2` | 0.1B | ~0.5 GB | Perfect for testing; fits any GPU. |
| `bigcode/starcoder2-3b` | 3B | ~6–7 GB | Works on 8 GB VRAM GPUs. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~14–15 GB | Needs 16–24 GB VRAM. |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | Tight on 16 GB; better with 24 GB. |
| `TheBloke/Mistral-7B-Instruct-v0.3-GPTQ` | 7B (4-bit) | ~8–10 GB | Quantized; fits 12–16 GB VRAM. |
| `meta-llama/Meta-Llama-3-13B-Instruct` | 13B | ~26 GB | Requires 24–26 GB VRAM. |
| `meta-llama/Meta-Llama-3-70B-Instruct` | 70B | 135–140 GB | Needs multi-GPU or heavy quantization. |

> **VRAM Tips:**
> - Keep **10–15% VRAM free** for KV cache and overhead.
> - **6–8 GB GPUs**: Stick to `gpt2` or quantized 7B models.
> - **12–16 GB GPUs**: Comfortable for 7B; tight for 8B.
> - **24 GB+ GPUs**: Suitable for 13B or multi-instance setups.

---

#### 3. Run the LoRAX Container

```bash
MODEL_ID="gpt2"
SHARDED_MODEL="false"
PORT=80

docker run --rm 

--name lorax 

--gpus all 

-e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" 

-e TRANSFORMERS_CACHE=/data 

-v "$HOME/lorax_model_cache":/data 

-v "$HOME/lorax_outlines_cache":/root/.cache/outlines 

--user "$(id -u):$(id -g)" 

-p ${PORT}:80 

ghcr.io/predibase/lorax:main 

--model-id "$MODEL_ID" 

--sharded "$SHARDED_MODEL"
```


**What This Does:**
- Starts the **LoRAX container** with GPU access.
- Mounts model cache to persist downloads.
- Maps port **80** (container) to your chosen **host port**.
- Loads the specified **model** (start with `gpt2`).

**Success:** Logs show model download/cache hit and “Model loaded”; health endpoint responds.  
**Common Failures:**
- Stalled download → Network or Hugging Face rate limits.
- `CUDA out of memory` → Model too large for GPU VRAM.

> **Fix for Stalled Downloads:**
> 1. Visit the model’s Hugging Face page (e.g., `https://huggingface.co/<model_id>/tree/main`).
> 2. Note the commit hash from the URL or “Files and Versions.”
> 3. Create the cache path: `$HOME/lorax_model_cache/<model_id>/snapshots/<commit_hash>/`.
> 4. Download all model files (config, tokenizer, `.safetensors`, etc.) to that directory.
> 5. Re-run the container; it should use the cached files.

---

### Option B: Build from Source 🛠️

Use this if you need custom changes or unreleased patches.

#### 1. Clone the Repository

```bash
git clone https://github.com/predibase/lorax.git
cd lorax
```

#### 2. Initialize Submodules (if needed)

```bash
git submodule update --init --recursive
```


#### 3. Build the Image

```bash
docker build -t my-lorax-server -f Dockerfile .
```


**Common Failures:**
- Build stalls → Add `--network=host` to the build command.
- Version conflicts → Adjust base image or dependencies.

#### 4. Run the Container

Use the same `docker run` command as in Option A, replacing `ghcr.io/predibase/lorax:main` with `my-lorax-server`.

**Common Failures:**
- “Exec format error” → Image built for wrong architecture.
- Immediate exit → Library mismatch; rebuild with compatible CUDA base.

---

## Phase 3: Test the API

Once logs show the server is ready, test the **LoRAX API**.

```bash
curl http://localhost:80/
```


**Example Inference:**

```bash
curl -X POST http://localhost:80/generate 

-H 'Content-Type: application/json' 

-d '{"prompt":"Hello","max_tokens":32}'
```


**Success:** Returns JSON with generated text.  
**Common Failures:**
- Connection refused → Container not running or wrong port (`docker ps`).
- 404 → Wrong endpoint; check root docs.
- 500 → Model not loaded or OOM (`docker logs lorax`).

> **Fix:** Check logs with `docker logs lorax` and verify port mapping.

---

## Phase 4: Performance & Scaling Tips

- **Concurrency:** Increase only after single-request stability (KV cache can cause OOM).
- **Tuning Options:** Adjust `--max-concurrent-requests`, batching, or tensor parallelization (if supported).
- **Monitor GPUs:**

```bash
watch -n1 nvidia-smi
```


---

## Troubleshooting Guide

**Format:** [Stage] Symptom → Cause → Fix

- **[Host]** `nvidia-smi` fails → Driver issue → Check `dmesg | grep -i nvidia | tail -n5`; reinstall driver or fix Secure Boot.
- **[Container]** “Could not select device driver” → Runtime misconfigured → Verify `/etc/docker/daemon.json`; redo toolkit setup.
- **[Docker]** Cache permission denied → Root-owned files → Run `sudo chown -R $(id -u):$(id -g) $HOME/lorax_model_cache`.
- **[Model Load]** CUDA OOM → Model too large → Check `nvidia-smi`; use smaller/quantized model.
- **[Model Load]** Download stalls → Network issue → Use manual download workaround.
- **[Model Load]** `RuntimeError: weight not found` → Quantized model incompatibility → Try FP16 or a different quantized model.
- **[API]** 404 on generate → Wrong route → Check `curl http://localhost:80/`; adjust client.
- **[API]** 500 error → OOM or bad params → Check `docker logs --tail 100 lorax | grep -i error`; reduce `max_tokens`.
- **[Performance]** Slow first call → Warmup overhead → Send a short warmup prompt.
- **[Performance]** Low GPU usage (<30%) → Small batches → Enable batching or increase concurrency.
- **[Stability]** Exit code 137 → Host OOM → Check `dmesg | tail`; reduce model size.

---

## 🧠 Decision Matrix

| **Situation** | **Action** |
|---------------|------------|
| `nvidia-smi` broken | Fix driver first. |
| Container `nvidia-smi` fails | Fix NVIDIA runtime config. |
| `gpt2` fails to load | Check environment/image. |
| `gpt2` works, larger model fails | Address VRAM/quantization issues. |
| API fails | Check routes, params, or logs. |
| API slow | Optimize concurrency or use smaller model. |

---

## 🧹 Cleanup & Reset

```bash
docker stop lorax
docker system prune -f
rm -rf $HOME/lorax_model_cache/*
sudo chown -R $(id -u):$(id -g) $HOME/lorax_model_cache
```


---

## 📜 Quick Command Recap

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
docker pull ghcr.io/predibase/lorax:main
MODEL_ID="gpt2"; docker run --rm --gpus all -v "$HOME/lorax_model_cache":/data -p 80:80 ghcr.io/predibase/lorax:main --model-id "$MODEL_ID" --sharded false
curl http://localhost:80/
```


---

## 🌟 Next Steps

- **Monitoring:** Add logging/metrics with Prometheus or parse stdout.
- **Security:** Set up a reverse proxy (nginx/traefik) with TLS for public access.
- **Automation:** Create health/warmup scripts (e.g., systemd or Docker Compose).
- **Reliability:** Add watchdog with `Restart=on-failure` (systemd or Docker policies).

---

**Happy Deploying!** 🎉

