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
> - **C.** Launch **LoRAX** with `MODEL_ID=mistralai/Mistral-7B-Instruct-v0.1` (the pre-built image is recommended for this check).
> - **D.** Test the API with `curl`.
> - **E.** Scale up to a larger model.

---

## Phase 1: Host Setup

Before diving into installations, let's quickly check if your system already has the necessary components. Run the `Check` command for each step. If it passes, you can **skip** the corresponding installation section. If it fails, expand the "Installation Guide" to proceed.

### 1. Check NVIDIA Driver ✅

Ensure your **NVIDIA driver** is working correctly.

```bash
nvidia-smi
```
**Success:** Displays a table with the driver version and GPU details.
<details>
<summary>Click to expand: Common Failures & Troubleshooting</summary>

- *`command not found`* → Driver not installed or PATH issue.
- *"NVIDIA-SMI has failed"* → Kernel module mismatch or Secure Boot blocking.

</details>

<details>
<summary>Click to expand: NVIDIA Driver Installation Guide</summary>

Installing NVIDIA drivers can be complex and varies greatly by OS and GPU. **We strongly recommend following the official NVIDIA documentation for your specific GPU and Linux distribution.** Example: [NVIDIA Drivers Downloads](https://www.nvidia.com/Download/index.aspx).

</details>

---

### 2. Check Docker Engine Installation 🐳

Run this command to check if Docker is installed and running:

```bash
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "Docker Engine: Installed and running. ✅"
else
    echo "Docker Engine: NOT detected or NOT running. ❌"
fi
```
**Success:** `Docker Engine: Installed and running. ✅`
<details>
<summary>Click to expand: Common Failures & Troubleshooting</summary>

- `Docker Engine: NOT detected or NOT running. ❌`
- *GPG/repo errors ("NO_PUBKEY", "Unsigned")* → Key issue; redo key setup.
- *Architecture mismatch* on non-x86 hosts.

</details>

<details>
<summary>Click to expand: Install Docker Engine</summary>

Set up **Docker** to run containers on **Ubuntu 22.04**.

```bash
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get autoremove -y --purge
sudo rm -rf /var/lib/docker /var/lib/containerd

sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**What This Does:**
- Updates package metadata.
- Installs tools for HTTPS repositories.
- Sets up Docker's GPG key and repository.
- Installs **Docker Engine**, CLI, and plugins.

**Success:** Run `docker --version` and `systemctl status docker` (should show *active (running)*).  
**Common Failures:**
- GPG/repo errors ("NO_PUBKEY", "Unsigned") → Key issue; redo key setup.
- Architecture mismatch on non-x86 hosts.

> **Fix:** Re-run key download steps and `apt update`.

</details>

---

### 3. Check NVIDIA Container Toolkit 🔧

Run this command to verify GPU access within a container (requires Docker and Toolkit):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
**Success:** Displays GPU details (similar to `nvidia-smi` on host).
<details>
<summary>Click to expand: Common Failures & Troubleshooting</summary>

- *"Unknown runtime specified nvidia"* or *"Could not select device driver"* → Toolkit not correctly installed or configured.

</details>

<details>
<summary>Click to expand: Install NVIDIA Container Toolkit</summary>

Enable GPU access inside **Docker containers**.

```bash
# SHORT, FORCEFUL NVIDIA TOOLKIT INSTALL FOR UBUNTU 22.04 (Vast Mystery Box)
set -euo pipefail

# -- CRITICAL CHECKS --
[[ "$(lsb_release -rs)" = "22.04" ]] || echo "[WARNING] Not Ubuntu 22.04. You WILL break stuff." 
command -v docker >/dev/null || { echo "[FATAL] Docker not found."; exit 1; }

# -- FORCE OVERWRITE EXISTING GPG KEY --
sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# -- ADD REPO & KEY (no prompt) --
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list \
| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

# -- INSTALL --
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# -- CONFIGURE --
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# -- SANITY TEST --
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi \
|| echo "[FATAL] Docker can't see your GPU. Drivers likely broken. Try 'nvidia-smi' on host."
```

**What This Does:**
- Adds the NVIDIA Container Toolkit repository.
- Installs the toolkit and configures Docker to use NVIDIA GPUs.

**Success:** Check `/etc/docker/daemon.json` for `runtimes.nvidia`. Test with a CUDA container (Step 5).  
**Common Failures:**
- `nvidia-ctk: command not found` → Installation failed; redo apt steps.
- "Could not select device driver" → Runtime misconfigured; re-run configure and restart.

> **Fix:** Re-run the toolkit installation and configuration steps.

</details>
---

### 4. Check User in Docker Group 👤

Run this command to check if your user is already in the 'docker' group:

```bash
groups | grep -q docker && echo "User is in the docker group." || echo "User is NOT in the docker group. Permissions needed."
```
**Success:** `User is in the docker group.`
<details>
<summary>Click to expand: Common Failures & Troubleshooting</summary>

- `User is NOT in the docker group. Permissions needed.`
- *Commands still require `sudo`* → Log out and back in.

</details>

<details>
<summary>Click to expand: Add User to Docker Group</summary>

Allow running **Docker** commands without `sudo`.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Success:** `groups` shows `docker`; `docker ps` works without `sudo`.  
**Common Failure:** Commands still require `sudo` → Log out and back in.

> **Tip:** Log out and log back in to apply group changes.

</details>

---

### 5. Hugging Face Authentication 🔑

Some models on Hugging Face require authentication to download. This is especially true for "gated" models like Mistral, Llama, and other proprietary models. You'll need a **Hugging Face Hub Token** to access these models.

**What is a Hugging Face Hub Token?**
A personal access token that acts like a password for programmatic access to Hugging Face. It allows LoRAX to download models on your behalf.

Run this command to check if your `HUGGING_FACE_HUB_TOKEN` is already set as an environment variable:

```bash
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "HUGGING_FACE_HUB_TOKEN is set. ✅"
else
    echo "HUGGING_FACE_HUB_TOKEN is NOT set. ❌"
fi
```
**Success:** `HUGGING_FACE_HUB_TOKEN is set. ✅`
<details>
<summary>Click to expand: Common Failures & Troubleshooting</summary>

- `HUGGING_FACE_HUB_TOKEN is NOT set. ❌` → Token missing or not exported correctly.

</details>

<details>
<summary>Click to expand: Set up HUGGING_FACE_HUB_TOKEN</summary>

#### Get Your Hugging Face Token

1. **Visit the token page:** Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Generate a new token:**
   - Click "New token"
   - Give it a name (e.g., "LoRAX Deployment")
   - Select "Read" role (sufficient for downloading models)
   - Click "Generate token"
3. **Copy the token:** It will look like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
4. **Request model access:** For gated models, visit their Hugging Face page and click "Request access" (e.g., [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3))

#### Set the Environment Variable

Add the token to your shell configuration so it's available for Docker:

```bash
# Add this line to your ~/.bashrc or ~/.zshrc file
export HUGGING_FACE_HUB_TOKEN='hf_YOUR_TOKEN_HERE'

# Reload your shell configuration
source ~/.bashrc  # or source ~/.zshrc if using zsh

# Verify it's set
echo $HUGGING_FACE_HUB_TOKEN
```

> **Important:** Replace `hf_YOUR_TOKEN_HERE` with your actual token. The Docker container will pick up this environment variable when passed with the `-e` flag.

> **Note:** For public models like `gpt2`, you don't need a token, but having one set up allows you to easily switch to gated models later.

</details>

---

## Phase 2: Deploy LoRAX

You can deploy LoRAX using either the **pre-built image** or by **building from source**. Both methods now support the same set of models:
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `meta-llama/Meta-Llama-3-8B-Instruct`

Choose your deployment path:
- **(A) Pre-built Image** – Fastest option, recommended for most users.
- **(B) Build from Source** – For custom changes or unreleased patches.

### 1. (Option A) Pull the Pre-built Image

```bash
docker pull ghcr.io/predibase/lorax:main
```

### 1. (Option B) Build the Image from Source

Want to build LoRAX from source for custom changes or the latest patches? Follow these steps:

```bash
# 1. Clone the repository (if you haven't already)
git clone -b feat/deployment-playbook-enhancements https://github.com/minhkhoango/lorax.git
cd lorax
# 2. Initialize submodules
git submodule update --init --recursive
```

> **Tip: Speed Up Your Build!**
> 
> By default, the Dockerfile uses `MAX_JOBS=2` to avoid out-of-memory (OOM) errors on machines with limited RAM. If you have a lot of RAM (e.g., 64GB, 96GB, or more), you can **dramatically speed up the build** by increasing this value.
>
> **How to adjust build speed:**
> 1. Open your `Dockerfile` at the root of your cloned repository (`~/lorax/Dockerfile`) in your editor.
> 2. Locate the line:
>    ```Dockerfile
>    ENV MAX_JOBS=2
>    ```
>    (This line is typically found around line 90 in the `Dockerfile` within the `kernel-builder` stage, but verify its exact location).
> 3. Change `2` to a higher number (e.g., `16`, `24`, or `32`) if your system has enough RAM.
> 4. Save your `Dockerfile` and rebuild the image.
>
> *Not sure how much RAM you have? Run `htop` or `free -h` in your terminal. If you run out of memory during build, lower `MAX_JOBS` and try again!*

Now, build your Docker image:

```bash
export DOCKER_BUILDKIT=1
docker build -t my-lorax-server -f Dockerfile .
```

---

### 2. Choose Your Model & Run the Container

Refer to the table below to select a model that fits your hardware and requirements:

| **Model** | **Params** | **VRAM (FP16/BF16)** | **Notes** |
|-----------|------------|-----------------------|-----------|
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~7 GB | Good for 8GB+ GPUs |
| `mistralai/Mistral-7B-Instruct-v0.1` | 7B | ~14–15 GB | Needs 16–24 GB VRAM. |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | Tight on 16 GB; better with 24 GB. |

> **VRAM Tips:**
> - Keep **10–15% VRAM free** for KV cache and overhead.
> - **6–8 GB GPUs**: Stick to quantized or smaller models.
> - **12–16 GB GPUs**: Comfortable for 7B; tight for 8B.
> - **24 GB+ GPUs**: Suitable for 13B or multi-instance setups.

#### Run the Container

Set your desired model and image name (see below):

```bash
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1" # or meta-llama/Llama-3.2-3B-Instruct, meta-llama/Meta-Llama-3-8B-Instruct
SHARDED_MODEL="false" # Set to 'true' for sharded (multi-GPU) models like 70B
PORT=80 # Host port to access the LoRAX server

# For pre-built image:
IMAGE_NAME="ghcr.io/predibase/lorax:main"
# For source-built image:
# IMAGE_NAME="my-lorax-server"

docker run --rm \
  --name lorax \
  --gpus all \
  -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  -e TRANSFORMERS_CACHE=/data \
  -v "$HOME/lorax_model_cache":/data \
  -v "$HOME/lorax_outlines_cache":/root/.cache/outlines \
  --user "$(id -u):$(id -g)" \
  -p ${PORT}:80 \
  $IMAGE_NAME \
  --model-id "$MODEL_ID" \
  --sharded "$SHARDED_MODEL"
```

<details>
<summary>Click to expand: Explanation of Docker Run Flags</summary>

**What This Does:**
- `docker run --rm --name lorax`: Starts a new container, removes it on exit, and names it `lorax`.
- `--gpus all`: Grants the container access to all available GPUs.
- `-e HUGGING_FACE_HUB_TOKEN`: Passes your Hugging Face authentication token.
- `-v "$HOME/lorax_model_cache":/data`: Mounts a local directory for persistent model caching.
- `-v "$HOME/lorax_outlines_cache":/root/.cache/outlines`: Mounts cache for Outlines library.
- `--user "$(id -u):$(id -g)"`: Runs the container process as your host user for permission consistency.
- `-p ${PORT}:80`: Maps the container's internal port 80 to your specified host port.
- `$IMAGE_NAME`: Specifies the Docker image to use (pre-built or source-built).
- `--model-id "$MODEL_ID"`: Sets the Hugging Face model to load.
- `--sharded "$SHARDED_MODEL"`: Configures for multi-GPU sharding if set to `true`.

</details>

---

## Phase 3: Test the API

Once logs show the server is ready, test the **LoRAX API**.

**Example Inference:**

```bash
curl 127.0.0.1:80/generate \
    -X POST \
    -d '{ "inputs": "[INST] What LLM model are you? [/INST]", "parameters": { "max_new_tokens": 64 } }' \
    -H 'Content-Type: application/json'
```  

If you're using a base model that supports LoRA adapters (like Mistral-7B) and have an adapter ID, you can test prompting a specific fine-tuned adapter.

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64,
            "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
        }
    }' \
    -H 'Content-Type: application/json'
```

Note: Replace vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k with an adapter_id that is compatible with 
your chosen base model.

**Success:** Logs show model download/cache hit and “Model loaded”; health endpoint responds.  
<details>
<summary>Click to expand: Common Failures during API Test</summary>

**Common Failures:** Refer to the Comprehensive Troubleshooting Guide below.

</details>


## Troubleshooting Guide

<details>
<summary>Click to expand: Comprehensive Troubleshooting Guide</summary>

**Format:** [Stage] Symptom → Cause → Fix

- **[Host]** `nvidia-smi` fails → Driver issue → Check `dmesg | grep -i nvidia | tail -n5`; reinstall driver or fix Secure Boot.
- **[Container]** “Could not select device driver” → Runtime misconfigured → Verify `/etc/docker/daemon.json`; redo toolkit setup.
- **[Docker]** Cache permission denied → Root-owned files → Run `sudo chown -R $(id -u):$(id -g) $HOME/lorax_model_cache`.
- **[Model Load]** CUDA OOM → Model too large → Check `nvidia-smi`; use smaller/quantized model.
- **[Model Load]** Download stalls → Network issue → Use manual download workaround.
- **[Model Load]** `RuntimeError: weight not found` or **`TypeError`** → Model or quantization incompatibility with the pre-built image. For detailed fixes, see the "Troubleshooting Model Compatibility (Build from Source)" section above.
- **[Download]** `UserWarning: Not enough free disk space` or `No space left on device` (during model download/caching):** The mounted model cache directory has insufficient space. Check `df -h $HOME/lorax_model_cache`, then `rm -rf` unused model folders. Consider larger disk if needed.
- **[Performance]** Slow first call → Warmup overhead → Send a short warmup prompt.
- **[Performance]** Low GPU usage (<30%) → Small batches → Enable batching or increase concurrency.
- **[Stability]** Exit code 137 → Host OOM → Check `dmesg | tail`; reduce model size.

</details>

<!-- Inserted section: Model Compatibility Beyond Mistral-7B (Build from Source) troubleshooting bullets -->

<details>
<summary>Model Compatibility Beyond Mistral-7B (Build from Source)</summary>

**Common Issues & Solutions:**

* **`TypeError: TensorParallelColumnLinear.load_multi() got an unexpected keyword argument 'fan_in_fan_out'` (for `gpt2`):**
    * **Cause:** This error is specific to `gpt2`'s `Conv1D` layer architecture and an API mismatch with the `vLLM` integration in LoRAX's custom modeling.
    * **Fix:** Ensure your `vLLM` is pinned to a compatible version/commit in `server/Makefile-vllm` (e.g., `v0.7.3` or specific fixes like `9985d06add07a4cc691dc54a7e34f54205c04d40` if explicitly needed). Rebuild your Docker image. The `--model-impl transformers` flag, while a workaround in some TGI contexts, is not supported by `lorax-launcher`.

* **`ImportError: No module named 'msgspec'` (for `Qwen` models or others using newer `vLLM` features):**
    * **Cause:** The `vLLM` version integrated in your build may require the `msgspec` Python library, which is not a default dependency.
    * **Fix:** Add `msgspec` to your `server/requirements.txt` file and rebuild your Docker image with `--no-cache` to ensure the new dependency is installed.

* **`RuntimeError: weight transformer.wte.weight does not exist` (for `bigcode/starcoder2-3b`):**
    * **Cause:** This indicates a specific naming convention or structural mismatch for certain weight files within the `bigcode/starcoder2-3b` checkpoint that LoRAX's `FlashSantacoderModel` is trying to load.
    * **Fix:** This often requires deeper debugging of the model's weight structure or changes within `lorax_server/models/custom_modeling/flash_santacoder_modeling.py`. Consider this model a known edge case that may require specific code adjustments beyond standard dependency management.

</details>

---

## 🧹 Cleanup & Reset

<details>
<summary>Click to expand: Cleanup & Reset Your Environment</summary>

```bash
docker stop lorax
docker system prune -f
rm -rf $HOME/lorax_model_cache/*
sudo chown -R $(id -u):$(id -g) $HOME/lorax_model_cache
```

</details>

---

## 📜 Quick Command Recap

```bash
# Check GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Pull and run LoRAX (Pre-built Image)
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"; \
docker run --rm --name lorax --gpus all -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  -e TRANSFORMERS_CACHE=/data -v "$HOME/lorax_model_cache":/data \
  -v "$HOME/lorax_outlines_cache":/root/.cache/outlines \
  --user "$(id -u):$(id -g)" -p 80:80 \
  ghcr.io/predibase/lorax:main --model-id "$MODEL_ID" --sharded false

# Test the API
curl 127.0.0.1:80/generate \
    -X POST \
    -d '{ "inputs": "[INST] What LLM model are you? [/INST]", "parameters": { "max_new_tokens": 64 } }' \
    -H 'Content-Type: application/json'
```


---

## 🌟 Next Steps

<details>
<summary>Click to expand: Beyond Basic Deployment (Next Steps)</summary>

- **Monitoring:** Add logging/metrics with Prometheus or parse stdout.
- **Security:** Set up a reverse proxy (nginx/traefik) with TLS for public access.
- **Automation:** Create health/warmup scripts (e.g., systemd or Docker Compose).
- **Reliability:** Add watchdog with `Restart=on-failure` (systemd or Docker policies).

</details>

---

**Happy Deploying!** 🎉

