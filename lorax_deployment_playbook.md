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

Before diving into installations, let's quickly check if your system already has the necessary components. Run the `Check` command for each step. If it passes, you can **skip** the corresponding installation section. If it fails, expand the "Installation Guide" to proceed.

### 1. Check NVIDIA Driver ✅

Ensure your **NVIDIA driver** is working correctly.

```bash
nvidia-smi
```

**Success:** Displays a table with the driver version and GPU details.  
**Common Failures:**
- *`command not found`* → Driver not installed or PATH issue.
- *"NVIDIA-SMI has failed"* → Kernel module mismatch or Secure Boot blocking.

<details>
<summary>Click to expand: NVIDIA Driver Installation Guide</summary>

Installing NVIDIA drivers can be complex and varies greatly by OS and GPU. **We strongly recommend following the official NVIDIA documentation for your specific GPU and Linux distribution.** Example: [NVIDIA Drivers Downloads](https://www.nvidia.com/Download/index.aspx).

</details>

---

### 2. Check Docker Engine Installation 🐳

Run this command to check if Docker is installed and running:

```bash
# Check if we're inside a containerized environment where Docker can't run
if grep -qa 'docker\|lxc' /proc/1/cgroup || [ -f /.dockerenv ]; then
    echo "⚠️  Detected: This environment is containerized (Docker/LXC)."
    echo "You CANNOT start Docker inside a container on most cloud GPU providers."
    echo "👉  If you need full Docker access, deploy on a bare-metal or privileged VM."
    echo "The script will exit in 10 seconds. Hit Ctrl+C to abort immediately."
    sleep 10
    echo "Exiting script."
    exit 0
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "Docker Engine: Installed and running. ✅"
else
    echo "Docker Engine: NOT detected or NOT running. ❌"
fi
```

> **Outcome:** If you see "Docker is installed and running.", you can skip the installation below.

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

> **Outcome:** If you see GPU details (similar to `nvidia-smi` on host), you can skip the installation below.
> **Common Failures:** "Unknown runtime specified nvidia" or "Could not select device driver" means the Toolkit is not correctly installed or configured.

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

> **Outcome:** If you see "User is in the docker group.", you can skip the steps below.

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

> **Outcome:** If you see "HUGGING_FACE_HUB_TOKEN is set. ✅", you can skip the manual setup steps below.

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

Choose one deployment path:
- **(A) Pre-built Image** – Fastest option, recommended for most users.
- **(B) Build from Source** – Only for custom changes or unreleased patches.

---

### Common Failures during Container Launch

<details>
<summary>Click to expand: Common Failures during Container Launch</summary>

These issues can occur when attempting to run *any* LoRAX Docker container, regardless of whether it's pre-built or from source.

* **`docker: Error response from daemon: Conflict. The container name "/lorax" is already in use...`**: This means a container named `lorax` is already running or exists from a previous session. You need to stop and remove it first.
    ```bash
    docker stop lorax # Stop the running container
    docker rm lorax   # Remove the stopped container (optional, if --rm was not used or failed previously)
    ```
    Then, re-run your `docker run` command.
* **`docker: invalid reference format`, `--gpus: command not found`, etc.**: You likely copied the `docker run` command incorrectly. Ensure there are **no spaces** after the backslash `\` at the end of each line, and copy the entire block at once.
* **`CUDA out of memory`** → The model you are trying to load is too large for your GPU's VRAM. Refer to the [GPU VRAM vs. Model Size Compatibility](#option-b-build-from-source-🛠️) table and choose a smaller or more quantized model.
* **Stalled model download** → Indicates a network issue or Hugging Face rate limit when downloading the model weights inside the container.
    > **Fix for Stalled Downloads:**
    > 1.  Visit the model’s Hugging Face page (e.g., `https://huggingface.co/<model_id>/tree/main`).
    > 2.  Note the commit hash from the URL or “Files and Versions.”
    > 3.  Create the cache path on your host: `$HOME/lorax_model_cache/<model_id>/snapshots/<commit_hash>/`.
    > 4.  Download all model files (config, tokenizer, `.safetensors`, etc.) to that directory.
    > 5.  Re-run the container; it should now use the cached files.
* **`RuntimeError: weight not found`** or **`TypeError`** → Model or quantization incompatibility with the pre-built image. For broader model compatibility, custom configurations, or support for a wider range of quantized models, please proceed with [Option B: Build from Source](#option-b-build-from-source-🛠️).

</details>

---

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

**Critical Compatibility Note:** Due to internal versioning and optimization, the `ghcr.io/predibase/lorax:main` pre-built Docker image is **only consistently compatible with `mistralai/Mistral-7B-Instruct-v0.1`** at this time. Attempts to load other models (including `gpt2`, `starcoder2-3b`, or any other quantized models) may result in `TypeError`, `RuntimeError: weight ... does not exist`, or other internal loading failures. For broader model compatibility, custom configurations, or support for a wider range of quantized models, please proceed with **Option B: Build from Source**.

For `mistralai/Mistral-7B-Instruct-v0.1`, a GPU with **16-24 GB VRAM is recommended** to ensure smooth operation and sufficient KV cache.

---

#### 3. Run the LoRAX Container

```bash
# Define your variables (MODEL_ID is set to the only supported model)
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
SHARDED_MODEL="false" # Set to 'true' for sharded (multi-GPU) models like 70B
PORT=80 # Host port to access the LoRAX server

docker run --rm \
  --name lorax \
  --gpus all \
  -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  -e TRANSFORMERS_CACHE=/data \
  -v "$HOME/lorax_model_cache":/data \
  -v "$HOME/lorax_outlines_cache":/root/.cache/outlines \
  --user "$(id -u):$(id -g)" \
  -p ${PORT}:80 \
  ghcr.io/predibase/lorax:main \
  --model-id "$MODEL_ID" \
  --sharded "$SHARDED_MODEL"
```

<details>
<summary>Click to expand: Explanation of Docker Run Flags</summary>

**What This Does:**
- Starts the **LoRAX container** named `lorax` with GPU access.
- Mounts model cache to persist downloads between container restarts.
- Maps port **80** (container) to your chosen **host port**.
- Loads the specified **model** (now only `mistralai/Mistral-7B-Instruct-v0.1`).
- Uses your Hugging Face token for authenticated model downloads.

</details>

**Success:** Logs show model download/cache hit and “Model loaded”; health endpoint responds.  
**Common Failures:** Refer to [Common Failures during Container Launch](#common-failures-during-container-launch)

> **Fix for Stalled Downloads:**
> 1. Visit the model’s Hugging Face page (e.g., `https://huggingface.co/<model_id>/tree/main`).
> 2. Note the commit hash from the URL or “Files and Versions.”
> 3. Create the cache path: `$HOME/lorax_model_cache/<model_id>/snapshots/<commit_hash>/`.
> 4. Download all model files (config, tokenizer, `.safetensors`, etc.) to that directory.
> 5. Re-run the container; it should use the cached files.

---

### Option B: Build from Source 🛠️

Use this if you need custom changes or unreleased patches, or if you want to run models other than `mistralai/Mistral-7B-Instruct-v0.1`.

#### GPU VRAM vs. Model Size Compatibility

When building from source, you gain the flexibility to choose a wider range of models. Use the following table as a guide for VRAM compatibility:

| **Model** | **Params** | **VRAM (FP16/BF16)** | **Notes** |
|-----------|------------|-----------------------|-----------|
| `gpt2` | 0.1B | ~0.5 GB | Perfect for testing; fits any GPU. |
| `bigcode/starcoder2-3b` | 3B | ~6–7 GB | Works on 8 GB VRAM GPUs. |
| `mistralai/Mistral-7B-Instruct-v0.1` | 7B | ~14–15 GB | Needs 16–24 GB VRAM. |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | ~16 GB | Tight on 16 GB; better with 24 GB. |
| `meta-llama/Meta-Llama-3-13B-Instruct` | 13B | ~26 GB | Requires 24–26 GB VRAM. |
| `meta-llama/Meta-Llama-3-70B-Instruct` | 70B | 135–140 GB | Needs multi-GPU or heavy quantization. |

> **VRAM Tips:**
> - Keep **10–15% VRAM free** for KV cache and overhead.
> - **6–8 GB GPUs**: Stick to quantized 7B models.
> - **12–16 GB GPUs**: Comfortable for 7B; tight for 8B.
> - **24 GB+ GPUs**: Suitable for 13B or multi-instance setups.

#### 1. Clone the LoRAX Repository (Including all necessary Submodules)

**Problem:** To build LoRAX from source, you need not only the main repository but also its nested external dependencies, which are managed as Git submodules (e.g., `flashinfer` for custom CUDA kernels). Skipping this can lead to "No such file or directory" errors during the build.

**Action:** First, clone the main repository, then immediately initialize and update all its submodules.

```bash
git clone -b feat/deployment-playbook-enhancements https://github.com/minhkhoango/lorax.git
cd lorax
git submodule update --init --recursive
```

#### 2. Build the Image

```bash
docker build -t my-lorax-server -f Dockerfile .
```


**Common Failures:**
- Build stalls → Add `--network=host` to the build command.
- Version conflicts → Adjust base image or dependencies.

> **Important Note on Build Parallelism (`MAX_JOBS`) & Memory:**
> Building custom CUDA kernels from source is a memory-intensive process. The `Dockerfile` is configured with `ENV MAX_JOBS=2` as a **very conservative default** for parallel compilation. This value aims to provide the highest stability and prevent Out-Of-Memory (OOM) crashes on a wide range of hardware, including instances with limited RAM relative to CPU cores.
>
> * **To Optimize for Faster Builds (Recommended):**
>     If you have significantly more RAM (e.g., 96GB or more) and want to speed up compilation, you can safely **increase `MAX_JOBS`**.
>     1.  **Open the `Dockerfile`** in your cloned `lorax` directory using your preferred text editor (e.g., `nano Dockerfile` or `code Dockerfile`).
>     2.  **Find the line:** `ENV MAX_JOBS=2` (it will be surrounded by comments explaining its purpose)
>     3.  **Change the value** to a higher number (e.g., `16`, `24`, or `32`). *Always monitor your RAM usage (`htop`) during the build to avoid crashes.*
>     4.  **Save the `Dockerfile`** and restart your build command (`docker build -t my-lorax-server -f Dockerfile .`).
>
> * **If your build still crashes with an OOM error:**
>     This indicates you have very limited RAM or other processes are consuming it. You **must reduce `MAX_JOBS` further**. Edit the `Dockerfile` as described above and change the value to `1`. Then, restart the build.

#### 4. Run the Container

Use the same `docker run` command as in Option A, replacing `ghcr.io/predibase/lorax:main` with `my-lorax-server`.

**Common Failures:**
- “Exec format error” → Image built for wrong architecture.
- Immediate exit → Library mismatch; rebuild with compatible CUDA base.

---

## Phase 3: Test the API

Once logs show the server is ready, test the **LoRAX API**.

**Example Inference:**

```bash
curl 127.0.0.1:80/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
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

**Common Failures:** Refer to [Common Failures during Container Launch](#common-failures-during-container-launch)

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
- **[Model Load]** `RuntimeError: weight not found` or **`TypeError`** → Model or quantization incompatibility with the pre-built image. For broader model compatibility, custom configurations, or support for a wider range of quantized models, proceed with Option B: Build from Source.
- **[API]** 404 on generate → Wrong route → Check `curl http://localhost:80/`; adjust client.
- **[API]** 500 error → OOM or bad params → Check `docker logs --tail 100 lorax | grep -i error`; reduce `max_tokens`.
- **[Performance]** Slow first call → Warmup overhead → Send a short warmup prompt.
- **[Performance]** Low GPU usage (<30%) → Small batches → Enable batching or increase concurrency.
- **[Stability]** Exit code 137 → Host OOM → Check `dmesg | tail`; reduce model size.

</details>

---

## 🧠 Decision Matrix

<details>
<summary>Click to expand: Quick Decision Matrix</summary>

| **Situation** | **Action** |
|---------------|------------|
| `nvidia-smi` broken | Fix driver first. |
| Container `nvidia-smi` fails | Fix NVIDIA runtime config. |
| `gpt2` fails to load | Check environment/image. If you need broader model compatibility, proceed with Option B: Build from Source. |
| `gpt2` works, larger model fails | Address VRAM/quantization issues or use Option B for more models. |
| API fails | Check routes, params, or logs. |
| API slow | Optimize concurrency or use smaller model. |

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

# Pull and run LoRAX
docker pull ghcr.io/predibase/lorax:main
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"; docker run --rm --name lorax --gpus all -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" -e TRANSFORMERS_CACHE=/data -v "$HOME/lorax_model_cache":/data -v "$HOME/lorax_outlines_cache":/root/.cache/outlines --user "$(id -u):$(id -g)" -p 80:80 ghcr.io/predibase/lorax:main --model-id "$MODEL_ID" --sharded false

# Test the API
curl http://localhost:80/
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

