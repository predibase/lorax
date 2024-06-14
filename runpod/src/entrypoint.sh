#!/bin/bash

# if [[ -z "${HF_MODEL_ID}" ]]; then
#   echo "HF_MODEL_ID must be set"
#   exit 1
# fi
# export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
  export REVISION="${HF_MODEL_REVISION}"
fi

if [[ -n "${SM_NUM_GPUS}" ]]; then
  export NUM_SHARD="${SM_NUM_GPUS}"
fi

if [[ -n "${HF_MODEL_QUANTIZE}" ]]; then
  export QUANTIZE="${HF_MODEL_QUANTIZE}"
fi

if [[ -n "${HF_MODEL_TRUST_REMOTE_CODE}" ]]; then
  export TRUST_REMOTE_CODE="${HF_MODEL_TRUST_REMOTE_CODE}"
fi

if [[ -n "${HF_MAX_TOTAL_TOKENS}" ]]; then
  export MAX_TOTAL_TOKENS="${HF_MAX_TOTAL_TOKENS}"
fi

if [[ -n "${HF_MAX_INPUT_LENGTH}" ]]; then
  export MAX_INPUT_LENGTH="${HF_MAX_INPUT_LENGTH}"
fi

if [[ -n "${HF_MAX_BATCH_TOTAL_TOKENS}" ]]; then
  export MAX_BATCH_TOTAL_TOKENS="${HF_MAX_BATCH_TOTAL_TOKENS}"
fi

if [[ -n "${HF_MAX_BATCH_PREFILL_TOKENS}" ]]; then
  export MAX_BATCH_PREFILL_TOKENS="${HF_MAX_BATCH_PREFILL_TOKENS}"
fi

# Start the text generation server
nohup lorax-launcher --port 8080 --model-id      predibase/Meta-Llama-3-8B-Instruct-dequantized      --adapter-source      hub      --default-adapter-source      pbase      --max-batch-prefill-tokens      32768      --max-total-tokens      8192      --max-input-length      8191      --max-concurrent-requests      1024 &

# Start the handler using python 3.10
python3.10 -u /handler.py
