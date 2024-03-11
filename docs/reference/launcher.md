# LoRAX Launcher

```shell
LoRAX Launcher

Usage: lorax-launcher [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `mistralai/Mistral-7B-Instruct-v0.1`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers
          
          [env: MODEL_ID=]
          [default: mistralai/Mistral-7B-Instruct-v0.1]

      --adapter-id <ADAPTER_ID>
          The name of the adapter to load. Can be a MODEL_ID as listed on <https://hf.co/models> or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers. Should be compatible with the model specified in `model_id`
          
          [env: ADAPTER_ID=]
          [default: ]

      --source <SOURCE>
          The source of the model to load. Can be `hub` or `s3`. `hub` will load the model from the huggingface hub. `s3` will load the model from the predibase S3 bucket
          
          [env: SOURCE=]
          [default: hub]

      --adapter-source <ADAPTER_SOURCE>
          The source of the model to load. Can be `hub` or `s3` or `pbase` `hub` will load the model from the huggingface hub. `s3` will load the model from the predibase S3 bucket. `pbase` will load an s3 model but resolve the metadata from a predibase server
          
          [env: ADAPTER_SOURCE=]
          [default: hub]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`
          
          [env: REVISION=]

      --validation-workers <VALIDATION_WORKERS>
          The number of tokenizer workers used for payload validation and truncation inside the router
          
          [env: VALIDATION_WORKERS=]
          [default: 2]

      --sharded <SHARDED>
          Whether to shard the model across multiple GPUs By default LoRAX will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard`
          
          [env: SHARDED=]
          [possible values: true, false]

      --num-shard <NUM_SHARD>
          The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 lorax-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 lorax-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance
          
          [env: NUM_SHARD=]

      --quantize <QUANTIZE>
          Whether you want the model to be quantized. This will use `bitsandbytes` for quantization on the fly, or `gptq`
          
          [env: QUANTIZE=]
          [possible values: bitsandbytes, bitsandbytes-nf4, bitsandbytes-fp4, gptq, awq]

      --compile
          Whether you want to compile the model into a CUDA graph. This will speed up decoding but increase GPU memory usage
          
          [env: COMPILE=]

      --dtype <DTYPE>
          The dtype to be forced upon the model. This option cannot be used with `--quantize`
          
          [env: DTYPE=]
          [possible values: float16, bfloat16]

      --trust-remote-code
          Whether you want to execute hub modelling code. Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision
          
          [env: TRUST_REMOTE_CODE=]

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly
          
          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 128]

      --max-best-of <MAX_BEST_OF>
          This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time, and return the best in terms of overall log probability over the entire generated sequence
          
          [env: MAX_BEST_OF=]
          [default: 2]

      --max-stop-sequences <MAX_STOP_SEQUENCES>
          This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt the model in a specific way and define their "own" stop token aligned with their prompt
          
          [env: MAX_STOP_SEQUENCES=]
          [default: 4]

      --max-input-length <MAX_INPUT_LENGTH>
          This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value, the longer prompt users can send which can impact the overall memory required to handle the load. Please note that some models have a finite range of sequence they can handle
          
          [env: MAX_INPUT_LENGTH=]
          [default: 1792]

      --max-total-tokens <MAX_TOTAL_TOKENS>
          This is the most important value to set as it defines the "memory budget" of running clients requests. Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511` max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less effective batching can be
          
          [env: MAX_TOTAL_TOKENS=]
          [default: 2048]

      --waiting-served-ratio <WAITING_SERVED_RATIO>
          This represents the ratio of waiting queries vs running queries where you want to start considering pausing the running queries to include the waiting ones into the same batch. `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's only 10 queries left in the current batch we check if we can fit those 12 waiting queries into the batching strategy, and if yes, then batching happens delaying the 10 running queries by a `prefill` run.
          
          This setting is only applied if there is room in the batch as defined by `max_batch_total_tokens`.
          
          [env: WAITING_SERVED_RATIO=]
          [default: 1.2]

      --max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>
          Limits the number of tokens for the prefill operation. Since this operation take the most memory and is compute bound, it is interesting to limit the number of requests that can be sent
          
          [env: MAX_BATCH_PREFILL_TOKENS=]
          [default: 4096]

      --max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.
          
          This represents the total amount of potential tokens within a batch. When using padding (not recommended) this would be equivalent of `batch_size` * `max_total_tokens`.
          
          However in the non-padded (flash attention) version this can be much finer.
          
          For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.
          
          Overall this number should be the largest possible amount that fits the remaining memory (after the model is loaded). Since the actual memory overhead depends on other parameters like if you're using quantization, flash attention or the model implementation, LoRAX cannot infer this number automatically.
          
          [env: MAX_BATCH_TOTAL_TOKENS=]

      --max-waiting-tokens <MAX_WAITING_TOKENS>
          This setting defines how many tokens can be passed before forcing the waiting queries to be put on the batch (if the size of the batch allows for it). New queries require 1 `prefill` forward, which is different from `decode` and therefore you need to pause the running batch in order to run `prefill` to create the correct values for the waiting queries to be able to join the batch.
          
          With a value too small, queries will always "steal" the compute to run `prefill` and running queries will be delayed by a lot.
          
          With a value too big, waiting queries could wait for a very long time before being allowed a slot in the running batch. If your server is busy that means that requests that could run in ~2s on an empty server could end up running in ~20s because the query had to wait for 18s.
          
          This number is expressed in number of tokens to make it a bit more "model" agnostic, but what should really matter is the overall latency for end users.
          
          [env: MAX_WAITING_TOKENS=]
          [default: 20]

      --max-active-adapters <MAX_ACTIVE_ADAPTERS>
          Maximum number of adapters that can be placed on the GPU and accept requests at a time
          
          [env: MAX_ACTIVE_ADAPTERS=]
          [default: 1024]

      --adapter-cycle-time-s <ADAPTER_CYCLE_TIME_S>
          The time in seconds between adapter exchanges
          
          [env: ADAPTER_CYCLE_TIME_S=]
          [default: 2]

      --hostname <HOSTNAME>
          The IP address to listen on
          
          [env: HOSTNAME=]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on
          
          [env: PORT=]
          [default: 3000]

      --shard-uds-path <SHARD_UDS_PATH>
          The name of the socket for gRPC communication between the webserver and the shards
          
          [env: SHARD_UDS_PATH=]
          [default: /tmp/lorax-server]

      --master-addr <MASTER_ADDR>
          The address the master shard will listen on. (setting used by torch distributed)
          
          [env: MASTER_ADDR=]
          [default: localhost]

      --master-port <MASTER_PORT>
          The address the master port will listen on. (setting used by torch distributed)
          
          [env: MASTER_PORT=]
          [default: 29500]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance
          
          [env: HUGGINGFACE_HUB_CACHE=]

      --weights-cache-override <WEIGHTS_CACHE_OVERRIDE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance
          
          [env: WEIGHTS_CACHE_OVERRIDE=]

      --disable-custom-kernels
          For some models (like llama), LoRAX implemented custom cuda kernels to speed up inference. Those kernels were only tested on A100. Use this flag to disable them if you're running on different hardware and encounter issues
          
          [env: DISABLE_CUSTOM_KERNELS=]

      --cuda-memory-fraction <CUDA_MEMORY_FRACTION>
          Limit the CUDA available memory. The allowed value equals the total visible memory multiplied by cuda-memory-fraction
          
          [env: CUDA_MEMORY_FRACTION=]
          [default: 1.0]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)
          
          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          [env: OTLP_ENDPOINT=]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: CORS_ALLOW_ORIGIN=]

      --watermark-gamma <WATERMARK_GAMMA>
          [env: WATERMARK_GAMMA=]

      --watermark-delta <WATERMARK_DELTA>
          [env: WATERMARK_DELTA=]

      --ngrok
          Enable ngrok tunneling
          
          [env: NGROK=]

      --ngrok-authtoken <NGROK_AUTHTOKEN>
          ngrok authentication token
          
          [env: NGROK_AUTHTOKEN=]

      --ngrok-edge <NGROK_EDGE>
          ngrok edge
          
          [env: NGROK_EDGE=]

  -e, --env
          Display a lot of information about your runtime environment

      --download-only
          Download model weights only
          
          [env: DOWNLOAD_ONLY=]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
