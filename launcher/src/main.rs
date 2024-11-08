use clap::{Parser, ValueEnum};
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Lines};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use tracing_subscriber::EnvFilter;

mod env_runtime;

fn get_config(
    model_id: &str,
    revision: &Option<String>,
) -> Result<Config, Box<dyn std::error::Error>> {
    let mut path = std::path::Path::new(model_id).to_path_buf();
    let model_id = model_id.to_string();
    let filename = if !path.exists() {
        // Assume it's a hub id

        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            // env variable has precedence over on file token.
            ApiBuilder::new().with_token(Some(token)).build()?
        } else {
            Api::new()?
        };
        let repo = if let Some(ref revision) = revision {
            api.repo(Repo::with_revision(
                model_id,
                RepoType::Model,
                revision.to_string(),
            ))
        } else {
            api.model(model_id)
        };
        repo.get("config.json")?
    } else {
        path.push("config.json");
        path
    };

    let content = std::fs::read_to_string(filename)?;
    let config: RawConfig = serde_json::from_str(&content)?;

    let config: Config = config.into();
    Ok(config)
}

#[derive(Deserialize)]
struct RawConfig {
    max_position_embeddings: Option<usize>,
    n_positions: Option<usize>,
    model_type: Option<String>,
    max_seq_len: Option<usize>,
    quantization_config: Option<QuantizationConfig>,
    n_embd: Option<usize>,
    hidden_size: Option<usize>,
    num_attention_heads: Option<usize>,
    head_dim: Option<usize>,
    vision_config: Option<VisionConfig>,
    is_encoder_decoder: Option<bool>,
}

#[derive(Deserialize)]
struct QuantizationConfig {
    quant_method: Option<Quantization>,
}

#[derive(Deserialize)]
struct VisionConfig {}

#[derive(Deserialize)]
struct Config {
    max_position_embeddings: Option<usize>,
    quantize: Option<Quantization>,
    head_dim: Option<usize>,
    model_type: Option<String>,
    vision_config: Option<VisionConfig>,
    is_encoder_decoder: bool,
}

impl From<RawConfig> for Config {
    fn from(other: RawConfig) -> Self {
        let max_position_embeddings = other
            .max_position_embeddings
            .or(other.max_seq_len)
            .or(other.n_positions);
        let quantize = other.quantization_config.and_then(|q| q.quant_method);
        let head_dim = other.head_dim.or_else(|| {
            match (other.hidden_size, other.n_embd, other.num_attention_heads) {
                (Some(hidden_size), _, Some(num_attention_heads))
                    if hidden_size % num_attention_heads == 0 =>
                {
                    Some(hidden_size / num_attention_heads)
                }
                // Legacy
                (_, Some(hidden_size), Some(num_attention_heads))
                    if hidden_size % num_attention_heads == 0 =>
                {
                    Some(hidden_size / num_attention_heads)
                }
                _ => None,
            }
        });
        let model_type = other.model_type;
        let vision_config = other.vision_config;
        let is_encoder_decoder = other.is_encoder_decoder.unwrap_or(false);
        Config {
            max_position_embeddings,
            quantize,
            head_dim,
            model_type,
            vision_config,
            is_encoder_decoder,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Quantization {
    /// 4 bit quantization. Requires a specific AWQ quantized model:
    ///   <https://hf.co/models?search=awq>.
    /// Should replace GPTQ models wherever possible because of the better latency
    Awq,
    /// 8 bit quantization, doesn't require specific model.
    /// Should be a drop-in replacement to bitsandbytes with much better performance.
    /// Kernels are from <https://github.com/NetEase-FuXi/EETQ.git>
    Eetq,
    /// Variable bit quantization. Requires a specific EXL2 quantized model:
    /// <https://hf.co/models?search=exl2>. Requires exllama2 kernels and does
    /// not support tensor parallelism (num_shard > 1).
    Exl2,
    /// 4 bit quantization. Requires a specific GTPQ quantized model: <https://hf.co/models?search=gptq>.
    /// text-generation-inference will use exllama (faster) kernels wherever possible, and use
    /// triton kernel (wider support) when it's not.
    /// AWQ has faster kernels.
    Gptq,
    /// Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half,
    /// but it is known that the model will be much slower to run than the native f16.
    // #[deprecated(
    //     since = "1.1.0",
    //     note = "Use `eetq` instead, which provides better latencies overall and is drop-in in most cases"
    // )]
    Bitsandbytes,
    /// Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x,
    /// but it is known that the model will be much slower to run than the native f16.
    BitsandbytesNf4,
    /// Bitsandbytes 4bit. nf4 should be preferred in most cases but maybe this one has better
    /// perplexity performance for you model
    BitsandbytesFp4,
    /// [FP8](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) (e4m3) works on H100 and above
    /// This dtype has native ops should be the fastest if available.
    /// This is currently not the fastest because of local unpacking + padding to satisfy matrix
    /// multiplication limitations.
    Fp8,
    /// FP8 with statically quantized KV cache
    Fp8_KV,
    /// 4 bit quantization. Requires a specific HQQ quantized model.
    Hqq_4bit,
    /// 3 bit quantization. Requires a specific HQQ quantized model.
    Hqq_3bit,
    /// 2 bit quantization. Requires a specific HQQ quantized model.
    Hqq_2bit,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            #[allow(deprecated)]
            // Use `eetq` instead, which provides better latencies overall and is drop-in in most cases
            Quantization::Bitsandbytes => {
                write!(f, "bitsandbytes")
            }
            Quantization::BitsandbytesNf4 => {
                write!(f, "bitsandbytes-nf4")
            }
            Quantization::BitsandbytesFp4 => {
                write!(f, "bitsandbytes-fp4")
            }
            Quantization::Exl2 => {
                write!(f, "exl2")
            }
            Quantization::Gptq => {
                write!(f, "gptq")
            }
            Quantization::Awq => {
                write!(f, "awq")
            }
            Quantization::Eetq => {
                write!(f, "eetq")
            }
            Quantization::Fp8 => {
                write!(f, "fp8")
            }
            Quantization::Fp8_KV => {
                write!(f, "fp8-kv")
            }
            Quantization::Hqq_4bit => {
                write!(f, "hqq-4bit")
            }
            Quantization::Hqq_3bit => {
                write!(f, "hqq-3bit")
            }
            Quantization::Hqq_2bit => {
                write!(f, "hqq-2bit")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    #[clap(name = "float16")]
    Float16,
    #[clap(name = "bfloat16")]
    BFloat16,
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            Dtype::Float16 => {
                write!(f, "float16")
            }
            Dtype::BFloat16 => {
                write!(f, "bfloat16")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum Backend {
    #[clap(name = "fa2")]
    FA2,
    #[clap(name = "flashinfer")]
    FlashInfer,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in sync with `server`.
        match self {
            Backend::FA2 => {
                write!(f, "fa2")
            }
            Backend::FlashInfer => {
                write!(f, "flashinfer")
            }
        }
    }
}

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the model to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models> like
    /// `gpt2` or `mistralai/Mistral-7B-Instruct-v0.1`.
    /// Or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers
    #[clap(default_value = "mistralai/Mistral-7B-Instruct-v0.1", long, env)]
    model_id: String,

    /// The name of the adapter to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models>
    /// or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers.
    /// Should be compatible with the model specified in `model_id`.
    #[clap(long, env)]
    adapter_id: Option<String>,

    /// The source of the model to load.
    /// Can be `hub` or `s3`.
    /// `hub` will load the model from the huggingface hub.
    /// `s3` will load the model from the predibase S3 bucket.
    #[clap(default_value = "hub", long, env)]
    source: String,

    /// The default source of the dynamic adapters to load.
    /// If not defined, we fallback to the value from `adapter_source`
    /// Can be `hub` or `s3` or `pbase`
    /// `hub` will load the model from the huggingface hub.
    /// `s3` will load the model from the predibase S3 bucket.
    /// `pbase` will load an s3 model but resolve the metadata from a predibase server
    #[clap(long, env)]
    default_adapter_source: Option<String>,

    /// The source of the static adapter to load.
    /// Can be `hub` or `s3` or `pbase`
    /// `hub` will load the model from the huggingface hub.
    /// `s3` will load the model from the predibase S3 bucket.
    /// `pbase` will load an s3 model but resolve the metadata from a predibase server
    #[clap(default_value = "hub", long, env)]
    adapter_source: String,

    /// The actual revision of the model if you're referring to a model
    /// on the hub. You can use a specific commit id or a branch like `refs/pr/2`.
    #[clap(long, env)]
    revision: Option<String>,

    /// The number of tokenizer workers used for payload validation and truncation inside the
    /// router.
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,

    /// Whether to shard the model across multiple GPUs
    /// By default LoRAX will use all available GPUs to run
    /// the model. Setting it to `false` deactivates `num_shard`.
    #[clap(long, env)]
    sharded: Option<bool>,

    /// The number of shards to use if you don't want to use all GPUs on a given machine.
    /// You can use `CUDA_VISIBLE_DEVICES=0,1 lorax-launcher... --num_shard 2`
    /// and `CUDA_VISIBLE_DEVICES=2,3 lorax-launcher... --num_shard 2` to
    /// launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.
    #[clap(long, env)]
    num_shard: Option<usize>,

    /// Whether you want the model to be quantized. This will use `bitsandbytes` for
    /// quantization on the fly, or `gptq`.
    #[clap(long, env, value_enum)]
    quantize: Option<Quantization>,

    /// Whether you want to compile the model into a CUDA graph.
    /// This will speed up decoding but increase GPU memory usage.
    #[clap(long, env, value_enum)]
    compile: bool,

    /// The number of speculative tokens to generate in the model per step.
    /// Defaults to 0, meaning no speculative decoding.
    #[clap(long, env)]
    speculative_tokens: Option<usize>,

    /// The list of adapter ids to preload during initialization (to avoid cold start times).
    #[clap(long, env)]
    preloaded_adapter_ids: Vec<String>,

    /// The source to use for the preloaded adapters.
    /// If unset, will default to using the `adapter_source` value.
    /// Can be `hub` or `s3` or `pbase`
    /// `hub` will load the model from the huggingface hub.
    /// `s3` will load the model from the predibase S3 bucket.
    /// `pbase` will load an s3 model but resolve the metadata from a predibase server
    #[clap(long, env)]
    preloaded_adapter_source: Option<String>,

    /// The API token to use when fetching adapters from pbase.
    /// If specified, will set the environment variable PREDIBASE_API_TOKEN.
    #[clap(long, env)]
    predibase_api_token: Option<String>,

    /// The dtype to be forced upon the model. This option cannot be used with `--quantize`.
    #[clap(long, env, value_enum)]
    dtype: Option<Dtype>,

    /// Whether you want to execute hub modelling code. Explicitly passing a `revision` is
    /// encouraged when loading a model with custom code to ensure no malicious code has been
    /// contributed in a newer revision.
    #[clap(long, env, value_enum)]
    trust_remote_code: bool,

    /// The maximum amount of concurrent requests for this particular deployment.
    /// Having a low limit will refuse clients requests instead of having them
    /// wait for too long and is usually good to handle backpressure correctly.
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,

    /// This is the maximum allowed value for clients to set `best_of`.
    /// Best of makes `n` generations at the same time, and return the best
    /// in terms of overall log probability over the entire generated sequence
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,

    /// This is the maximum allowed value for clients to set `stop_sequences`.
    /// Stop sequences are used to allow the model to stop on more than just
    /// the EOS token, and enable more complex "prompting" where users can preprompt
    /// the model in a specific way and define their "own" stop token aligned with
    /// their prompt.
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,

    /// This is the maximum allowed input length (expressed in number of tokens)
    /// for users. The larger this value, the longer prompt users can send which
    /// can impact the overall memory required to handle the load.
    /// Please note that some models have a finite range of sequence they can handle.
    /// Default to min(max_position_embeddings - 1, 4095)
    #[clap(long, env)]
    max_input_length: Option<usize>,

    /// This is the most important value to set as it defines the "memory budget"
    /// of running clients requests.
    /// Clients will send input sequences and ask to generate `max_new_tokens`
    /// on top. with a value of `1512` users can send either a prompt of
    /// `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for
    /// `1511` max_new_tokens.
    /// The larger this value, the larger amount each request will be in your RAM
    /// and the less effective batching can be.
    /// Default to min(max_position_embeddings, 4096)
    #[clap(long, env)]
    max_total_tokens: Option<usize>,

    /// This represents the ratio of waiting queries vs running queries where
    /// you want to start considering pausing the running queries to include the waiting
    /// ones into the same batch.
    /// `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's
    /// only 10 queries left in the current batch we check if we can fit those 12
    /// waiting queries into the batching strategy, and if yes, then batching happens
    /// delaying the 10 running queries by a `prefill` run.
    ///
    /// This setting is only applied if there is room in the batch
    /// as defined by `max_batch_total_tokens`.
    #[clap(default_value = "1.2", long, env)]
    waiting_served_ratio: f32,

    /// Limits the number of tokens for the prefill operation.
    /// Since this operation take the most memory and is compute bound, it is interesting
    /// to limit the number of requests that can be sent.
    /// Default to `max_input_tokens + 50` to give a bit of room.
    #[clap(long, env)]
    max_batch_prefill_tokens: Option<u32>,

    /// **IMPORTANT** This is one critical control to allow maximum usage
    /// of the available hardware.
    ///
    /// This represents the total amount of potential tokens within a batch.
    /// When using padding (not recommended) this would be equivalent of
    /// `batch_size` * `max_total_tokens`.
    ///
    /// However in the non-padded (flash attention) version this can be much finer.
    ///
    /// For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100`
    /// or a single query of `1000` tokens.
    ///
    /// Overall this number should be the largest possible amount that fits the
    /// remaining memory (after the model is loaded). Since the actual memory overhead
    /// depends on other parameters like if you're using quantization, flash attention
    /// or the model implementation, LoRAX cannot infer this number
    /// automatically.
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,

    /// This setting defines how many tokens can be passed before forcing the waiting
    /// queries to be put on the batch (if the size of the batch allows for it).
    /// New queries require 1 `prefill` forward, which is different from `decode`
    /// and therefore you need to pause the running batch in order to run `prefill`
    /// to create the correct values for the waiting queries to be able to join the batch.
    ///
    /// With a value too small, queries will always "steal" the compute to run `prefill`
    /// and running queries will be delayed by a lot.
    ///
    /// With a value too big, waiting queries could wait for a very long time
    /// before being allowed a slot in the running batch. If your server is busy
    /// that means that requests that could run in ~2s on an empty server could
    /// end up running in ~20s because the query had to wait for 18s.
    ///
    /// This number is expressed in number of tokens to make it a bit more
    /// "model" agnostic, but what should really matter is the overall latency
    /// for end users.
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,

    /// Whether to prioritize running prefill before decode to increase batch size during decode (throughput) over
    /// liveness in earlier requests (latency). For batch use cases that are not latnecy sensitive, this should be set
    /// to true.
    #[clap(long, env)]
    eager_prefill: Option<bool>,

    /// Split prefill requests into multiple chunks and batch them with decode requests. For high QPS scenarios, this
    /// can greatly improve throughput by overlapping request types. See: https://arxiv.org/pdf/2308.16369.
    #[clap(long, env)]
    chunked_prefill: Option<bool>,

    /// Whether to use the prefix caching mechanism. This will skip computing attention on previously cached prefixes
    /// in the prompt. Useful in cases where many queries need to be run over a shared context, or for long multi-turn
    /// chats conversations.
    #[clap(long, env)]
    prefix_caching: Option<bool>,

    /// Whether to merge the weights of the adapter with the base model weights. This will disable dynamic adapter
    /// loading.
    #[clap(long, env, value_enum)]
    merge_adapter_weights: bool,

    /// Maximum number of adapters that can be placed on the GPU and accept requests at a time.
    #[clap(default_value = "1024", long, env)]
    max_active_adapters: usize,

    /// The time in seconds between adapter exchanges.
    #[clap(default_value = "2", long, env)]
    adapter_cycle_time_s: u64,

    /// Reservation of memory set aside for loading adapters onto the GPU.
    /// Increasing this value will reduce the size of the KV cache in exchange for allowing more
    /// adapters to be loaded onto the GPU at once.
    /// This value is NOT scaled relative to `cuda_memory_fraction`, but is expressed in absolute terms.
    #[clap(default_value = "0.1", long, env)]
    adapter_memory_fraction: f32,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3000", long, short, env)]
    port: u16,

    /// The name of the socket for gRPC communication between the webserver
    /// and the shards.
    #[clap(default_value = "/tmp/lorax-server", long, env)]
    shard_uds_path: String,

    /// The address the master shard will listen on. (setting used by torch distributed)
    #[clap(default_value = "localhost", long, env)]
    master_addr: String,

    /// The address the master port will listen on. (setting used by torch distributed)
    #[clap(default_value = "29500", long, env)]
    master_port: usize,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    huggingface_hub_cache: Option<String>,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    weights_cache_override: Option<String>,

    /// For some models (like llama), LoRAX implemented custom
    /// cuda kernels to speed up inference. Those kernels were only tested on A100.
    /// Use this flag to disable them if you're running on different hardware and
    /// encounter issues.
    #[clap(long, env)]
    disable_custom_kernels: bool,

    /// Limit the CUDA available memory.
    /// The allowed value equals the total visible memory multiplied by cuda-memory-fraction.
    #[clap(default_value = "1.0", long, env)]
    cuda_memory_fraction: f32,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env)]
    json_output: bool,

    #[clap(long, env)]
    otlp_endpoint: Option<String>,

    #[clap(long, env)]
    cors_allow_origin: Vec<String>,

    #[clap(long, env)]
    cors_allow_header: Vec<String>,

    #[clap(long, env)]
    cors_expose_header: Vec<String>,

    #[clap(long, env)]
    cors_allow_method: Vec<String>,

    #[clap(long, env)]
    cors_allow_credentials: Option<bool>,

    #[clap(long, env)]
    watermark_gamma: Option<f32>,
    #[clap(long, env)]
    watermark_delta: Option<f32>,

    /// Enable ngrok tunneling
    #[clap(long, env)]
    ngrok: bool,

    /// ngrok authentication token
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,

    /// ngrok edge
    #[clap(long, env)]
    ngrok_edge: Option<String>,

    /// Display a lot of information about your runtime environment
    #[clap(long, short, action)]
    env: bool,

    /// Download model weights only
    #[clap(long, env)]
    download_only: bool,

    /// The path to the tokenizer config file. This path is used to load the tokenizer configuration which may
    /// include a `chat_template`. If not provided, the default config will be used from the model hub.
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,

    /// The backend to use for the model. Can be `fa2` or `flashinfer`.
    #[clap(default_value = "fa2", long, env, value_enum)]
    backend: Backend,

    /// The embedding dimension to use for the model.
    #[clap(long, env)]
    embedding_dim: Option<usize>,

    #[clap(long, env)]
    disable_sgmv: bool,
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed(usize),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_id: String,
    adapter_id: String,
    revision: Option<String>,
    source: String,
    adapter_source: String,
    quantize: Option<Quantization>,
    compile: bool,
    speculative_tokens: Option<usize>,
    preloaded_adapter_ids: Vec<String>,
    preloaded_adapter_source: Option<String>,
    predibase_api_token: Option<String>,
    dtype: Option<Dtype>,
    trust_remote_code: bool,
    uds_path: String,
    rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: usize,
    huggingface_hub_cache: Option<String>,
    weights_cache_override: Option<String>,
    disable_custom_kernels: bool,
    watermark_gamma: Option<f32>,
    watermark_delta: Option<f32>,
    cuda_memory_fraction: f32,
    adapter_memory_fraction: f32,
    prefix_caching: Option<bool>,
    chunked_prefill: Option<bool>,
    merge_adapter_weights: bool,
    backend: Backend,
    otlp_endpoint: Option<String>,
    status_sender: mpsc::Sender<ShardStatus>,
    shutdown: Arc<AtomicBool>,
    _shutdown_sender: mpsc::Sender<()>,
    embedding_dim: Option<usize>,
    disable_sgmv: bool,
) {
    // Enter shard-manager tracing span
    let _span = tracing::span!(tracing::Level::INFO, "shard-manager", rank = rank).entered();

    // Get UDS path
    let uds_string = format!("{uds_path}-{rank}");
    let uds = Path::new(&uds_string);
    // Clean previous runs
    if uds.exists() {
        fs::remove_file(uds).unwrap();
    }

    // Process args
    let mut shard_args = vec![
        "serve".to_string(),
        model_id,
        "--uds-path".to_string(),
        uds_path,
        "--logger-level".to_string(),
        "INFO".to_string(),
        "--json-output".to_string(),
        "--source".to_string(),
        source,
        "--adapter-source".to_string(),
        adapter_source,
    ];

    // Check if adapter id is non-empty string
    if !adapter_id.is_empty() {
        shard_args.push("--adapter-id".to_string());
        shard_args.push(adapter_id);
    }

    // Activate trust remote code
    if trust_remote_code {
        shard_args.push("--trust-remote-code".to_string());
    }

    // Activate tensor parallelism
    if world_size > 1 {
        shard_args.push("--sharded".to_string());
    }

    if let Some(quantize) = quantize {
        shard_args.push("--quantize".to_string());
        shard_args.push(quantize.to_string())
    }

    // CUDA graph compilation
    if compile {
        shard_args.push("--compile".to_string());
    }

    // Speculative decoding
    if let Some(speculative_tokens) = speculative_tokens {
        shard_args.push("--speculative-tokens".to_string());
        shard_args.push(speculative_tokens.to_string())
    }

    // Preloaded adapters
    for adapter_id in preloaded_adapter_ids {
        shard_args.push("--preloaded-adapter-ids".to_string());
        shard_args.push(adapter_id);
    }

    // Merge adapter weights
    if merge_adapter_weights {
        shard_args.push("--merge-adapter-weights".to_string());
    }
    // Preloaded adapter source
    if let Some(preloaded_adapter_source) = preloaded_adapter_source {
        shard_args.push("--preloaded-adapter-source".to_string());
        shard_args.push(preloaded_adapter_source);
    }

    if let Some(dtype) = dtype {
        shard_args.push("--dtype".to_string());
        shard_args.push(dtype.to_string())
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_args.push("--revision".to_string());
        shard_args.push(revision)
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = otlp_endpoint {
        shard_args.push("--otlp-endpoint".to_string());
        shard_args.push(otlp_endpoint);
    }

    // Embedding dimension
    if let Some(embedding_dim) = embedding_dim {
        shard_args.push("--embedding-dim".to_string());
        shard_args.push(embedding_dim.to_string())
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    if let Some(predibase_api_token) = predibase_api_token {
        envs.push((
            "PREDIBASE_API_TOKEN".into(),
            predibase_api_token.to_string().into(),
        ));
    }

    // Torch Distributed Env vars
    envs.push(("RANK".into(), rank.to_string().into()));
    envs.push(("WORLD_SIZE".into(), world_size.to_string().into()));
    envs.push(("MASTER_ADDR".into(), master_addr.into()));
    envs.push(("MASTER_PORT".into(), master_port.to_string().into()));
    envs.push(("NCCL_ASYNC_ERROR_HANDLING".into(), "1".into()));

    // CUDA memory fraction
    envs.push((
        "CUDA_MEMORY_FRACTION".into(),
        cuda_memory_fraction.to_string().into(),
    ));

    // Adapter memory fraction
    envs.push((
        "ADAPTER_MEMORY_FRACTION".into(),
        adapter_memory_fraction.to_string().into(),
    ));

    // Prefix caching
    if let Some(prefix_caching) = prefix_caching {
        envs.push(("PREFIX_CACHING".into(), prefix_caching.to_string().into()));
    }

    // Chunked prefill
    if let Some(chunked_prefill) = chunked_prefill {
        envs.push(("CHUNKED_PREFILL".into(), chunked_prefill.to_string().into()));
    }

    // Backend
    if backend == Backend::FlashInfer {
        envs.push(("FLASH_INFER".into(), "1".into()));
    }

    if disable_sgmv {
        envs.push(("DISABLE_SGMV".into(), "1".into()))
    }

    // Safetensors load fast
    envs.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Disable progress bars to prevent hanging in containers
    envs.push(("HF_HUB_DISABLE_PROGRESS_BARS".into(), "1".into()));

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // If huggingface_hub_cache is some, pass it to the shard
    // Useful when running inside a docker container
    if let Some(huggingface_hub_cache) = huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // If weights_cache_override is some, pass it to the shard
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // If disable_custom_kernels is true, pass it to the shard as an env var
    if disable_custom_kernels {
        envs.push(("DISABLE_CUSTOM_KERNELS".into(), "True".into()))
    }

    // Watermark Gamma
    if let Some(watermark_gamma) = watermark_gamma {
        envs.push(("WATERMARK_GAMMA".into(), watermark_gamma.to_string().into()))
    }

    // Watermark Delta
    if let Some(watermark_delta) = watermark_delta {
        envs.push(("WATERMARK_DELTA".into(), watermark_delta.to_string().into()))
    }

    // Start process
    tracing::info!("Starting shard");
    let mut p = match Command::new("lorax-server")
        .args(shard_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("lorax-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            }
            {
                tracing::error!("{}", err);
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }
    };

    let shard_stdout = BufReader::new(p.stdout.take().unwrap());

    thread::spawn(move || {
        log_lines(shard_stdout.lines());
    });

    let shard_stderr = BufReader::new(p.stderr.take().unwrap());

    // We read stderr in another thread as it seems that lines() can block in some cases
    let (err_sender, err_receiver) = mpsc::channel();
    thread::spawn(move || {
        for line in shard_stderr.lines().flatten() {
            err_sender.send(line).unwrap_or(());
        }
    });

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if let Some(exit_status) = p.try_wait().unwrap() {
            let mut err = String::new();
            while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                err = err + "\n" + &line;
            }

            tracing::error!("Shard complete standard error output:\n{err}");

            if let Some(signal) = exit_status.signal() {
                tracing::error!("Shard process was signaled to shutdown with signal {signal}");
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }

        // We received a shutdown signal
        if shutdown.load(Ordering::SeqCst) {
            p.kill().unwrap();
            let _ = p.wait();
            tracing::info!("Shard terminated");
            return;
        }

        // Shard is ready
        if uds.exists() && !ready {
            tracing::info!("Shard ready in {:?}", start_time.elapsed());
            status_sender.send(ShardStatus::Ready).unwrap();
            ready = true;
        } else if !ready && wait_time.elapsed() > Duration::from_secs(10) {
            tracing::info!("Waiting for shard to be ready...");
            wait_time = Instant::now();
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<AtomicBool>, shutdown_receiver: &mpsc::Receiver<()>) {
    tracing::info!("Shutting down shards");
    // Update shutdown value to true
    // This will be picked up by the shard manager
    shutdown.store(true, Ordering::SeqCst);

    // Wait for shards to shutdown
    // This will block till all shutdown_sender are dropped
    let _ = shutdown_receiver.recv();
}

fn num_cuda_devices() -> Option<usize> {
    let devices = match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(devices) => devices,
        Err(_) => env::var("NVIDIA_VISIBLE_DEVICES").ok()?,
    };
    let n_devices = devices.split(',').count();
    Some(n_devices)
}

#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum PythonLogLevelEnum {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
    Critical,
}

#[derive(Deserialize)]
struct PythonLogLevel {
    name: PythonLogLevelEnum,
}

#[derive(Deserialize)]
struct PythonLogRecord {
    level: PythonLogLevel,
}

#[derive(Deserialize)]
struct PythonLogMessage {
    text: String,
    record: PythonLogRecord,
}

impl PythonLogMessage {
    fn trace(&self) {
        match self.record.level.name {
            PythonLogLevelEnum::Trace => tracing::trace!("{}", self.text),
            PythonLogLevelEnum::Debug => tracing::debug!("{}", self.text),
            PythonLogLevelEnum::Info => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Success => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Warning => tracing::warn!("{}", self.text),
            PythonLogLevelEnum::Error => tracing::error!("{}", self.text),
            PythonLogLevelEnum::Critical => tracing::error!("{}", self.text),
        }
    }
}

impl TryFrom<&String> for PythonLogMessage {
    type Error = serde_json::Error;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        serde_json::from_str::<Self>(value)
    }
}

fn log_lines<S: Sized + BufRead>(lines: Lines<S>) {
    for line in lines.flatten() {
        match PythonLogMessage::try_from(&line) {
            Ok(log) => log.trace(),
            Err(_) => tracing::debug!("{line}"),
        }
    }
}

fn find_num_shards(
    sharded: Option<bool>,
    num_shard: Option<usize>,
) -> Result<usize, LauncherError> {
    // get the number of shards given `sharded` and `num_shard`
    let num_shard = match (sharded, num_shard) {
        (Some(true), None) => {
            // try to default to the number of available GPUs
            tracing::info!("Parsing num_shard from CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES");
            let n_devices = num_cuda_devices()
                .expect("--num-shard and CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES are not set");
            if n_devices <= 1 {
                return Err(LauncherError::NotEnoughCUDADevices(format!(
                    "`sharded` is true but only found {n_devices} CUDA devices"
                )));
            }
            n_devices
        }
        (Some(true), Some(num_shard)) => {
            // we can't have only one shard while sharded
            if num_shard <= 1 {
                return Err(LauncherError::ArgumentValidation(
                    "`sharded` is true but `num_shard` <= 1".to_string(),
                ));
            }
            num_shard
        }
        (Some(false), Some(num_shard)) => num_shard,
        (Some(false), None) => 1,
        (None, None) => num_cuda_devices().unwrap_or(1),
        (None, Some(num_shard)) => num_shard,
    };
    if num_shard < 1 {
        return Err(LauncherError::ArgumentValidation(
            "`num_shard` cannot be < 1".to_string(),
        ));
    }
    Ok(num_shard)
}

#[derive(Debug)]
enum LauncherError {
    ArgumentValidation(String),
    NotEnoughCUDADevices(String),
    DownloadError,
    ShardCannotStart,
    ShardDisconnected,
    ShardFailed,
    WebserverFailed,
    WebserverCannotStart,
}

fn download_convert_model(
    model_id: String,
    args: &Args,
    running: Arc<AtomicBool>,
) -> Result<(), LauncherError> {
    // Enter download tracing span
    let _span = tracing::span!(tracing::Level::INFO, "download").entered();

    let mut download_args = vec![
        "download-weights".to_string(),
        model_id,
        "--extension".to_string(),
        ".safetensors".to_string(),
        "--logger-level".to_string(),
        "INFO".to_string(),
        "--json-output".to_string(),
        "--source".to_string(),
        args.source.clone(),
        "--adapter-source".to_string(),
        args.adapter_source.clone(),
    ];

    // Model optional revision
    if let Some(revision) = &args.revision {
        download_args.push("--revision".to_string());
        download_args.push(revision.to_string())
    }

    // check if option has a value
    if let Some(adapter_id) = &args.adapter_id {
        download_args.push("--adapter-id".to_string());
        download_args.push(adapter_id.to_string());
    }

    // Embedding dimension
    if let Some(embedding_dim) = args.embedding_dim {
        download_args.push("--embedding-dim".to_string());
        download_args.push(embedding_dim.to_string())
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // If huggingface_hub_cache is set, pass it to the download process
    // Useful when running inside a docker container
    if let Some(ref huggingface_hub_cache) = args.huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // Disable progress bars to prevent hanging in containers
    envs.push(("HF_HUB_DISABLE_PROGRESS_BARS".into(), "1".into()));

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // If args.weights_cache_override is some, pass it to the download process
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = &args.weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // Start process
    tracing::info!("Starting download process.");
    let mut download_process = match Command::new("lorax-server")
        .args(download_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("lorax-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            } else {
                tracing::error!("{}", err);
            }

            return Err(LauncherError::DownloadError);
        }
    };

    let download_stdout = BufReader::new(download_process.stdout.take().unwrap());

    thread::spawn(move || {
        log_lines(download_stdout.lines());
    });

    let download_stderr = BufReader::new(download_process.stderr.take().unwrap());

    // We read stderr in another thread as it seems that lines() can block in some cases
    let (err_sender, err_receiver) = mpsc::channel();
    thread::spawn(move || {
        for line in download_stderr.lines().flatten() {
            err_sender.send(line).unwrap_or(());
        }
    });

    loop {
        if let Some(status) = download_process.try_wait().unwrap() {
            if status.success() {
                tracing::info!("Successfully downloaded weights.");
                break;
            }

            let mut err = String::new();
            while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                err = err + "\n" + &line;
            }
            if let Some(signal) = status.signal() {
                tracing::error!(
                    "Download process was signaled to shutdown with signal {signal}: {err}"
                );
            } else {
                tracing::error!("Download encountered an error: {err}");
            }

            return Err(LauncherError::DownloadError);
        }
        if !running.load(Ordering::SeqCst) {
            terminate("download", download_process, Duration::from_secs(10)).unwrap();
            return Ok(());
        }
        sleep(Duration::from_millis(100));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spawn_shards(
    num_shard: usize,
    args: &Args,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
    shutdown_sender: mpsc::Sender<()>,
    status_receiver: &mpsc::Receiver<ShardStatus>,
    status_sender: mpsc::Sender<ShardStatus>,
    running: Arc<AtomicBool>,
) -> Result<(), LauncherError> {
    // Start shard processes
    for rank in 0..num_shard {
        let model_id = args.model_id.clone();
        let adapter_id = args.adapter_id.clone().unwrap_or_default();
        let revision = args.revision.clone();
        let source: String = args.source.clone();
        let adapter_source: String = args.adapter_source.clone();
        let uds_path = args.shard_uds_path.clone();
        let master_addr = args.master_addr.clone();
        let huggingface_hub_cache = args.huggingface_hub_cache.clone();
        let weights_cache_override = args.weights_cache_override.clone();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        let otlp_endpoint = args.otlp_endpoint.clone();
        let quantize = args.quantize;
        let compile = args.compile;
        let speculative_tokens = args.speculative_tokens;
        let preloaded_adapter_ids = args.preloaded_adapter_ids.clone();
        let preloaded_adapter_source = args.preloaded_adapter_source.clone();
        let predibase_api_token = args.predibase_api_token.clone();
        let dtype = args.dtype;
        let trust_remote_code = args.trust_remote_code;
        let master_port = args.master_port;
        let disable_custom_kernels = args.disable_custom_kernels;
        let watermark_gamma = args.watermark_gamma;
        let watermark_delta = args.watermark_delta;
        let cuda_memory_fraction = args.cuda_memory_fraction;
        let adapter_memory_fraction = args.adapter_memory_fraction;
        let prefix_caching = args.prefix_caching;
        let chunked_prefill = args.chunked_prefill;
        let merge_adapter_weights = args.merge_adapter_weights;
        let backend = args.backend;
        let embedding_dim = args.embedding_dim;
        let disable_sgmv = args.disable_sgmv;
        thread::spawn(move || {
            shard_manager(
                model_id,
                adapter_id,
                revision,
                source,
                adapter_source,
                quantize,
                compile,
                speculative_tokens,
                preloaded_adapter_ids,
                preloaded_adapter_source,
                predibase_api_token,
                dtype,
                trust_remote_code,
                uds_path,
                rank,
                num_shard,
                master_addr,
                master_port,
                huggingface_hub_cache,
                weights_cache_override,
                disable_custom_kernels,
                watermark_gamma,
                watermark_delta,
                cuda_memory_fraction,
                adapter_memory_fraction,
                prefix_caching,
                chunked_prefill,
                merge_adapter_weights,
                backend,
                otlp_endpoint,
                status_sender,
                shutdown,
                shutdown_sender,
                embedding_dim,
                disable_sgmv,
            )
        });
    }
    drop(shutdown_sender);

    // Wait for shard to start
    let mut shard_ready = 0;
    while running.load(Ordering::SeqCst) {
        match status_receiver.try_recv() {
            Ok(ShardStatus::Ready) => {
                shard_ready += 1;
                if shard_ready == num_shard {
                    break;
                }
            }
            Err(TryRecvError::Empty) => {
                sleep(Duration::from_millis(100));
            }
            Ok(ShardStatus::Failed(rank)) => {
                tracing::error!("Shard {rank} failed to start");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardCannotStart);
            }
            Err(TryRecvError::Disconnected) => {
                tracing::error!("Shard status channel disconnected");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardDisconnected);
            }
        }
    }
    Ok(())
}

fn spawn_webserver(
    args: Args,
    max_input_tokens: usize,
    max_total_tokens: usize,
    max_batch_prefill_tokens: u32,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
) -> Result<Child, LauncherError> {
    // All shard started
    // Start webserver
    tracing::info!("Starting Webserver");
    let mut router_args = vec![
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-best-of".to_string(),
        args.max_best_of.to_string(),
        "--max-stop-sequences".to_string(),
        args.max_stop_sequences.to_string(),
        "--max-input-length".to_string(),
        max_input_tokens.to_string(),
        "--max-total-tokens".to_string(),
        max_total_tokens.to_string(),
        "--max-batch-prefill-tokens".to_string(),
        max_batch_prefill_tokens.to_string(),
        "--max-active-adapters".to_string(),
        args.max_active_adapters.to_string(),
        "--adapter-cycle-time-s".to_string(),
        args.adapter_cycle_time_s.to_string(),
        "--waiting-served-ratio".to_string(),
        args.waiting_served_ratio.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--validation-workers".to_string(),
        args.validation_workers.to_string(),
        "--hostname".to_string(),
        args.hostname.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-name".to_string(),
        args.model_id,
    ];
    // Set the default adapter source as "default_adapter_source" if defined, otherwise, "adapter_source"
    // adapter_source in the router is used to set the default adapter source for dynamically loaded adapters.
    let adapter_source;
    if let Some(default_adapter_source) = args.default_adapter_source {
        adapter_source = default_adapter_source
    } else {
        adapter_source = args.adapter_source
    }

    router_args.push("--adapter-source".to_string());
    router_args.push(adapter_source.to_string());

    // Tokenizer config path
    if let Some(ref tokenizer_config_path) = args.tokenizer_config_path {
        router_args.push("--tokenizer-config-path".to_string());
        router_args.push(tokenizer_config_path.to_string());
    }

    // Model optional max batch total tokens
    if let Some(max_batch_total_tokens) = args.max_batch_total_tokens {
        router_args.push("--max-batch-total-tokens".to_string());
        router_args.push(max_batch_total_tokens.to_string());
    }

    // Model optional revision
    if let Some(ref revision) = args.revision {
        router_args.push("--revision".to_string());
        router_args.push(revision.to_string())
    }

    if args.json_output {
        router_args.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = args.otlp_endpoint {
        router_args.push("--otlp-endpoint".to_string());
        router_args.push(otlp_endpoint);
    }

    // CORS origins
    for origin in args.cors_allow_origin.into_iter() {
        router_args.push("--cors-allow-origin".to_string());
        router_args.push(origin);
    }

    // CORS methods
    for origin in args.cors_allow_method.into_iter() {
        router_args.push("--cors-allow-method".to_string());
        router_args.push(origin);
    }

    // CORS Allow headers
    for origin in args.cors_allow_header.into_iter() {
        router_args.push("--cors-allow-header".to_string());
        router_args.push(origin);
    }

    // CORS expose headers
    for origin in args.cors_expose_header.into_iter() {
        router_args.push("--cors-expose-header".to_string());
        router_args.push(origin);
    }

    // CORS credentials
    for origin in args.cors_allow_credentials.into_iter() {
        router_args.push("--cors-allow-credentials".to_string());
        router_args.push(origin.to_string());
    }

    if args.eager_prefill.unwrap_or(false) {
        router_args.push("--eager-prefill".to_string());
    }

    if args.prefix_caching.unwrap_or(false) {
        router_args.push("--prefix-caching".to_string());
    }

    // Ngrok
    if args.ngrok {
        router_args.push("--ngrok".to_string());
        router_args.push("--ngrok-authtoken".to_string());
        router_args.push(args.ngrok_authtoken.unwrap());
        router_args.push("--ngrok-edge".to_string());
        router_args.push(args.ngrok_edge.unwrap());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    let mut webserver = match Command::new("lorax-router")
        .args(router_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            tracing::error!("Failed to start webserver: {}", err);
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("lorax-router not found in PATH");
                tracing::error!("Please install it with `make install-router`")
            } else {
                tracing::error!("{}", err);
            }

            shutdown_shards(shutdown, shutdown_receiver);
            return Err(LauncherError::WebserverCannotStart);
        }
    };

    // Redirect STDOUT and STDERR to the console
    let webserver_stdout = webserver.stdout.take().unwrap();
    let webserver_stderr = webserver.stderr.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(webserver_stdout);
        let stderr = BufReader::new(webserver_stderr);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
        for line in stderr.lines() {
            println!("{}", line.unwrap());
        }
    });
    Ok(webserver)
}

fn terminate(process_name: &str, mut process: Child, timeout: Duration) -> io::Result<ExitStatus> {
    tracing::info!("Terminating {process_name}");

    let terminate_time = Instant::now();
    signal::kill(Pid::from_raw(process.id() as i32), Signal::SIGTERM).unwrap();

    tracing::info!("Waiting for {process_name} to gracefully shutdown");

    while terminate_time.elapsed() < timeout {
        if let Some(status) = process.try_wait()? {
            tracing::info!("{process_name} terminated");
            return Ok(status);
        }
        sleep(Duration::from_millis(100));
    }

    tracing::info!("Killing {process_name}");

    process.kill()?;
    let exit_status = process.wait()?;

    tracing::info!("{process_name} killed");
    Ok(exit_status)
}

fn main() -> Result<(), LauncherError> {
    // Pattern match configuration
    let args: Args = Args::parse();

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    if args.json_output {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .compact()
            .init();
    }

    if args.env {
        let env_runtime = env_runtime::Env::new();
        tracing::info!("{}", env_runtime);
    }

    tracing::info!("{:?}", args);

    let config: Option<Config> = get_config(&args.model_id, &args.revision).ok();
    let max_default = 4096;
    let max_position_embeddings = if let Some(config) = &config {
        if let Some(max_position_embeddings) = config.max_position_embeddings {
            if max_position_embeddings > max_default {
                let max = max_position_embeddings;
                if args.max_input_length.is_none()
                    && args.max_total_tokens.is_none()
                    && args.max_batch_prefill_tokens.is_none()
                {
                    tracing::info!("Model supports up to {max} but tgi will now set its default to {max_default} instead. This is to save VRAM by refusing large prompts in order to allow more users on the same hardware. You can increase that size using `--max-batch-prefill-tokens={} --max-total-tokens={max} --max-input-tokens={}`.", max + 50, max - 1);
                }
                max_default
            } else {
                max_position_embeddings
            }
        } else {
            max_default
        }
    } else {
        max_default
    };

    // Defaults
    let max_input_tokens = {
        match args.max_input_length {
            Some(max_input_tokens) => max_input_tokens,
            None => {
                let value = max_position_embeddings - 1;
                tracing::info!("Default `max_input_tokens` to {value}");
                value
            }
        }
    };
    let max_total_tokens = {
        match args.max_total_tokens {
            Some(max_total_tokens) => max_total_tokens,
            None => {
                let value = max_position_embeddings;
                tracing::info!("Default `max_total_tokens` to {value}");
                value
            }
        }
    };
    let max_batch_prefill_tokens = {
        match args.max_batch_prefill_tokens {
            Some(max_batch_prefill_tokens) => max_batch_prefill_tokens,
            None => {
                // Adding some edge in order to account for potential block_size alignement
                // issue.
                let value: u32 = (max_input_tokens + 50) as u32;
                tracing::info!("Default `max_batch_prefill_tokens` to {value}");
                value
            }
        }
    };

    // Validate args
    if max_input_tokens >= max_total_tokens {
        return Err(LauncherError::ArgumentValidation(
            "`max_input_length` must be < `max_total_tokens`".to_string(),
        ));
    }

    if args.validation_workers == 0 {
        return Err(LauncherError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }
    if args.trust_remote_code {
        tracing::warn!(
            "`trust_remote_code` is set. Trusting that model `{}` do not contain malicious code.",
            args.model_id
        );
    }

    let num_shard = find_num_shards(args.sharded, args.num_shard)?;
    if num_shard > 1 {
        tracing::info!("Sharding model on {num_shard} processes");
    }

    if let Some(ref max_batch_total_tokens) = args.max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(LauncherError::ArgumentValidation(format!(
                "`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {} and {}",
                max_batch_prefill_tokens, max_batch_total_tokens
            )));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(LauncherError::ArgumentValidation(format!(
                "`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {} and {}",
                max_total_tokens, max_batch_total_tokens
            )));
        }
    }

    if args.ngrok {
        if args.ngrok_authtoken.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-authtoken` must be set when using ngrok tunneling".to_string(),
            ));
        }

        if args.ngrok_edge.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-edge` must be set when using ngrok tunneling".to_string(),
            ));
        }
    }

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Download and convert model weights
    download_convert_model(args.model_id.to_string(), &args, running.clone())?;

    // we're just downloading the model
    if args.download_only {
        return Ok(());
    }

    if !running.load(Ordering::SeqCst) {
        // Launcher was asked to stop
        return Ok(());
    }

    // Shared shutdown bool
    let shutdown = Arc::new(AtomicBool::new(false));
    // Shared shutdown channel
    // When shutting down, the main thread will wait for all senders to be dropped
    let (shutdown_sender, shutdown_receiver) = mpsc::channel();

    // Shared channel to track shard status
    let (status_sender, status_receiver) = mpsc::channel();

    spawn_shards(
        num_shard,
        &args,
        shutdown.clone(),
        &shutdown_receiver,
        shutdown_sender,
        &status_receiver,
        status_sender,
        running.clone(),
    )?;

    // We might have received a termination signal
    if !running.load(Ordering::SeqCst) {
        shutdown_shards(shutdown, &shutdown_receiver);
        return Ok(());
    }

    let mut webserver = spawn_webserver(
        args,
        max_input_tokens,
        max_total_tokens,
        max_batch_prefill_tokens,
        shutdown.clone(),
        &shutdown_receiver,
    )
    .map_err(|err| {
        shutdown_shards(shutdown.clone(), &shutdown_receiver);
        err
    })?;

    // Default exit code
    let mut exit_code = Ok(());

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed(rank)) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} crashed");
            exit_code = Err(LauncherError::ShardFailed);
            break;
        };

        match webserver.try_wait().unwrap() {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, &shutdown_receiver);
                return Err(LauncherError::WebserverFailed);
            }
            None => {
                sleep(Duration::from_millis(100));
            }
        };
    }

    // Graceful termination
    terminate("webserver", webserver, Duration::from_secs(90)).unwrap();
    shutdown_shards(shutdown, &shutdown_receiver);

    exit_code
}
