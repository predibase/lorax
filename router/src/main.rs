/// LoRAX webserver entrypoint
use axum::http::{HeaderName, HeaderValue};
use clap::Parser;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use lorax_client::{ClientError, ShardedClient};
use lorax_router::{server, HubModelInfo, HubTokenizerConfig};
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use std::fs::File;
use std::io::BufReader;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::tokenizer::Tokenizer;
use tower_http::cors::{AllowCredentials, AllowHeaders, AllowMethods, AllowOrigin, ExposeHeaders};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "10", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "1792", long, env)]
    max_input_length: usize,
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,
    #[clap(default_value = "0.3", long, env)]
    waiting_served_ratio: f32,
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "128", long, env)]
    max_active_adapters: usize,
    #[clap(default_value = "2", long, env)]
    adapter_cycle_time_s: u64,
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/lorax-server-0", long, env)]
    master_shard_uds_path: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    embedding_model: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
    #[clap(long, env)]
    cors_allow_method: Option<Vec<String>>,
    #[clap(long, env)]
    cors_allow_header: Option<Vec<String>>,
    #[clap(long, env)]
    cors_expose_header: Option<Vec<String>>,
    #[clap(long, env)]
    cors_allow_credentials: Option<bool>,
    #[clap(long, env)]
    ngrok: bool,
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,
    #[clap(long, env)]
    ngrok_edge: Option<String>,
    #[clap(default_value = "hub", long, env)]
    adapter_source: String,
    #[clap(long, env)]
    eager_prefill: bool,
}

#[tokio::main]
async fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_active_adapters,
        adapter_cycle_time_s,
        hostname,
        port,
        master_shard_uds_path,
        tokenizer_name,
        tokenizer_config_path,
        revision,
        validation_workers,
        json_output,
        embedding_model,
        otlp_endpoint,
        cors_allow_origin,
        cors_allow_method,
        cors_allow_header,
        cors_expose_header,
        cors_allow_credentials,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        adapter_source,
        eager_prefill,
    } = args;

    init_logging(otlp_endpoint, json_output);

    // Validate args
    if max_input_length >= max_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_input_length` must be < `max_total_tokens`".to_string(),
        ));
    }
    if max_input_length as u32 > max_batch_prefill_tokens {
        return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be >= `max_input_length`. Given: {max_batch_prefill_tokens} and {max_input_length}")));
    }

    if validation_workers == 0 {
        return Err(RouterError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }

    if let Some(ref max_batch_total_tokens) = max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {max_batch_prefill_tokens} and {max_batch_total_tokens}")));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_batch_total_tokens}")));
        }
    }

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let cors_allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin
                .iter()
                .map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    // CORS allowed methods
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowMethods.
    let cors_allow_method: Option<AllowMethods> = cors_allow_method.map(|cors_allow_methods| {
        AllowMethods::list(
            cors_allow_methods
                .iter()
                .map(|method| method.parse::<axum::http::Method>().unwrap()),
        )
    });

    // CORS allowed headers
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowHeaders.
    let cors_allow_header: Option<AllowHeaders> = cors_allow_header.map(|cors_allow_headers| {
        AllowHeaders::list(
            cors_allow_headers
                .iter()
                .map(|header| header.parse::<HeaderName>().unwrap()),
        )
    });

    // CORS expose headers
    let cors_expose_header: Option<ExposeHeaders> = cors_expose_header.map(|cors_expose_headers| {
        ExposeHeaders::list(
            cors_expose_headers
                .iter()
                .map(|header| header.parse::<HeaderName>().unwrap()),
        )
    });

    // CORS allow credentials
    // Parse bool into AllowCredentials
    let cors_allow_credentials: Option<AllowCredentials> = cors_allow_credentials
        .map(|cors_allow_credentials| AllowCredentials::from(cors_allow_credentials));

    // Parse Huggingface hub token
    let authorization_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();

    // Tokenizer instance
    // This will only be used to validate payloads
    let local_path = Path::new(&tokenizer_name);

    // Shared API builder initialization
    let api_builder = || {
        let mut builder = ApiBuilder::new()
            .with_progress(false)
            .with_token(authorization_token);

        if let Ok(cache_dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        builder
    };

    // Decide if we need to use the API based on the revision and local path
    let use_api = revision.is_some() || !local_path.exists() || !local_path.is_dir();

    // Initialize API if needed
    #[derive(Clone)]
    enum Type {
        Api(Api),
        Cache(Cache),
        None,
    }

    let api = if use_api {
        if std::env::var("HF_HUB_OFFLINE") == Ok("1".to_string()) {
            let cache = Cache::default();
            tracing::warn!("Offline mode active using cache defaults");
            Type::Cache(cache)
        } else {
            tracing::info!("Using the Hugging Face API");
            match api_builder().build() {
                Ok(api) => Type::Api(api),
                Err(_) => {
                    tracing::warn!("Unable to build the Hugging Face API");
                    Type::None
                }
            }
        }
    } else {
        Type::None
    };

    // Load tokenizer and model info
    let (
        tokenizer_filename,
        _config_filename,
        tokenizer_config_filename,
        _preprocessor_config_filename,
        _processor_config_filename,
        model_info,
    ) = match api {
        Type::None => (
            Some(local_path.join("tokenizer.json")),
            Some(local_path.join("config.json")),
            Some(local_path.join("tokenizer_config.json")),
            Some(local_path.join("preprocessor_config.json")),
            Some(local_path.join("processor_config.json")),
            None,
        ),
        Type::Api(api) => {
            let api_repo = api.repo(Repo::with_revision(
                tokenizer_name.to_string(),
                RepoType::Model,
                revision.clone().unwrap_or_else(|| "main".to_string()),
            ));

            let tokenizer_filename = match api_repo.get("tokenizer.json").await {
                Ok(tokenizer_filename) => Some(tokenizer_filename),
                Err(_) => get_base_tokenizer(&api, &api_repo).await,
            };
            let config_filename = api_repo.get("config.json").await.ok();
            let tokenizer_config_filename = api_repo.get("tokenizer_config.json").await.ok();
            let preprocessor_config_filename = api_repo.get("preprocessor_config.json").await.ok();
            let processor_config_filename = api_repo.get("processor_config.json").await.ok();

            let model_info = if let Some(model_info) = get_model_info(&api_repo).await {
                Some(model_info)
            } else {
                tracing::warn!("Could not retrieve model info from the Hugging Face hub.");
                None
            };
            (
                tokenizer_filename,
                config_filename,
                tokenizer_config_filename,
                preprocessor_config_filename,
                processor_config_filename,
                model_info,
            )
        }
        Type::Cache(cache) => {
            let repo = cache.repo(Repo::with_revision(
                tokenizer_name.to_string(),
                RepoType::Model,
                revision.clone().unwrap_or_else(|| "main".to_string()),
            ));
            (
                repo.get("tokenizer.json"),
                repo.get("config.json"),
                repo.get("tokenizer_config.json"),
                repo.get("preprocessor_config.json"),
                repo.get("processor_config.json"),
                None,
            )
        }
    };
    let model_info = model_info.unwrap_or_else(|| HubModelInfo {
        model_id: tokenizer_name.to_string(),
        sha: None,
        pipeline_tag: None,
    });

    // Read the JSON contents of the file as an instance of 'HubTokenizerConfig'.
    let tokenizer_config: Option<HubTokenizerConfig> = if let Some(filename) = tokenizer_config_path
    {
        HubTokenizerConfig::from_file(filename)
    } else {
        tokenizer_config_filename.and_then(HubTokenizerConfig::from_file)
    };

    let tokenizer_config = tokenizer_config.unwrap_or_else(|| {
        tracing::warn!("Could not find tokenizer config locally and no API specified");
        HubTokenizerConfig::default()
    });

    // Load tokenizer
    let tokenizer: Option<Tokenizer> = tokenizer_filename.and_then(|filename| {
        let mut tokenizer = Tokenizer::from_file(filename).ok();
        if let Some(tokenizer) = &mut tokenizer {
            if let Some(class) = &tokenizer_config.tokenizer_class {
                if class == "LlamaTokenizer" || class == "LlamaTokenizerFast"{
                    if let Ok(post_processor) = create_post_processor(tokenizer, &tokenizer_config) {
                        tracing::info!("Overriding LlamaTokenizer with TemplateProcessing to follow python override defined in https://github.com/huggingface/transformers/blob/4aa17d00690b7f82c95bb2949ea57e22c35b4336/src/transformers/models/llama/tokenization_llama_fast.py#L203-L205");
                        tokenizer.with_post_processor(post_processor);
                    }
                }
            }
        }
        tokenizer
    });

    if tokenizer.is_none() {
        tracing::warn!("Could not find a fast tokenizer implementation for {tokenizer_name}");
        tracing::warn!("Rust input length validation and truncation is disabled");
    }

    // if pipeline-tag == lorax we default to return_full_text = true
    let compat_return_full_text = match &model_info.pipeline_tag {
        None => {
            tracing::warn!("no pipeline tag found for model {tokenizer_name}");
            false
        }
        Some(pipeline_tag) => pipeline_tag.as_str() == "lorax",
    };

    // Instantiate sharded client from the master unix socket
    let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
        .await
        .map_err(RouterError::Connection)?;
    // Clear the cache; useful if the webserver rebooted
    sharded_client
        .clear_cache(None)
        .await
        .map_err(RouterError::Cache)?;
    // Get info from the shard
    let shard_info = sharded_client.info().await.map_err(RouterError::Info)?;

    // Warmup model
    tracing::info!("Warming up model");
    let max_supported_batch_total_tokens = match sharded_client
        .warmup(
            max_input_length as u32,
            max_batch_prefill_tokens,
            max_total_tokens as u32,
        )
        .await
        .map_err(RouterError::Warmup)?
    {
        // Older models do not support automatic max-batch-total-tokens
        None => {
            let max_batch_total_tokens = max_batch_total_tokens
                .unwrap_or(16000.max((max_total_tokens as u32).max(max_batch_prefill_tokens)));
            tracing::warn!("Model does not support automatic max batch total tokens");
            max_batch_total_tokens
        }
        // Flash attention models return their max supported total tokens
        Some(max_supported_batch_total_tokens) => {
            // Warn if user added his own max-batch-total-tokens as we will ignore it
            let mut max_effective_batch_total_tokens = max_supported_batch_total_tokens;
            if max_batch_total_tokens.is_some() {
                // Check if manual value is lower than inferred value
                let max_batch_total_tokens = max_batch_total_tokens.unwrap();
                if max_batch_total_tokens < max_supported_batch_total_tokens {
                    max_effective_batch_total_tokens = max_batch_total_tokens;
                } else {
                    tracing::warn!(
                        "`--max-batch-total-tokens` is deprecated for Flash \
                    Attention models."
                    );
                    tracing::warn!(
                        "Inferred max batch total tokens: {max_supported_batch_total_tokens}"
                    );
                }
            }
            max_effective_batch_total_tokens
        }
    };
    tracing::info!("Setting max batch total tokens to {max_supported_batch_total_tokens}");
    tracing::info!("Connected");

    let addr = match hostname.parse() {
        Ok(ip) => SocketAddr::new(ip, port),
        Err(_) => {
            tracing::warn!("Invalid hostname, defaulting to 0.0.0.0");
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port)
        }
    };

    // Run server
    server::run(
        model_info,
        shard_info,
        compat_return_full_text,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_supported_batch_total_tokens,
        max_waiting_tokens,
        max_active_adapters,
        adapter_cycle_time_s,
        sharded_client,
        tokenizer,
        validation_workers,
        addr,
        cors_allow_origin,
        cors_allow_method,
        cors_allow_credentials,
        cors_allow_header,
        cors_expose_header,
        tokenizer_config,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        adapter_source,
        embedding_model,
        eager_prefill,
    )
    .await?;
    Ok(())
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
fn init_logging(otlp_endpoint: Option<String>, json_output: bool) {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = match json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "lorax.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            axum_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}

/// get model info from the Huggingface Hub
pub async fn get_model_info(api: &ApiRepo) -> Option<HubModelInfo> {
    let response = api.info_request().send().await.ok()?;

    if response.status().is_success() {
        let hub_model_info: HubModelInfo =
            serde_json::from_str(&response.text().await.ok()?).ok()?;
        if let Some(sha) = &hub_model_info.sha {
            tracing::info!(
                "Serving revision {sha} of model {}",
                hub_model_info.model_id
            );
        }
        Some(hub_model_info)
    } else {
        None
    }
}

/// get base tokenizer
pub async fn get_base_tokenizer(api: &Api, api_repo: &ApiRepo) -> Option<PathBuf> {
    let config_filename = api_repo.get("config.json").await.ok()?;

    // Open the file in read-only mode with buffer.
    let file = File::open(config_filename).ok()?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let config: serde_json::Value = serde_json::from_reader(reader).ok()?;

    if let Some(serde_json::Value::String(base_model_id)) = config.get("base_model_name_or_path") {
        let api_base_repo = api.repo(Repo::with_revision(
            base_model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        api_base_repo.get("tokenizer.json").await.ok()
    } else {
        None
    }
}

/// Create a post_processor for the LlamaTokenizer
pub fn create_post_processor(
    tokenizer: &Tokenizer,
    tokenizer_config: &HubTokenizerConfig,
) -> Result<TemplateProcessing, tokenizers::processors::template::TemplateProcessingBuilderError> {
    let add_bos_token = tokenizer_config.add_bos_token.unwrap_or(true);
    let add_eos_token = tokenizer_config.add_eos_token.unwrap_or(false);

    let bos_token = tokenizer_config.bos_token.as_ref();
    let eos_token = tokenizer_config.eos_token.as_ref();

    if add_bos_token && bos_token.is_none() {
        panic!("add_bos_token = true but bos_token is None");
    }

    if add_eos_token && eos_token.is_none() {
        panic!("add_eos_token = true but eos_token is None");
    }

    let mut single = Vec::new();
    let mut pair = Vec::new();
    let mut special_tokens = Vec::new();

    if add_bos_token {
        if let Some(bos) = bos_token {
            let bos_token_id = tokenizer
                .token_to_id(bos.as_str())
                .expect("Should have found the bos token id");
            special_tokens.push((bos.as_str(), bos_token_id));
            single.push(format!("{}:0", bos.as_str()));
            pair.push(format!("{}:0", bos.as_str()));
        }
    }

    single.push("$A:0".to_string());
    pair.push("$A:0".to_string());

    if add_eos_token {
        if let Some(eos) = eos_token {
            let eos_token_id = tokenizer
                .token_to_id(eos.as_str())
                .expect("Should have found the eos token id");
            special_tokens.push((eos.as_str(), eos_token_id));
            single.push(format!("{}:0", eos.as_str()));
            pair.push(format!("{}:0", eos.as_str()));
        }
    }

    if add_bos_token {
        if let Some(bos) = bos_token {
            pair.push(format!("{}:1", bos.as_str()));
        }
    }

    pair.push("$B:1".to_string());

    if add_eos_token {
        if let Some(eos) = eos_token {
            pair.push(format!("{}:1", eos.as_str()));
        }
    }

    let post_processor = TemplateProcessing::builder()
        .try_single(single)?
        .try_pair(pair)?
        .special_tokens(special_tokens)
        .build()?;

    Ok(post_processor)
}

#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Unable to connect to the Python model shards: {0}")]
    Connection(ClientError),
    #[error("Unable to clear the Python model shards cache: {0}")]
    Cache(ClientError),
    #[error("Unable to get the Python model shards info: {0}")]
    Info(ClientError),
    #[error("Unable to warmup the Python model shards: {0}")]
    Warmup(ClientError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
    #[error("Axum webserver failed: {0}")]
    Axum(#[from] axum::BoxError),
}
