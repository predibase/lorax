/// HTTP Server logic
use crate::adapter::{extract_adapter_params, BASE_MODEL_ADAPTER_ID};
use crate::config::Config;
use crate::health::Health;
use crate::infer::{InferError, InferResponse, InferStreamResponse};
use crate::validation::ValidationError;
use crate::{
    default_json_schema, AdapterParameters, AlternativeToken, BatchClassifyRequest, BestOfSequence,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionStreamResponse, ChatCompletionStreamResponseChoice, ChatMessage, ClassifyRequest,
    CompatGenerateRequest, CompletionFinishReason, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse, Details,
    EmbedRequest, EmbedResponse, Entity, ErrorResponse, FinishReason, GenerateParameters,
    GenerateRequest, GenerateResponse, HubModelInfo, Infer, Info, JsonSchema, LogProbs,
    OpenAiResponseFormat, PrefillToken, ResponseFormat, ResponseFormatType, SimpleToken,
    StreamDetails, StreamResponse, Token, TokenizeRequest, TokenizeResponse, UsageInfo, Validation,
};
use crate::{json, HubPreprocessorConfig, HubProcessorConfig, HubTokenizerConfig};
use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use futures::stream::StreamExt;
use futures::Stream;
use lorax_client::{ShardInfo, ShardedClient};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use once_cell::sync::OnceCell;
use reqwest_middleware::ClientBuilder;
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};
use tower_http::cors::{
    AllowCredentials, AllowHeaders, AllowMethods, AllowOrigin, CorsLayer, ExposeHeaders,
};
use tracing::{info_span, instrument, Instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

pub static DEFAULT_ADAPTER_SOURCE: OnceCell<String> = OnceCell::new();

/// Generate tokens if `stream == false` or a stream of token if `stream == true`
#[utoipa::path(
post,
tag = "LoRAX",
path = "/",
request_body = CompatGenerateRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = GenerateResponse),
("text/event-stream" = StreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn compat_generate(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<CompatGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;

    // default return_full_text given the pipeline_tag
    if req.parameters.return_full_text.is_none() {
        req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if req.stream {
        Ok(generate_stream(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(req.into()),
        )
        .await
        .into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(req.into()),
        )
        .await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(vec![generation.0])).into_response())
    }
}

const OPEN_AI_END_EVENT: &str = "[DONE]";

/// OpenAI compatible completions endpoint
#[utoipa::path(
post,
tag = "OpenAI Compatible",
path = "/v1/completions",
request_body = CompletionRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = CompletionResponse),
("text/event-stream" = CompletionStreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn completions_v1(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;
    if req.model == info.model_id.as_str() {
        // Allow user to specify the base model, but treat it as an empty adapter_id
        tracing::info!("Replacing base model {0} with empty adapter_id", req.model);
        req.model = "".to_string();
    }
    let mut gen_req = CompatGenerateRequest::from(req);

    // default return_full_text given the pipeline_tag
    if gen_req.parameters.return_full_text.is_none() {
        gen_req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if gen_req.stream {
        let callback = move |resp: StreamResponse| {
            Event::default()
                .json_data(CompletionStreamResponse::from(resp))
                .map_or_else(
                    |err| {
                        tracing::error!("Failed to serialize CompletionStreamResponse: {err}");
                        Event::default()
                    },
                    |data| data,
                )
        };

        let (headers, stream) = generate_stream_with_callback(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
            callback,
            Some(Event::default().data(OPEN_AI_END_EVENT)),
        )
        .await;
        Ok((headers, Sse::new(stream).keep_alive(KeepAlive::default())).into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
        )
        .await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(CompletionResponse::from(generation.0))).into_response())
    }
}

/// OpenAI compatible chat completions endpoint
#[utoipa::path(
post,
tag = "OpenAI Compatible",
path = "/v1/chat/completions",
request_body = ChatCompletionRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = ChatCompletionResponse),
("text/event-stream" = ChatCompletionStreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn chat_completions_v1(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let mut req = req.0;
    if req.model == info.model_id.as_str() {
        // Allow user to specify the base model, but treat it as an empty adapter_id
        tracing::info!("Replacing base model {0} with empty adapter_id", req.model);
        req.model = "".to_string();
    }

    // apply chat template to flatten the request into a single input
    let inputs = match infer.apply_chat_template(req.messages) {
        Ok(inputs) => inputs,
        Err(err) => {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: err.to_string(),
                    error_type: err.error_type().to_string(),
                }),
            ));
        }
    };

    let mut adapter_id = Some(req.model.clone());
    if req.model == info.model_id.as_str() {
        // Allow user to specify the base model, but treat it as an empty adapter_id
        tracing::debug!("Replacing base model {0} with empty adapter_id", req.model);
        adapter_id = None;
    }

    // Modify input values to ResponseFormat to be OpenAI API compatible
    let response_format: Option<ResponseFormat> = match req.response_format {
        None => None,
        Some(openai_format) => {
            let response_format_type = openai_format.response_format_type.clone();
            match response_format_type {
                // Ignore when type is text
                ResponseFormatType::Text => None,

                // For json_object, use the fixed schema
                ResponseFormatType::JsonObject => Some(ResponseFormat {
                    r#type: response_format_type.clone(),
                    schema: default_json_schema(),
                }),

                // For json_schema, use schema_value if available, otherwise fallback to the fixed schema
                ResponseFormatType::JsonSchema => openai_format
                    .json_schema
                    .and_then(|schema| schema.schema)
                    .map_or_else(
                        || {
                            Some(ResponseFormat {
                                r#type: response_format_type.clone(),
                                schema: default_json_schema(),
                            })
                        },
                        |schema_value: serde_json::Value| {
                            Some(ResponseFormat {
                                r#type: response_format_type.clone(),
                                schema: Some(schema_value),
                            })
                        },
                    ),
            }
        }
    };

    let mut gen_req = CompatGenerateRequest {
        inputs: inputs.to_string(),
        parameters: GenerateParameters {
            adapter_id: adapter_id,
            adapter_source: req.adapter_source,
            adapter_parameters: None,
            api_token: req.api_token,
            best_of: req.n.map(|x| x as usize),
            temperature: req.temperature,
            repetition_penalty: req.repetition_penalty,
            top_k: req.top_k,
            top_p: req.top_p,
            typical_p: None,
            do_sample: !req.n.is_none(),
            max_new_tokens: req.max_tokens.map(|x| x as u32),
            ignore_eos_token: req.ignore_eos_token.unwrap_or(false),
            return_full_text: None,
            stop: req.stop,
            truncate: None,
            watermark: false,
            details: true,
            decoder_input_details: false,
            return_k_alternatives: None,
            apply_chat_template: false,
            seed: req.seed,
            response_format: response_format,
        },
        stream: req.stream.unwrap_or(false),
    };

    // default return_full_text given the pipeline_tag
    if gen_req.parameters.return_full_text.is_none() {
        gen_req.parameters.return_full_text = Some(default_return_full_text.0)
    }

    // switch on stream
    if gen_req.stream {
        let callback = move |resp: StreamResponse| {
            Event::default()
                .json_data(ChatCompletionStreamResponse::from(resp))
                .map_or_else(
                    |err| {
                        tracing::error!("Failed to serialize ChatCompletionStreamResponse: {err}");
                        Event::default()
                    },
                    |data| data,
                )
        };

        let (headers, stream) = generate_stream_with_callback(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
            callback,
            Some(Event::default().data(OPEN_AI_END_EVENT)),
        )
        .await;
        Ok((headers, Sse::new(stream).keep_alive(KeepAlive::default())).into_response())
    } else {
        let (headers, generation) = generate(
            infer,
            info,
            request_logger_sender,
            req_headers,
            Json(gen_req.into()),
        )
        .await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(ChatCompletionResponse::from(generation.0))).into_response())
    }
}

/// LoRAX endpoint info
#[utoipa::path(
get,
tag = "LoRAX",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "LoRAX",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "LoRAX is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
/// Health check method
async fn health(
    infer: Extension<Infer>,
    health: Extension<Health>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if health.shard_info().supports_classification {
        let classify_request = ClassifyRequest {
            inputs: "San Francisco".to_string(),
        };
        match infer.classify(classify_request).await {
            Ok(_) => {}
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    if health.shard_info().supports_embeddings {
        let embed_request = EmbedRequest {
            inputs: "San Francisco".to_string(),
        };
        match infer.embed(embed_request).await {
            Ok(_) => {}
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    if health.shard_info().supports_generation {
        let generate_request = GenerateRequest {
            inputs: "Who?".to_string(),
            parameters: GenerateParameters {
                adapter_id: None,
                adapter_source: None,
                adapter_parameters: None,
                api_token: None,
                best_of: None,
                temperature: None,
                top_k: None,
                top_p: None,
                typical_p: None,
                do_sample: false,
                seed: None,
                repetition_penalty: None,
                watermark: false,
                return_full_text: None,
                stop: vec![],
                truncate: None,
                details: false,
                decoder_input_details: false,
                return_k_alternatives: None,
                apply_chat_template: false,
                response_format: None,
                max_new_tokens: Some(1),
                ignore_eos_token: false,
            },
        };
        match infer.generate(generate_request).await {
            Ok(response) => {
                if response.generated_text.text.len() == 0 {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Empty generation".to_string(),
                            error_type: "failed healthcheck".to_string(),
                        }),
                    ));
                }
            }
            Err(error) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: error.to_string(),
                        error_type: error.error_type().to_string(),
                    }),
                ));
            }
        }
    }
    Ok(())
}

/// Generate tokens
#[utoipa::path(
post,
tag = "LoRAX",
path = "/generate",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = GenerateResponse),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.0.parameters,
total_time,
validation_time,
queue_time,
inference_time,
time_per_token,
seed,
)
)]
async fn generate(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    mut req: Json<GenerateRequest>,
) -> Result<(HeaderMap, Json<GenerateResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");

    tracing::debug!("Input: {}", req.0.inputs);

    let compute_characters = req.0.inputs.chars().count();
    let mut add_prompt = None;
    if req.0.parameters.return_full_text.unwrap_or(false) {
        add_prompt = Some(req.0.inputs.clone());
    }

    let inputs = req.0.inputs.clone();

    let details = req.0.parameters.details || req.0.parameters.decoder_input_details;
    let (adapter_source, adapter_parameters) = extract_adapter_params(
        req.0.parameters.adapter_id.clone(),
        req.0.parameters.adapter_source.clone(),
        req.0.parameters.adapter_parameters.clone(),
    );

    if req.parameters.api_token.is_none() {
        // If no API token was explicitly provided in the request payload, try to set it from the request headers.
        let _ = req_headers.get("authorization").map_or((), |x| {
            x.to_str().map_or((), |y| {
                y.strip_prefix("Bearer ").map_or((), |token| {
                    req.parameters.api_token = Some(token.to_string());
                })
            })
        });
    }

    let api_token = req.parameters.api_token.clone();

    // Inference
    let (response, best_of_responses) = match req.0.parameters.best_of {
        Some(best_of) if best_of > 1 => {
            let (response, best_of_responses) = infer.generate_best_of(req.0, best_of).await?;
            (response, Some(best_of_responses))
        }
        _ => (infer.generate(req.0).await?, None),
    };

    let generated_tokens = response.generated_text.generated_tokens;
    let prompt_tokens = response.prompt_tokens;
    let total_tokens = prompt_tokens + generated_tokens;

    // Token details
    let details = match details {
        true => {
            // convert best_of_responses
            let best_of_sequences = best_of_responses.map(|responses: Vec<InferResponse>| {
                responses
                    .into_iter()
                    .map(|response: InferResponse| {
                        // Add prompt if return_full_text
                        let mut output_text = response.generated_text.text;
                        if let Some(prompt) = &add_prompt {
                            output_text = prompt.clone() + &output_text;
                        }

                        BestOfSequence {
                            generated_text: output_text,
                            finish_reason: FinishReason::from(
                                response.generated_text.finish_reason,
                            ),
                            generated_tokens: response.generated_text.generated_tokens,
                            prefill: response.prefill,
                            tokens: response.tokens,
                            seed: response.generated_text.seed,
                        }
                    })
                    .collect()
            });

            Some(Details {
                finish_reason: FinishReason::from(response.generated_text.finish_reason),
                prompt_tokens: prompt_tokens,
                generated_tokens: generated_tokens,
                prefill: response.prefill,
                tokens: response.tokens,
                seed: response.generated_text.seed,
                best_of_sequences,
            })
        }
        false => None,
    };

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;
    let time_per_token = inference_time / response.generated_text.generated_tokens;

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));
    span.record("time_per_token", format!("{time_per_token:?}"));
    span.record("seed", format!("{:?}", response.generated_text.seed));
    span.record("prompt_tokens", format!("{prompt_tokens:?}"));
    span.record("generated_tokens", format!("{generated_tokens:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-prompt-tokens",
        prompt_tokens.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-generated-tokens",
        generated_tokens.to_string().parse().unwrap(),
    );
    headers.insert("x-total-tokens", total_tokens.to_string().parse().unwrap());
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-token",
        time_per_token.as_millis().to_string().parse().unwrap(),
    );

    headers.insert("x-model-id", info.model_id.parse().unwrap());

    let adapter_id_string = adapter_parameters
        .adapter_ids
        .iter()
        .map(|id| id.as_str())
        // filter out base model adapter id
        .filter(|id| *id != BASE_MODEL_ADAPTER_ID)
        .collect::<Vec<_>>()
        .join(",");

    if adapter_id_string.len() > 0 {
        headers.insert("x-adapter-id", adapter_id_string.parse().unwrap());
        headers.insert("x-adapter-source", adapter_source.unwrap().parse().unwrap());
    }

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );
    metrics::histogram!(
        "lorax_request_mean_time_per_token_duration",
        time_per_token.as_secs_f64()
    );
    metrics::histogram!(
        "lorax_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    if info.request_logger_url.is_some() {
        let _ = request_logger_sender
            .send((
                total_tokens as i64,
                adapter_id_string,
                inputs,
                response.generated_text.text.clone(),
                api_token.unwrap_or("".to_string()),
                info.model_id.clone(),
            ))
            .await;
    }

    // Send response
    let mut output_text = response.generated_text.text;
    if let Some(prompt) = add_prompt {
        output_text = prompt + &output_text;
    }

    tracing::debug!("Output: {}", output_text);
    tracing::info!("Success");

    let response = GenerateResponse {
        generated_text: output_text,
        details,
    };
    Ok((headers, Json(response)))
}

/// Generate a stream of token using Server-Sent Events
#[utoipa::path(
post,
tag = "LoRAX",
path = "/generate_stream",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = StreamResponse,
content_type = "text/event-stream"),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"}),
content_type = "text/event-stream"),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"}),
content_type = "text/event-stream"),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"}),
content_type = "text/event-stream"),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"}),
content_type = "text/event-stream"),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.0.parameters,
total_time,
validation_time,
queue_time,
inference_time,
time_per_token,
seed,
)
)]
async fn generate_stream(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    req: Json<GenerateRequest>,
) -> (
    HeaderMap,
    Sse<impl Stream<Item = Result<Event, Infallible>>>,
) {
    let callback = |resp: StreamResponse| Event::default().json_data(resp).unwrap();
    let (headers, stream) = generate_stream_with_callback(
        infer,
        info,
        request_logger_sender,
        req_headers,
        req,
        callback,
        None,
    )
    .await;
    (headers, Sse::new(stream).keep_alive(KeepAlive::default()))
}
async fn generate_stream_with_callback(
    infer: Extension<Infer>,
    info: Extension<Info>,
    request_logger_sender: Extension<
        Arc<mpsc::Sender<(i64, String, String, String, String, String)>>,
    >,
    req_headers: HeaderMap,
    mut req: Json<GenerateRequest>,
    callback: impl Fn(StreamResponse) -> Event,
    end_event: Option<Event>,
) -> (HeaderMap, impl Stream<Item = Result<Event, Infallible>>) {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");

    tracing::debug!("Input: {}", req.0.inputs);

    let compute_characters = req.0.inputs.chars().count();

    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());

    if req.parameters.api_token.is_none() {
        // If no API token was explicitly provided in the request payload, try to set it from the request headers.
        let _ = req_headers.get("authorization").map_or((), |x| {
            x.to_str().map_or((), |y| {
                y.strip_prefix("Bearer ").map_or((), |token| {
                    req.parameters.api_token = Some(token.to_string());
                })
            })
        });
    }

    let api_token = req.parameters.api_token.clone();

    let (adapter_source, adapter_parameters) = extract_adapter_params(
        req.0.parameters.adapter_id.clone(),
        req.0.parameters.adapter_source.clone(),
        req.0.parameters.adapter_parameters.clone(),
    );

    let adapter_id_string = adapter_parameters
        .adapter_ids
        .iter()
        .map(|id| id.as_str())
        // filter out base model adapter id
        .filter(|id| *id != BASE_MODEL_ADAPTER_ID)
        .collect::<Vec<_>>()
        .join(",");

    if adapter_id_string.len() > 0 {
        headers.insert("x-adapter-id", adapter_id_string.parse().unwrap());
        headers.insert("x-adapter-source", adapter_source.unwrap().parse().unwrap());
    }

    headers.insert("x-model-id", info.model_id.parse().unwrap());

    let stream = async_stream::stream! {
            // Inference
            let mut end_reached = false;
            let mut error = false;

            let mut prefill_tokens_length = 0;

            let mut add_prompt = None;
            if req.0.parameters.return_full_text.unwrap_or(false) {
                add_prompt = Some(req.0.inputs.clone());
            }
            let inputs = req.0.inputs.clone();
            let details = req.0.parameters.details;

            let best_of = req.0.parameters.best_of.unwrap_or(1);
            if best_of != 1 {
                let err = InferError::from(ValidationError::BestOfStream);
                metrics::increment_counter!("lorax_request_failure", "err" => "validation");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            } else if req.0.parameters.decoder_input_details {
                let err = InferError::from(ValidationError::PrefillDetailsStream);
                metrics::increment_counter!("lorax_request_failure", "err" => "validation");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            } else {
                match infer.generate_stream(req.0).instrument(info_span!(parent: &span, "async_stream")).await {
                    // Keep permit as long as generate_stream lives
                    Ok((_permit, mut response_stream)) => {
                        // Server-Sent Event stream
                        while let Some(response) = response_stream.next().await {
                            match response {
                                Ok(response) => {
                                    match response {
                                        // Prefill is ignored
                                        InferStreamResponse::Prefill {
                                            tokens_length,
                                            ..
                                        } => {
                                            prefill_tokens_length = tokens_length;
                                        }
                                        // Yield event for every new token
                                        InferStreamResponse::Token(token) => {
                                            tracing::debug!(parent: &span, "Token: {:?}", token);

                                            // StreamResponse
                                            let stream_token = StreamResponse {
                                                token,
                                                generated_text: None,
                                                details: None,
                                            };

                                            yield Ok(callback(stream_token))
                                        }
                                        // Yield event for last token and compute timings
                                        InferStreamResponse::End {
                                            token,
                                            generated_text,
                                            start,
                                            queued,
                                        } => {
                                            // Token details
                                            let details = match details {
                                                true => Some(StreamDetails {
                                                    finish_reason: FinishReason::from(generated_text.finish_reason),
                                                    prompt_tokens: prefill_tokens_length,
                                                    generated_tokens: generated_text.generated_tokens,
                                                    seed: generated_text.seed,
                                                }),
                                                false => None,
                                            };

                                            // Timings
                                            let total_time = start_time.elapsed();
                                            let validation_time = queued - start_time;
                                            let queue_time = start - queued;
                                            let inference_time = Instant::now() - start;
                                            let time_per_token = inference_time / generated_text.generated_tokens;

                                            // Tracing metadata
                                            span.record("total_time", format!("{total_time:?}"));
                                            span.record("validation_time", format!("{validation_time:?}"));
                                            span.record("queue_time", format!("{queue_time:?}"));
                                            span.record("inference_time", format!("{inference_time:?}"));
                                            span.record("time_per_token", format!("{time_per_token:?}"));
                                            span.record("seed", format!("{:?}", generated_text.seed));
                                            span.record("prompt_tokens",  format!("{prefill_tokens_length:?}"));
                                            span.record("generated_tokens",  format!("{:?}", generated_text.generated_tokens));

                                            // Metrics
                                            metrics::increment_counter!("lorax_request_success");
                                            metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_validation_duration", validation_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_inference_duration", inference_time.as_secs_f64());
                                            metrics::histogram!("lorax_request_mean_time_per_token_duration", time_per_token.as_secs_f64());
                                            metrics::histogram!("lorax_request_generated_tokens", generated_text.generated_tokens as f64);



                                            // StreamResponse
                                            end_reached = true;

                                            let mut output_text = generated_text.text;
                                            if let Some(prompt) = add_prompt {
                                                output_text = prompt + &output_text;
                                            }

                                            tracing::debug!(parent: &span, "Output: {}", output_text);
                                            tracing::info!(parent: &span, "Success");

                                            let total_tokens = generated_text.generated_tokens + prefill_tokens_length;
                                            if info.request_logger_url.is_some() {
                                                let _ = request_logger_sender.send((
                                                    total_tokens as i64,
                                                    adapter_id_string,
                                                    inputs,
                                                    output_text.clone(),
                                                    api_token.unwrap_or("".to_string()),
                                                    info.model_id.clone(),
                                                ))
                                                .await;
                                            }

                                            let stream_token = StreamResponse {
                                                token,
                                                generated_text: Some(output_text),
                                                details
                                            };

                                            yield Ok(callback(stream_token));
                                            if let Some(end_event) = end_event {
                                                yield Ok(end_event);
                                            }
                                            break;
                                        },
                                        InferStreamResponse::Embed {
                                            ..
                                        } => {
                                            let err = InferError::from(ValidationError::EmbeddingModel);
                                            metrics::increment_counter!("lorax_request_failure", "err" => "bad_request");
                                            tracing::error!("{err}");
                                            yield Ok(Event::from(err));
                                            break;
                                        }
                                        InferStreamResponse::Classify {
                                            ..
                                        } => {
                                            let err = InferError::from(ValidationError::ClassifyModelError);
                                            metrics::increment_counter!("lorax_request_failure", "err" => "bad_request");
                                            tracing::error!("{err}");
                                            yield Ok(Event::from(err));
                                            break;
                                        }
                                    }
                                }
                                // yield error
                                Err(err) => {
                                    error = true;
                                    yield Ok(Event::from(err));
                                    break;
                                }
                            }
                        }
                    },
                    // yield error
                    Err(err) => {
                        error = true;
                        yield Ok(Event::from(err));
                    }
                }
                // Check if generation reached the end
                // Skip if we already sent an error
                if !end_reached && !error {
                    let err = InferError::IncompleteGeneration;
                    metrics::increment_counter!("lorax_request_failure", "err" => "incomplete");
                    tracing::error!("{err}");
                    yield Ok(Event::from(err));
                }
            }
    };

    (headers, stream)
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "LoRAX",
path = "/metrics",
responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

async fn request_logger(
    request_logger_url: Option<String>,
    mut rx: mpsc::Receiver<(i64, String, String, String, String, String)>,
) {
    if request_logger_url.is_none() {
        tracing::info!("REQUEST_LOGGER_URL not set, request logging is disabled");
        return;
    }

    let url_string = request_logger_url.unwrap();
    tracing::info!("Request logging enabled, sending logs to {url_string}");

    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
    let client = ClientBuilder::new(reqwest::Client::new())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build();
    while let Some((tokens, adapter_id, input, output, api_token, model_id)) = rx.recv().await {
        // Make a request out to localhost:8899 with the tokens, api_token, and model_id
        let res = client
            .post(&url_string)
            .json(&json!({
                "tokens": tokens,
                "adapter_id": adapter_id,
                "input": input,
                "output": output,
                "api_token": api_token,
                "model_id": model_id
            }))
            .send()
            .await;

        if let Err(e) = res {
            tracing::error!("Failed to log request: {e}");
        }
    }
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: ShardInfo,
    compat_return_full_text: bool,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_active_adapters: usize,
    adapter_cycle_time_s: u64,
    client: ShardedClient,
    config: Option<Config>,
    tokenizer: Option<Tokenizer>,
    (preprocessor_config, _processor_config): (Option<HubPreprocessorConfig>, HubProcessorConfig),
    validation_workers: usize,
    addr: SocketAddr,
    cors_allow_origin: Option<AllowOrigin>, // exact match
    cors_allow_methods: Option<AllowMethods>,
    cors_allow_credentials: Option<AllowCredentials>,
    cors_allow_headers: Option<AllowHeaders>,
    cors_expose_headers: Option<ExposeHeaders>,
    tokenizer_config: HubTokenizerConfig,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    adapter_source: String,
    eager_prefill: bool,
    prefix_caching: bool,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    health,
    get_model_info,
    compat_generate,
    generate,
    generate_stream,
    completions_v1,
    chat_completions_v1,
    tokenize,
    metrics,
    ),
    components(
    schemas(
    Info,
    UsageInfo,
    ResponseFormat,
    ResponseFormatType,
    OpenAiResponseFormat,
    JsonSchema,
    CompatGenerateRequest,
    GenerateRequest,
    GenerateParameters,
    AdapterParameters,
    AlternativeToken,
    PrefillToken,
    Token,
    SimpleToken,
    TokenizeRequest,
    TokenizeResponse,
    GenerateResponse,
    BestOfSequence,
    Details,
    FinishReason,
    StreamResponse,
    StreamDetails,
    ErrorResponse,
    ChatMessage,
    LogProbs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    CompletionResponseStreamChoice,
    CompletionFinishReason,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    )
    ),
    tags(
    (name = "LoRAX", description = "LoRAX API"),
    (name = "OpenAI Compatible", description = "OpenAI compatible API"),
    (name = "Tokenization", description = "Tokenizer API"),
    ),
    info(
    title = "LoRAX",
    license(
    name = "Apache 2.0",
    url = "https://www.apache.org/licenses/LICENSE-2.0"
    )
    )
    )]
    struct ApiDoc;

    let cloned_tokenizer = tokenizer.clone().map(|t| Arc::new(Mutex::new(t)));
    let arc_tokenizer = tokenizer.clone().map(Arc::new);

    // Create state
    let validation = Validation::new(
        validation_workers,
        tokenizer,
        config,
        preprocessor_config,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
    );
    let generation_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new(
        client.clone(),
        generation_health.clone(),
        shard_info.clone(),
    );

    // For non-causal LMs, the max batch total tokens is equal to the max batch prefill tokens
    let is_causal_lm = shard_info.supports_generation;
    let effective_max_batch_total_tokens = if is_causal_lm {
        max_batch_total_tokens
    } else {
        max_batch_prefill_tokens
    };

    let infer = Infer::new(
        client.clone(),
        validation,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        effective_max_batch_total_tokens,
        max_waiting_tokens,
        max_concurrent_requests,
        max_active_adapters,
        adapter_cycle_time_s,
        shard_info.requires_padding,
        shard_info.window_size,
        generation_health,
        eager_prefill,
        tokenizer_config,
        arc_tokenizer,
        shard_info.block_size,
        shard_info.speculate,
        shard_info.preloaded_adapters,
        prefix_caching,
        is_causal_lm,
    );

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("lorax_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("lorax_request_generated_tokens"));
    let generated_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Input Length buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("lorax_request_max_new_tokens"));
    let max_new_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("lorax_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &generated_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // CORS layer
    let cors_allow_origin = cors_allow_origin.unwrap_or(AllowOrigin::any());
    // unwrap allow methods with default get and post
    let cors_allow_methods =
        cors_allow_methods.unwrap_or(AllowMethods::list(vec![Method::GET, Method::POST]));

    let cors_allow_headers =
        cors_allow_headers.unwrap_or(AllowHeaders::list(vec![http::header::CONTENT_TYPE]));

    let cors_expose_headers = cors_expose_headers.unwrap_or(ExposeHeaders::default());
    let cors_allow_credentials = cors_allow_credentials.unwrap_or(AllowCredentials::default());

    // log cors stuff
    tracing::info!(
        "CORS: origin: {cors_allow_origin:?}, methods: {cors_allow_methods:?}, headers: {cors_allow_headers:?}, expose-headers: {cors_expose_headers:?} credentials: {cors_allow_credentials:?}",
    );

    let cors_layer = CorsLayer::new()
        .allow_methods(cors_allow_methods)
        .allow_headers(cors_allow_headers)
        .allow_credentials(cors_allow_credentials)
        .expose_headers(cors_expose_headers)
        .allow_origin(cors_allow_origin);

    // log all the cors layer
    tracing::info!("CORS: {cors_layer:?}");

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_total_tokens,
        max_waiting_tokens,
        validation_workers,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
        request_logger_url: std::env::var("REQUEST_LOGGER_URL").ok(),
        eager_prefill,
    };

    DEFAULT_ADAPTER_SOURCE
        .set(adapter_source.clone())
        .unwrap_or_else(|_| {
            panic!("DEFAULT_ADAPTER_SOURCE was already set!");
        });

    // Kick off thread here that writes to the log file
    let (tx, rx) = mpsc::channel(32);
    let request_logger_sender = Arc::new(tx);
    if info.request_logger_url.is_some() {
        tokio::spawn(request_logger(info.request_logger_url.clone(), rx));
    } else {
        tracing::info!("REQUEST_LOGGER_URL not set, request logging is disabled");
    }

    // Create router
    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        // Base routes
        .route("/", post(compat_generate))
        .route("/info", get(get_model_info))
        .route("/generate", post(generate))
        .route("/embed", post(embed))
        .route("/classify", post(classify))
        .route("/classify_batch", post(classify_batch))
        .route("/generate_stream", post(generate_stream))
        .route("/v1/completions", post(completions_v1))
        .route("/v1/chat/completions", post(chat_completions_v1))
        // AWS Sagemaker route
        .route("/invocations", post(compat_generate))
        // Base Health route
        .route("/health", get(health))
        // Inference API health route
        .route("/", get(health))
        // AWS Sagemaker health route
        .route("/ping", get(health))
        // Prometheus metrics route
        .route("/metrics", get(metrics))
        .route("/tokenize", post(tokenize))
        .layer(Extension(info))
        .layer(Extension(client.clone()))
        .layer(Extension(request_logger_sender.clone()))
        .layer(Extension(health_ext.clone()))
        .layer(Extension(compat_return_full_text))
        .layer(Extension(infer))
        .layer(Extension(prom_handle.clone()))
        .layer(opentelemetry_tracing_layer())
        .layer(cors_layer)
        .layer(Extension(cloned_tokenizer));

    if ngrok {
        #[cfg(feature = "ngrok")]
        {
            use ngrok::config::TunnelBuilder;

            let _ = addr;

            let authtoken =
                ngrok_authtoken.expect("`ngrok-authtoken` must be set when using ngrok tunneling");

            let edge = ngrok_edge.expect("`ngrok-edge` must be set when using ngrok tunneling");

            let tunnel = ngrok::Session::builder()
                .authtoken(authtoken)
                .connect()
                .await
                .unwrap()
                .labeled_tunnel()
                .label("edge", edge);

            let listener = tunnel.listen().await.unwrap();

            // Run prom metrics and health locally too
            tokio::spawn(
                axum::Server::bind(&addr)
                    .serve(
                        Router::new()
                            .route("/health", get(health))
                            .route("/metrics", get(metrics))
                            .layer(Extension(health_ext))
                            .layer(Extension(prom_handle))
                            .into_make_service(),
                    )
                    //Wait until all requests are finished to shut down
                    .with_graceful_shutdown(shutdown_signal()),
            );

            // Run server
            axum::Server::builder(listener)
                .serve(app.into_make_service())
                //Wait until all requests are finished to shut down
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        #[cfg(not(feature = "ngrok"))]
        {
            let _ngrok_authtoken = ngrok_authtoken;
            let _ngrok_domain = ngrok_domain;
            let _ngrok_username = ngrok_username;
            let _ngrok_password = ngrok_password;

            panic!("`lorax-router` was compiled without the `ngrok` feature");
        }
    } else {
        // Run server
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            // Wait until all requests are finished to shut down
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }
    Ok(())
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

impl From<i32> for FinishReason {
    fn from(finish_reason: i32) -> Self {
        let finish_reason = lorax_client::FinishReason::from_i32(finish_reason).unwrap();
        match finish_reason {
            lorax_client::FinishReason::Length => FinishReason::Length,
            lorax_client::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            lorax_client::FinishReason::StopSequence => FinishReason::StopSequence,
        }
    }
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::GenerationError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::IncompleteGeneration => StatusCode::INTERNAL_SERVER_ERROR,
            InferError::TemplateError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::EmbeddingFailure => StatusCode::INTERNAL_SERVER_ERROR,
            InferError::ClassificationFailure => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}

/// Embed inputs
#[utoipa::path(
    post,
    tag = "Embedding",
    path = "/embed",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Embeddings ids", body = EmbedResponse),
    (status = 500, description = "Incomplete embedding", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn embed(
    infer: Extension<Infer>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Input: {}", req.inputs);
    // Inference
    let response = infer.embed(req).await?;
    Ok(Json(response))
}

#[utoipa::path(
    post,
    tag = "Classify",
    path = "/classify",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Classifications", body = ClassifyResponse),
    (status = 500, description = "Incomplete classification", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn classify(
    infer: Extension<Infer>,
    Json(req): Json<ClassifyRequest>,
) -> Result<(HeaderMap, Json<Vec<Entity>>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Input: {}", req.inputs);
    let response = infer.classify(req).await?;

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );

    Ok((headers, Json(response.predictions)))
}

#[utoipa::path(
    post,
    tag = "ClassifyBatch",
    path = "/classify_batch",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Classifications", body = BatchClassifyResponse),
    (status = 500, description = "Incomplete classification", body = ErrorResponse),
    )
)]
#[instrument(skip_all)]
async fn classify_batch(
    infer: Extension<Infer>,
    Json(req): Json<BatchClassifyRequest>,
) -> Result<(HeaderMap, Json<Vec<Vec<Entity>>>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("lorax_request_count");
    tracing::debug!("Inputs: {:?}", req.inputs);
    let num_inputs = req.inputs.len();
    let responses = infer.classify_batch(req).await?;

    // Timings
    let now = Instant::now();
    let total_time = start_time.elapsed();
    let mut validation_times = Vec::with_capacity(responses.len());
    let mut queue_times = Vec::with_capacity(responses.len());
    let mut inference_times = Vec::with_capacity(responses.len());

    for r in &responses {
        validation_times.push(r.queued - r.start);
        queue_times.push(r.start - r.queued);
        inference_times.push(now - r.start);
    }

    let validation_time = validation_times.iter().sum::<Duration>() / responses.len() as u32;
    let queue_time = queue_times.iter().sum::<Duration>() / responses.len() as u32;
    let inference_time = inference_times.iter().sum::<Duration>() / responses.len() as u32;

    // Rust Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("num_inputs", num_inputs);
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("lorax_request_success");
    metrics::histogram!("lorax_request_duration", total_time.as_secs_f64());
    metrics::histogram!("lorax_request_input_count", num_inputs as f64);
    metrics::histogram!(
        "lorax_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("lorax_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "lorax_request_inference_duration",
        inference_time.as_secs_f64()
    );

    let batch_entity_vec = responses.into_iter().map(|r| r.predictions).collect();
    Ok((headers, Json(batch_entity_vec)))
}

/// Tokenize inputs
#[utoipa::path(
    post,
    tag = "Tokenization",
    path = "/tokenize",
    request_body = TokenizeRequest,
    responses(
    (status = 200, description = "Tokenized ids", body = TokenizeResponse),
    (status = 404, description = "No tokenizer found", body = ErrorResponse,
    example = json ! ({"error": "No fast tokenizer available"})),
    )
    )]
#[instrument(skip_all)]
async fn tokenize(
    Extension(cloned_tokenizer): Extension<Option<Arc<Mutex<Tokenizer>>>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(tokenizer) = cloned_tokenizer {
        let input = req.inputs.clone();
        let add_special_tokens = match req.add_special_tokens {
            None => true,
            _ => req.add_special_tokens.unwrap(),
        };
        let tokenizer = tokenizer.lock().unwrap();
        let char_offset = tokenizer
            .encode_char_offsets(&input[..], add_special_tokens)
            .unwrap();
        let tokens: Vec<SimpleToken> = char_offset
            .get_ids()
            .iter()
            .zip(char_offset.get_offsets().iter())
            .map(|(&id, &(start, stop))| {
                let text: String = tokenizer.id_to_token(id).unwrap();
                SimpleToken {
                    id,
                    text,
                    start,
                    stop,
                }
            })
            .collect();
        Ok(Json(TokenizeResponse(tokens)))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No fast tokenizer or tokenizer.json for this model".to_string(),
                error_type: "no fast tokenizer".to_string(),
            }),
        ))
    }
}
