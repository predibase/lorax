use crate::adapter::{extract_adapter_params, Adapter, BASE_MODEL_ADAPTER_ID};
use crate::batch::{ValidClassifyRequest, ValidEmbedRequest};
use crate::queue::AdapterEvent;
use crate::scheduler::AdapterScheduler;
use crate::validation::{Validation, ValidationError};
use crate::{
    AdapterParameters, AlternativeToken, BatchClassifyRequest, ChatTemplateVersions,
    ClassifyRequest, EmbedRequest, EmbedResponse, Entity, Entry, HubTokenizerConfig, Message,
    MessageChunk, TextMessage, Token, TokenizerConfigToken, Tool,
};
use crate::{GenerateRequest, PrefillToken};
use futures::future::try_join_all;
use futures::stream::StreamExt;
/// Batching and inference logic
use itertools::izip;
use itertools::multizip;
use lorax_client::{
    Batch, CachedBatch, ClassifyPredictionList, ClientError, Embedding, GeneratedText, Generation,
    PrefillTokens, PreloadedAdapter, ShardedClient,
};
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;
use nohash_hasher::IntMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Mutex, Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info_span, instrument, Span};

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct ChatTemplateInputs<'a> {
    messages: Vec<TextMessage>,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    add_generation_prompt: bool,
    tools: Option<Vec<Tool>>,
    guideline: Option<&'a str>,
}

/// Raise a exception (custom function) used in the chat templates
fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

#[derive(Clone)]
struct ChatTemplateRenderer {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    use_default_tool_template: bool,
    variables: HashSet<String>,
}

impl ChatTemplateRenderer {
    fn new(
        template: String,
        bos_token: Option<TokenizerConfigToken>,
        eos_token: Option<TokenizerConfigToken>,
    ) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);
        tracing::debug!("Loading template: {}", template_str);

        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        // get the list of variables that are used in the template
        let variables = template.undeclared_variables(true);

        // check if the `tools` variable is used in the template
        let use_default_tool_template = !variables.contains("tools");
        tracing::debug!("Use default tool template: {}", use_default_tool_template);

        Self {
            template,
            bos_token: bos_token.map(|token| token.as_str().to_string()),
            eos_token: eos_token.map(|token| token.as_str().to_string()),
            use_default_tool_template,
            variables,
        }
    }

    fn apply(
        &self,
        guideline: Option<&str>,
        mut messages: Vec<Message>,
        tools_and_prompt: Option<(Vec<Tool>, String)>,
    ) -> Result<String, InferError> {
        // check if guideline is expected but not provided
        if self.variables.contains("guideline") && guideline.is_none() {
            return Err(InferError::MissingTemplateVariable("guideline".to_string()));
        }

        let tools = match tools_and_prompt {
            Some((tools, tool_prompt)) => {
                // check if the `tools` variable is used in the template
                // if not, we need to append the tools to the last message
                let text = if self.use_default_tool_template {
                    match serde_json::to_string(&tools) {
                        Ok(tools_str) => format!("\n---\n{}\n{}", tools_str, tool_prompt),
                        Err(e) => return Err(InferError::ToolError(e.to_string())),
                    }
                } else {
                    // if the `tools` variable is used in the template, we just append the tool_prompt
                    format!("\n---\n{}", tool_prompt)
                };
                if let Some(last_message) = messages.last_mut() {
                    last_message.content.push(MessageChunk::Text { text });
                }
                Some(tools)
            }
            None => None,
        };

        let messages: Vec<TextMessage> = messages.into_iter().map(|c| c.into()).collect();

        self.template
            .render(ChatTemplateInputs {
                guideline,
                messages,
                bos_token: self.bos_token.as_deref(),
                eos_token: self.eos_token.as_deref(),
                add_generation_prompt: true,
                tools,
            })
            .map_err(InferError::TemplateError)
    }
}

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Manages the queues of the various adapters
    adapter_scheduler: AdapterScheduler,
    /// Maps adapter ID to a unique index
    adapter_to_index: Arc<Mutex<HashMap<AdapterParameters, u32>>>,
    /// Chat template
    chat_template: Option<ChatTemplateRenderer>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
    /// tokenizer for NER processing
    tokenizer: Option<Arc<Tokenizer>>,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: ShardedClient,
        validation: Validation,
        waiting_served_ratio: f32,
        max_batch_prefill_tokens: u32,
        max_batch_total_tokens: u32,
        max_waiting_tokens: usize,
        max_concurrent_requests: usize,
        max_active_adapters: usize,
        adapter_cycle_time_s: u64,
        requires_padding: bool,
        window_size: Option<u32>,
        generation_health: Arc<AtomicBool>,
        eager_prefill: bool,
        tokenizer_config: HubTokenizerConfig,
        tokenizer: Option<Arc<Tokenizer>>,
        block_size: u32,
        speculate: u32,
        preloaded_adapters: Vec<PreloadedAdapter>,
        prefix_caching: bool,
        chunked_prefill: bool,
        is_causal_lm: bool,
    ) -> Self {
        let adapter_event = Arc::new(AdapterEvent {
            batching_task: Notify::new(),
        });

        // Routes requests to the appropriate adapter queue
        let adapter_scheduler = AdapterScheduler::new(
            client.clone(),
            adapter_event.clone(),
            requires_padding,
            block_size,
            window_size,
            max_active_adapters,
            adapter_cycle_time_s,
            speculate,
            max_batch_total_tokens,
            prefix_caching,
            chunked_prefill,
            is_causal_lm,
        );

        // Initialize with base model adapter (empty) mapping to index 0
        let mut adapter_to_index = HashMap::from([(
            AdapterParameters {
                adapter_ids: vec![BASE_MODEL_ADAPTER_ID.to_string()],
                ..Default::default()
            },
            0,
        )]);

        // Pre-populate the adapter_to_index with the preloaded adapters
        for adapter in preloaded_adapters.iter() {
            if let Some(adapter_parameters) = &adapter.adapter_parameters {
                adapter_to_index.insert(
                    AdapterParameters {
                        adapter_ids: adapter_parameters.adapter_ids.clone(),
                        ..Default::default()
                    },
                    adapter.adapter_index,
                );
            }
        }

        let adapter_to_index = Arc::new(Mutex::new(adapter_to_index));

        let chat_template = tokenizer_config
            .chat_template
            .and_then(|t| match t {
                ChatTemplateVersions::Single(template) => Some(template),
                ChatTemplateVersions::Multiple(templates) => templates
                    .into_iter()
                    .find(|t| t.name == "default")
                    .map(|t| t.template),
            })
            .map(|t| {
                ChatTemplateRenderer::new(t, tokenizer_config.bos_token, tokenizer_config.eos_token)
            });

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            adapter_event,
            generation_health,
            adapter_scheduler.clone(),
            eager_prefill,
            chunked_prefill,
        ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        Self {
            validation,
            adapter_scheduler,
            adapter_to_index,
            chat_template,
            limit_concurrent_requests: semaphore,
            tokenizer: tokenizer,
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip(self))]
    pub(crate) async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<
        (
            OwnedSemaphorePermit,
            UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
        ),
        InferError,
    > {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("lorax_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let (adapter_source, adapter_parameters) = extract_adapter_params(
            request.parameters.adapter_id.clone(),
            request.parameters.adapter_source.clone(),
            request.parameters.adapter_parameters.clone(),
        );

        let adapter_idx;
        {
            // TODO(travis): can optimize concurrency here using RWLock
            let mut adapter_to_index = self.adapter_to_index.lock().await;
            let adapter_key = adapter_parameters.clone();
            if adapter_to_index.contains_key(&adapter_key) {
                adapter_idx = *adapter_to_index.get(&adapter_key).unwrap();
            } else {
                adapter_idx = adapter_to_index.len() as u32;
                adapter_to_index.insert(adapter_key, adapter_idx);
            }
        }

        let api_token = request.parameters.api_token.clone();
        let adapter = Adapter::new(
            adapter_parameters,
            adapter_source.unwrap(),
            adapter_idx,
            api_token,
        );

        // Validate request
        let valid_request = self
            .validation
            .validate(request, adapter.clone())
            .await
            .map_err(|err| {
                metrics::increment_counter!("lorax_request_failure", "err" => "validation");
                tracing::error!("{err}");
                err
            })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Process the request by sending it to the queue associated with `adapter`
        self.adapter_scheduler.process(
            adapter.clone(),
            Entry {
                request: Arc::new(valid_request),
                response_tx,
                span: Span::current(),
                temp_span: None,
                queue_time: Instant::now(),
                batch_time: None,
                block_allocation: None,
                id: None,
            },
        );

        // Return stream
        Ok((permit, UnboundedReceiverStream::new(response_rx)))
    }

    /// Tokenizer the input
    #[instrument(skip_all)]
    pub(crate) async fn tokenize(
        &self,
        request: GenerateRequest,
    ) -> Result<Option<tokenizers::Encoding>, InferError> {
        // Tokenize request
        let inputs = request.inputs;
        let truncate = request.parameters.truncate;
        let encoding = self
            .validation
            .tokenize(inputs, request.add_special_tokens, truncate)
            .await
            .map_err(|err| {
                tracing::error!("Error occurred during tokenization. {err}");
                err
            })?;

        // Return Encoding
        Ok(encoding.map(|(encoding, _)| encoding))
    }

    /// Apply the chat template to the chat request
    #[instrument(skip_all)]
    pub(crate) fn apply_chat_template(
        &self,
        guideline: Option<String>,
        messages: Vec<Message>,
        tools_and_prompt: Option<(Vec<Tool>, String)>,
    ) -> Result<String, InferError> {
        self.chat_template
            .as_ref()
            .ok_or_else(|| InferError::TemplateError(ErrorKind::TemplateNotFound.into()))?
            .apply(guideline.as_deref(), messages, tools_and_prompt)
            .map_err(|e| {
                metrics::increment_counter!("lorax_request_failure", "err" => "template");
                tracing::error!("{e}");
                e
            })
    }

    /// Add a new request to the queue and return a InferResponse
    #[instrument(skip(self))]
    pub(crate) async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        // Create stream and keep semaphore permit as long as generate lives
        let (_permit, mut stream) = self.generate_stream(request).await?;

        // Return values
        let mut result_prefill = Vec::new();
        let mut result_tokens = Vec::new();
        let mut result_prefill_length = 0;
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

        // Iterate on stream
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill {
                    tokens,
                    tokens_length,
                } => {
                    // Create Token objects
                    // We do that here instead of in the Python code as Rust for loops are faster
                    if let Some(tokens_val) = tokens {
                        result_prefill = tokens_val
                            .ids
                            .into_iter()
                            .zip(tokens_val.logprobs.into_iter())
                            .zip(tokens_val.texts.into_iter())
                            .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
                            .collect();
                    }
                    result_prefill_length = tokens_length;
                }
                // Push last token
                InferStreamResponse::Token(token) => result_tokens.push(token),
                // Final message
                // Set return values
                InferStreamResponse::End {
                    token,
                    generated_text,
                    start,
                    queued,
                } => {
                    result_tokens.push(token);
                    result_generated_text = Some(generated_text);
                    result_start = Some(start);
                    result_queued = Some(queued)
                }
                InferStreamResponse::Embed { .. } => {
                    // This should not happen
                    tracing::error!("Received an Embed message in generate. This is a bug.");
                }
                InferStreamResponse::Classify { .. } => {
                    // This should not happen
                    tracing::error!("Received an Classify message in generate. This is a bug.");
                }
            }
        }

        // Check that we received a `InferStreamResponse::End` message
        if let (Some(generated_text), Some(queued), Some(start)) =
            (result_generated_text, result_queued, result_start)
        {
            Ok(InferResponse {
                prefill: result_prefill,
                tokens: result_tokens,
                prompt_tokens: result_prefill_length,
                generated_text,
                queued,
                start,
            })
        } else {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("lorax_request_failure", "err" => "incomplete");
            tracing::error!("{err}");
            Err(err)
        }
    }

    #[instrument(skip(self))]
    pub(crate) async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("lorax_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let (adapter_source, adapter_parameters) = extract_adapter_params(
            request.parameters.adapter_id.clone(),
            request.parameters.adapter_source.clone(),
            request.parameters.adapter_parameters.clone(),
        );

        let adapter_idx;
        {
            // TODO(travis): can optimize concurrency here using RWLock
            let mut adapter_to_index = self.adapter_to_index.lock().await;
            let adapter_key = adapter_parameters.clone();
            if adapter_to_index.contains_key(&adapter_key) {
                adapter_idx = *adapter_to_index.get(&adapter_key).unwrap();
            } else {
                adapter_idx = adapter_to_index.len() as u32;
                adapter_to_index.insert(adapter_key, adapter_idx);
            }
        }

        let api_token = request.parameters.api_token.clone();
        let adapter = Adapter::new(
            adapter_parameters,
            adapter_source.unwrap(),
            adapter_idx,
            api_token,
        );

        // TODO(travis): robust validation
        // Validate request
        // let valid_request = self
        //     .validation
        //     .validate(request, adapter.clone())
        //     .await
        //     .map_err(|err| {
        //         metrics::increment_counter!("lorax_request_failure", "err" => "validation");
        //         tracing::error!("{err}");
        //         err
        //     })?;

        let inputs = request.inputs.clone();
        let (tokenized_inputs, input_length) = self
            .validation
            .validate_input(request.inputs, true, None, Some(1))
            .await?;

        let valid_request = ValidEmbedRequest {
            inputs,
            tokenized_inputs,
            input_length: input_length as u32,
            adapter: adapter.clone(),
        };

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Process the request by sending it to the queue associated with `adapter`
        self.adapter_scheduler.process(
            adapter.clone(),
            Entry {
                request: Arc::new(valid_request),
                response_tx,
                span: Span::current(),
                temp_span: None,
                queue_time: Instant::now(),
                batch_time: None,
                block_allocation: None,
                id: None,
            },
        );

        // Return values
        let mut return_embeddings = None;

        let mut stream = UnboundedReceiverStream::new(response_rx);
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill { .. } => {
                    tracing::error!("Received a Prefill message in embed. This is a bug.");
                }
                // Push last token
                InferStreamResponse::Token(..) => {
                    tracing::error!("Received a Token message in embed. This is a bug.");
                }
                // Final message
                // Set return values
                InferStreamResponse::End { .. } => {
                    tracing::error!("Received an End message in embed. This is a bug.");
                }
                InferStreamResponse::Classify { .. } => {
                    tracing::error!("Received a Classify message in embed. This is a bug.");
                }
                InferStreamResponse::Embed {
                    embedding,
                    start: _,
                    queued: _,
                } => {
                    return_embeddings = Some(embedding.values);
                }
            }
        }

        if let Some(return_embeddings) = return_embeddings {
            Ok(EmbedResponse {
                embeddings: return_embeddings,
            })
        } else {
            let err = InferError::EmbeddingFailure;
            metrics::increment_counter!("lorax_request_failure", "err" => "embedding_failure");
            tracing::error!("{err}");
            Err(err)
        }
    }

    #[instrument(skip(self))]
    pub(crate) async fn classify(
        &self,
        request: ClassifyRequest,
    ) -> Result<InferClassifyResponse, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("lorax_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let adapter = Adapter::new(
            AdapterParameters {
                adapter_ids: vec![BASE_MODEL_ADAPTER_ID.to_string()],
                ..Default::default()
            },
            "hub".to_string(),
            0,
            None,
        );

        let inputs = request.inputs.clone();
        let (tokenized_inputs, input_length) = self
            .validation
            .validate_input(request.inputs, true, None, Some(1))
            .await?;

        let valid_request = ValidClassifyRequest {
            inputs: inputs.clone(),
            tokenized_inputs,
            input_length: input_length as u32,
            adapter: adapter.clone(),
        };

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Process the request by sending it to the queue associated with `adapter`
        self.adapter_scheduler.process(
            adapter.clone(),
            Entry {
                request: Arc::new(valid_request),
                response_tx,
                span: Span::current(),
                temp_span: None,
                queue_time: Instant::now(),
                batch_time: None,
                block_allocation: None,
                id: None,
            },
        );

        // Return values
        let mut return_entities = None;
        let mut result_start = None;
        let mut result_queued = None;

        let mut stream = UnboundedReceiverStream::new(response_rx);
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill { .. } => {
                    tracing::error!("Received a Prefill message in classify. This is a bug.");
                }
                // Push last token
                InferStreamResponse::Token(..) => {
                    tracing::error!("Received a Token message in classify. This is a bug.");
                }
                // Final message
                // Set return values
                InferStreamResponse::End { .. } => {
                    tracing::error!("Received an End message in classify. This is a bug.");
                }
                InferStreamResponse::Embed { .. } => {
                    tracing::error!("Received an Embed message in classify. This is a bug.");
                }
                InferStreamResponse::Classify {
                    predictions,
                    start,
                    queued,
                    id: _,
                } => {
                    let entities = aggregate_ner_output_simple(
                        inputs.clone(),
                        predictions,
                        self.tokenizer.clone().unwrap(),
                    );
                    return_entities = Some(entities);
                    result_start = Some(start);
                    result_queued = Some(queued);
                }
            }
        }

        if let Some(return_entities) = return_entities {
            Ok(InferClassifyResponse {
                predictions: return_entities,
                queued: result_queued.unwrap(),
                start: result_start.unwrap(),
            })
        } else {
            let err = InferError::ClassificationFailure;
            metrics::increment_counter!("lorax_request_failure", "err" => "classification_failure");
            tracing::error!("{err}");
            Err(err)
        }
    }

    #[instrument(skip(self))]
    pub(crate) async fn classify_batch(
        &self,
        request: BatchClassifyRequest,
    ) -> Result<Vec<InferClassifyResponse>, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("lorax_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let adapter = Adapter::new(
            AdapterParameters {
                adapter_ids: vec![BASE_MODEL_ADAPTER_ID.to_string()],
                ..Default::default()
            },
            "hub".to_string(),
            0,
            None,
        );

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        let request_id_map: HashMap<u64, String> = request
            .inputs
            .iter()
            .enumerate()
            .map(|(id, input)| (id as u64, input.clone()))
            .collect();

        // Call validate_input on every input in the request and await the results
        let futures: Vec<_> = request
            .inputs
            .iter()
            .map(|input| {
                self.validation
                    .validate_input(input.clone(), true, None, Some(1))
            })
            .collect();

        let all_tokenized_inputs = try_join_all(futures).await?;

        for ((id, r_inputs), (tokenized_inputs, input_length)) in
            request.inputs.iter().enumerate().zip(all_tokenized_inputs)
        {
            let inputs = r_inputs.to_string().clone();
            let valid_request = ValidClassifyRequest {
                inputs,
                tokenized_inputs,
                input_length: input_length as u32,
                adapter: adapter.clone(),
            };

            // Process the request by sending it to the queue associated with `adapter`
            self.adapter_scheduler.process(
                adapter.clone(),
                Entry {
                    request: Arc::new(valid_request),
                    response_tx: response_tx.clone(),
                    span: Span::current(),
                    temp_span: None,
                    queue_time: Instant::now(),
                    batch_time: None,
                    block_allocation: None,
                    id: Some(id as u64),
                },
            );
        }

        drop(response_tx); // Close the sending end

        // Return values

        let mut all_entities = HashMap::new();
        let mut stream = UnboundedReceiverStream::new(response_rx);
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Classify {
                    predictions,
                    start,
                    queued,
                    id,
                } => {
                    let request_inputs = request_id_map.get(&id.unwrap()).unwrap().clone();
                    let entities = aggregate_ner_output_simple(
                        request_inputs,
                        predictions.clone(),
                        self.tokenizer.clone().unwrap(),
                    );
                    all_entities.insert(
                        id.unwrap(),
                        InferClassifyResponse {
                            predictions: entities,
                            queued,
                            start,
                        },
                    );
                }
                _ => {
                    tracing::error!(
                        "Received unexpected message type in classify_batch. This is a bug."
                    );
                }
            }
        }
        if all_entities.is_empty() {
            let err = InferError::ClassificationFailure;
            metrics::increment_counter!("lorax_request_failure", "err" => "classification_failure");
            tracing::error!("{err}");
            Err(err)
        } else {
            let mut sorted_responses: Vec<_> = all_entities.into_iter().collect();
            sorted_responses.sort_by_key(|&(id, _)| id);

            let sorted_responses: Vec<InferClassifyResponse> = sorted_responses
                .into_iter()
                .map(|(_, response)| response)
                .collect();

            Ok(sorted_responses)
        }
    }

    /// Add best_of new requests to the queue and return a InferResponse of the sequence with
    /// the highest log probability per token
    #[instrument(skip(self))]
    pub(crate) async fn generate_best_of(
        &self,
        request: GenerateRequest,
        best_of: usize,
    ) -> Result<(InferResponse, Vec<InferResponse>), InferError> {
        // validate  best_of parameter separately
        let best_of = self.validation.validate_best_of(best_of)?;

        // create multiple generate requests
        let mut infer_responses: Vec<InferResponse> =
            try_join_all((0..best_of).map(|_| self.generate(request.clone()))).await?;

        // get the sequence with the highest log probability per token
        let mut max_index = 0;
        let mut max_logprob: f32 = f32::MIN;

        for (i, response) in infer_responses.iter().enumerate() {
            // mean logprobs of the generated tokens
            let sequence_logprob = response
                .tokens
                .iter()
                .map(|token| token.logprob)
                .sum::<f32>()
                / response.tokens.len() as f32;

            // set best sequence
            if sequence_logprob > max_logprob {
                max_index = i;
                max_logprob = sequence_logprob;
            }
        }
        let best_response = infer_responses.remove(max_index);
        Ok((best_response, infer_responses))
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Grabs a queue from the adapter manager.
/// Then, batches requests and sends them to the inference server
#[allow(clippy::too_many_arguments)]
async fn batching_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    adapter_event: Arc<AdapterEvent>,
    generation_health: Arc<AtomicBool>,
    adapter_scheduler: AdapterScheduler,
    eager_prefill: bool,
    chunked_prefill: bool,
) {
    // Infinite loop
    loop {
        // Fire if a new request comes in or an adapter becomes ready
        adapter_event.batching_task.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut batch_entries, batch, span)) = adapter_scheduler
            .next_batch(
                HashSet::new(),
                None,
                max_batch_prefill_tokens,
                max_batch_total_tokens,
            )
            .await
        {
            let mut cached_batch = batch_entries
                .process_first(&mut client, batch, None, span, &generation_health)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let mut batch_size = batch.size;
                let batch_max_tokens = batch.max_tokens;
                let current_tokens = batch.current_tokens;
                let mut batches = vec![batch];
                metrics::gauge!("lorax_batch_current_size", batch_size as f64);
                metrics::gauge!("lorax_batch_current_max_tokens", batch_max_tokens as f64);

                // Cleanup any adapters that are in an errored state
                // TODO(travis): can execute this more efficiently by making it event-driven
                adapter_scheduler.remove_errored_adapters().await;

                let mut token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);
                let (min_size, _max_size, prefill_token_budget) = if chunked_prefill {
                    // Since the next batch will be concatenated with the current batch,
                    // the current batch tokens must be subtracted to the prefill budget
                    let prefill_token_budget =
                        max_batch_prefill_tokens.saturating_sub(current_tokens);
                    // We can ignore min_size and max_size
                    // Models than rely on max_size cannot support chunking
                    // Regarding min_size, chunking allow us to consistently run at the compute
                    // bound, making min_size useless.
                    (None, None, prefill_token_budget)
                } else {
                    let min_size = if waiting_tokens >= max_waiting_tokens {
                        // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                        // to add a new batch even though its size might be small
                        None
                    } else {
                        // Minimum batch size
                        // TODO: temporarily disable to avoid incorrect deallocation +
                        //       reallocation when using prefix caching.
                        Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                    };

                    let max_batch_size: Option<usize> = None; // TODO(travis)
                    let max_size =
                        max_batch_size.map(|max_size| max_size.saturating_sub(batch_size as usize));

                    (min_size, max_size, max_batch_prefill_tokens)
                };

                // let min_size = if waiting_tokens >= max_waiting_tokens || eager_prefill {
                //     // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                //     // to add a new batch even though its size might be small
                //     None
                // } else {
                //     // Minimum batch size
                //     Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                // };

                let mut adapters_in_use = batch_entries.adapters_in_use();

                // Try to get a new batch
                while let Some((new_entries, new_batch, span)) = adapter_scheduler
                    .next_batch(
                        adapters_in_use.clone(),
                        min_size,
                        prefill_token_budget,
                        token_budget,
                    )
                    .await
                {
                    let new_batch_size = new_batch.size;
                    batch_size += new_batch_size;

                    let new_batch_max_tokens = new_batch.max_tokens;
                    token_budget = token_budget.saturating_sub(new_batch_max_tokens);

                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::increment_counter!("lorax_batch_concat", "reason" => "backpressure");
                    } else {
                        if chunked_prefill {
                            metrics::increment_counter!("lorax_batch_concat", "reason" => "chunking")
                        } else {
                            metrics::increment_counter!("lorax_batch_concat", "reason" => "wait_exceeded")
                        };
                    }

                    let cached_batch = if chunked_prefill {
                        // Concat current batch to the new one
                        batches.pop()
                    } else {
                        // Request are waiting only if we don't support chunking
                        batch_entries.mut_state().batch_entries.iter_mut().for_each(
                            |(_, entry)| {
                                // Create a new span to add the info that this entry is waiting
                                // because a new batch is being computed
                                let entry_waiting_span = info_span!(parent: &entry.span, "waiting");
                                // Add relationships
                                span.follows_from(&entry_waiting_span);
                                entry_waiting_span.follows_from(&span);
                                // Update entry
                                entry.temp_span = Some(entry_waiting_span);
                            },
                        );
                        None
                    };

                    let new_adapters_in_use = new_entries.adapters_in_use();
                    batch_entries.extend(new_entries);

                    // Generate one token for this new batch to have the attention past in cache
                    let new_cached_batch = batch_entries
                        .process_first(
                            &mut client,
                            new_batch,
                            cached_batch,
                            span,
                            &generation_health,
                        )
                        .await;

                    adapters_in_use.extend(new_adapters_in_use);

                    // Reset waiting counter
                    waiting_tokens = 1;
                    // Extend current batch with the new batch
                    if let Some(new_cached_batch) = new_cached_batch {
                        batches.push(new_cached_batch);
                    } else if chunked_prefill {
                        // New cached batch is empty, no work left
                        break;
                    }

                    if !eager_prefill {
                        // Do not continue to loop if we are not in eager prefill mode
                        break;
                    }
                }

                // Create span for this batch to add context to inference calls
                let next_batch_size = batch_entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);

                batch_entries
                    .mut_state()
                    .batch_entries
                    .iter_mut()
                    .for_each(|(_, entry)| {
                        // Create a new span to link the batch back to this entry
                        let entry_batch_span = info_span!(parent: &entry.span, "infer");
                        // Add relationships
                        next_batch_span.follows_from(&entry_batch_span);
                        entry_batch_span.follows_from(&next_batch_span);
                        // Update entry
                        entry.temp_span = Some(entry_batch_span);
                    });

                cached_batch = batch_entries
                    .process_next(&mut client, batches, next_batch_span, &generation_health)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("lorax_batch_current_size", 0.0);
            metrics::gauge!("lorax_batch_current_max_tokens", 0.0);
        }
    }
}

#[instrument(skip_all)]
pub(crate) async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    cached_batch: Option<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("lorax_batch_inference_count", "method" => "prefill");

    match client.prefill(batch, cached_batch).await {
        Ok((generations, next_batch)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            let removed = filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries, removed).await;

            // TODO(travis)
            // if let Some(concat_duration) = timings.concat {
            //     metrics::histogram!("lorax_batch_concat_duration", "method" => "decode")
            //         .record(concat_duration.as_secs_f64());
            // }

            metrics::histogram!("lorax_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "prefill");
            metrics::increment_counter!("lorax_batch_inference_success", "method" => "prefill");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            generation_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("lorax_batch_inference_failure", "method" => "prefill");
            None
        }
    }
}

#[instrument(skip_all)]
pub(crate) async fn decode(
    client: &mut ShardedClient,
    batches: Vec<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_ids: Vec<u64> = batches.iter().map(|b| b.id).collect();
    metrics::increment_counter!("lorax_batch_inference_count", "method" => "decode");

    match client.decode(batches).await {
        Ok((generations, next_batch)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            let removed = filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries, removed).await;

            metrics::histogram!("lorax_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "decode");
            metrics::increment_counter!("lorax_batch_inference_success", "method" => "decode");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            generation_health.store(false, Ordering::SeqCst);
            for id in batch_ids {
                let _ = client.clear_cache(Some(id)).await;
            }
            send_errors(err, entries);
            metrics::increment_counter!("lorax_batch_inference_failure", "method" => "decode");
            None
        }
    }
}

#[instrument(skip_all)]
pub(crate) async fn embed(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("lorax_batch_inference_count", "method" => "embed");

    match client.embed(batch).await {
        Ok(results) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            results.into_iter().for_each(|embedding| {
                let id = embedding.request_id;
                // Get entry
                // We can `expect` here as the request id should always be in the entries
                let entry = entries
                    .get(&id)
                    .expect("ID not found in entries. This is a bug.");

                // Send generation responses back to the infer task
                // If the receive an error from the Flume channel, it means that the client dropped the
                // request and we need to stop generating hence why we unwrap_or(true)
                let stopped = send_embeddings(embedding, entry)
                    .map_err(|err| {
                        tracing::error!("Entry response channel error.");
                        metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
                        err
                    })
                    .unwrap_or(true);
                if stopped {
                    entries
                        .remove(&id)
                        .expect("ID not found in entries. This is a bug.");
                }
            });

            metrics::histogram!("lorax_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "embed");
            metrics::increment_counter!("lorax_batch_inference_success", "method" => "embed");
            None
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            generation_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("lorax_batch_inference_failure", "method" => "embed");
            None
        }
    }
}

#[instrument(skip_all)]
pub(crate) async fn classify(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("lorax_batch_inference_count", "method" => "classify");

    match client.classify(batch).await {
        Ok(results) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            results.into_iter().for_each(|predictions| {
                let id = predictions.request_id;
                // Get entry
                // We can `expect` here as the request id should always be in the entries
                let entry = entries
                    .get(&id)
                    .expect("ID not found in entries. This is a bug.");

                // Send generation responses back to the infer task
                // If the receive an error from the Flume channel, it means that the client dropped the
                // request and we need to stop generating hence why we unwrap_or(true)
                let stopped = send_classifications(predictions, entry)
                    .map_err(|err| {
                        tracing::error!("Entry response channel error.");
                        metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
                        err
                    })
                    .unwrap_or(true);
                if stopped {
                    entries
                        .remove(&id)
                        .expect("ID not found in entries. This is a bug.");
                }
            });

            metrics::histogram!("lorax_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "classify");
            metrics::increment_counter!("lorax_batch_inference_success", "method" => "classify");
            None
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            generation_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("lorax_batch_inference_failure", "method" => "classify");
            None
        }
    }
}
/// Filter a `batch` and remove all requests not present in `entries`
#[instrument(skip_all)]
async fn filter_batch(
    client: &mut ShardedClient,
    next_batch: Option<CachedBatch>,
    entries: &IntMap<u64, Entry>,
    removed: bool,
) -> Option<CachedBatch> {
    let mut batch = next_batch?;

    // No need to filter is we haven't removed any entries
    if !removed {
        return Some(batch);
    }

    let id = batch.id;

    // Retain only requests that are still in entries
    batch.request_ids.retain(|id| entries.contains_key(id));

    if batch.request_ids.is_empty() {
        // All requests have been filtered out
        // Next batch is now empty
        // Clear it from the Python shards cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.clear_cache(Some(id)).await.unwrap();
        None
    } else {
        // Filter Python shard cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.filter_batch(id, batch.request_ids).await.unwrap()
    }
}

/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
/// and filter entries
#[instrument(skip_all)]
fn filter_send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) -> bool {
    let mut removed = false;
    generations.into_iter().for_each(|generation| {
        let id = generation.request_id;
        // Get entry
        // We can `expect` here as the request id should always be in the entries
        let entry = entries
            .get(&id)
            .expect("ID not found in entries. This is a bug.");

        // Create and enter a span to link this function back to the entry
        let _span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_generation", generation = ?generation).entered();
        // Send generation responses back to the infer task
        // If the receive an error from the Flume channel, it means that the client dropped the
        // request and we need to stop generating hence why we unwrap_or(true)
        let stopped = send_responses(generation, entry).inspect_err(|_err| {
            tracing::error!("Entry response channel error.");
            metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
        }).unwrap_or(true);
        if stopped {
            entries.remove(&id).expect("ID not found in entries. This is a bug.");
            removed = true;
        }
    });
    removed
}

/// Send responses through the `entry` response channel
fn send_responses(
    generation: Generation,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is closed
    let request_id = generation.request_id;
    if entry.response_tx.is_closed() {
        tracing::error!("Entry id={request_id:?} response channel closed.");
        metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
        return Ok(true);
    }

    let mut stopped = false;

    if generation.prefill_tokens_length > 0 {
        // Send message
        entry.response_tx.send(Ok(InferStreamResponse::Prefill {
            tokens: generation.prefill_tokens,
            tokens_length: generation.prefill_tokens_length,
        }))?;
    }

    // Create last Token
    let next_tokens = generation.next_tokens.unwrap_or_default();
    let alternative_tokens = if next_tokens.alternative_tokens.is_empty() {
        // Pad with Nones the same length as the IDs so it zips correctly
        vec![None; next_tokens.ids.len()]
    } else {
        // Convertion from AlternativeToken to Option<AlternativeToken>
        next_tokens
            .alternative_tokens
            .into_iter()
            .map(Some)
            .collect()
    };

    let ntokens = next_tokens.ids.len();
    metrics::histogram!("lorax_request_skipped_tokens", (ntokens - 1) as f64);
    let mut iterator = multizip((
        next_tokens.ids,
        next_tokens.logprobs,
        next_tokens.texts,
        next_tokens.is_special,
        alternative_tokens,
    ))
    .peekable();

    while let Some((id, logprob, text, special, alternative_tokens)) = iterator.next() {
        let token = Token {
            id,
            text,
            logprob,
            special,
            alternative_tokens: alternative_tokens.and_then(|at| {
                Some(
                    at.ids
                        .into_iter()
                        .zip(at.logprobs.into_iter())
                        .zip(at.texts.into_iter())
                        .map(|((id, logprob), text)| AlternativeToken { id, text, logprob })
                        .collect(),
                )
            }),
        };

        match (&generation.generated_text, iterator.peek()) {
            (Some(generated_text), None) => {
                // Generation has ended
                stopped = true;
                // Send message
                entry.response_tx.send(Ok(InferStreamResponse::End {
                    token,
                    generated_text: generated_text.clone(),
                    queued: entry.queue_time,
                    start: entry.batch_time.unwrap(),
                }))?;
            }
            _ => {
                // Send message
                entry
                    .response_tx
                    .send(Ok(InferStreamResponse::Token(token)))?;
            }
        }
    }

    Ok(stopped)
}

/// Send responses through the `entry` response channel
fn send_embeddings(
    embedding: Embedding,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected
    if entry.response_tx.is_closed() {
        return Ok(true);
    }

    entry.response_tx.send(Ok(InferStreamResponse::Embed {
        embedding: embedding.clone(),
        queued: entry.queue_time,
        start: entry.batch_time.unwrap(),
    }))?;

    // TODO(travis): redundant as we always return true, just make it return nothing
    Ok(true)
}

/// Send responses through the `entry` response channel
fn send_classifications(
    predictions: ClassifyPredictionList,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected
    if entry.response_tx.is_closed() {
        return Ok(true);
    }

    entry.response_tx.send(Ok(InferStreamResponse::Classify {
        predictions: predictions.clone(),
        queued: entry.queue_time,
        start: entry.batch_time.unwrap(),
        id: entry.id,
    }))?;

    // TODO(travis): redundant as we always return true, just make it return nothing
    Ok(true)
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::GenerationError(error.to_string());
        metrics::increment_counter!("lorax_request_failure", "err" => "generation");
        tracing::error!("{err}");

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(err))
            .unwrap_or(());
    });
}

#[derive(Debug)]
pub(crate) enum InferStreamResponse {
    // Optional first message
    Prefill {
        tokens: Option<PrefillTokens>,
        tokens_length: u32,
    },
    // Intermediate messages
    Token(Token),
    // Embeddings
    Embed {
        embedding: Embedding,
        // For now allow this field even though it is unused.
        // TODO:(magdy) enable tracing for these requests
        #[allow(dead_code)]
        start: Instant,
        #[allow(dead_code)]
        queued: Instant,
    },
    Classify {
        predictions: ClassifyPredictionList,
        // For now allow this field even though it is unused.
        // TODO:(magdy) enable tracing for these requests
        #[allow(dead_code)]
        start: Instant,
        #[allow(dead_code)]
        queued: Instant,
        id: Option<u64>, // to support batching
    },
    // Last message
    End {
        token: Token,
        generated_text: GeneratedText,
        start: Instant,
        queued: Instant,
    },
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) prefill: Vec<PrefillToken>,
    pub(crate) tokens: Vec<Token>,
    pub(crate) prompt_tokens: u32,
    pub(crate) generated_text: GeneratedText,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Input validation error: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Incomplete generation")]
    IncompleteGeneration,
    #[error("Failed applying chat template to inputs: {0}")]
    TemplateError(#[from] minijinja::Error),
    #[error("Embedding Failure")]
    EmbeddingFailure,
    #[error("Classification Failure")]
    ClassificationFailure,
    #[error("Tool error: {0}")]
    ToolError(String),
    #[error("Missing template vatiable: {0}")]
    MissingTemplateVariable(String),
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::GenerationError(_) => "generation",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::IncompleteGeneration => "incomplete_generation",
            InferError::TemplateError(_) => "template_error",
            InferError::EmbeddingFailure => "embedding_failure",
            InferError::ClassificationFailure => "classification_failure",
            InferError::ToolError(_) => "tool_error",
            InferError::MissingTemplateVariable(_) => "missing_template_variable",
        }
    }
}

#[derive(Debug)]
pub(crate) struct InferClassifyResponse {
    pub(crate) predictions: Vec<Entity>,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
}

fn get_tag(token_class: &str) -> (String, String) {
    // TODO: don't make the null tag hardcoded
    let parts: Vec<&str> = token_class.split('-').collect();
    if parts.len() == 2 {
        (parts[0].to_string(), parts[1].to_string())
    } else {
        ("0".to_string(), "0".to_string())
    }
}

fn aggregate_ner_output_simple(
    input: String,
    classify_prediction_list: ClassifyPredictionList,
    tokenizer: Arc<Tokenizer>,
) -> Vec<Entity> {
    // Encode the input
    let encoded = tokenizer.encode(input.clone(), false).unwrap();

    let predicted_token_classes =
        &classify_prediction_list.predictions[1..classify_prediction_list.predictions.len() - 1];
    let scores = &classify_prediction_list.scores[1..classify_prediction_list.scores.len() - 1];

    // Initialize result and tracking variables
    let mut ner_results = Vec::new();
    let mut current_entity: Option<Entity> = None;
    let mut entity_scores = Vec::new();

    for (offset, token_class, score) in
        izip!(encoded.get_offsets(), predicted_token_classes, scores)
    {
        let (bi, tag) = get_tag(token_class);
        if bi == "B"
            || (current_entity.is_some() && tag != current_entity.as_ref().unwrap().entity_group)
        {
            if let Some(entity) = current_entity.take() {
                ner_results.push(entity);
                entity_scores.clear();
                entity_scores.push(*score);
            }
            current_entity = Some(Entity {
                entity_group: tag,
                score: *score,
                word: "".to_string(), // stub for now. set later in second pass
                start: offset.0,
                end: offset.1,
            });
        } else if current_entity.is_some() {
            entity_scores.push(*score);
            let entity = current_entity.as_mut().unwrap();
            entity.score = entity_scores.iter().sum::<f32>() / entity_scores.len() as f32;
            entity.end = offset.1;
        }
    }
    if let Some(entity) = current_entity.take() {
        ner_results.push(entity);
    }
    let mut new_ner_results = Vec::with_capacity(ner_results.len());
    for mut entity in ner_results {
        entity.word = input[entity.start..entity.end]
            .to_string()
            .to_ascii_lowercase(); // Needed to match the huggingface NER format
        new_ner_results.push(entity);
    }
    new_ner_results
}
