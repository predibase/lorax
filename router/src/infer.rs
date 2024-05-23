/// Batching and inference logic
use crate::adapter::{extract_adapter_params, Adapter, BASE_MODEL_ADAPTER_ID};
use crate::queue::AdapterEvent;
use crate::scheduler::AdapterScheduler;
use crate::validation::{Validation, ValidationError};
use crate::{AdapterParameters, AlternativeToken, Entry, Token};
use crate::{GenerateRequest, PrefillToken};
use flume::r#async::RecvStream;
use flume::SendTimeoutError;
use futures::future::try_join_all;
use futures::stream::StreamExt;
use itertools::multizip;
use lorax_client::{
    Batch, CachedBatch, ClientError, GeneratedText, Generation, PrefillTokens, ShardedClient,
};
use nohash_hasher::IntMap;
use std::collections::{HashMap, HashSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tracing::{info_span, instrument, Instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Manages the queues of the various adapters
    adapter_scheduler: AdapterScheduler,
    /// Maps adapter ID to a unique index
    adapter_to_index: Arc<Mutex<HashMap<AdapterParameters, u32>>>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
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
    ) -> Self {
        let adapter_event = Arc::new(AdapterEvent {
            batching_task: Notify::new(),
        });

        // Routes requests to the appropriate adapter queue
        let adapter_scheduler = AdapterScheduler::new(
            client.clone(),
            adapter_event.clone(),
            requires_padding,
            16,
            window_size,
            max_active_adapters,
            adapter_cycle_time_s,
        );

        // Initialize with base model adapter (empty) mapping to index 0
        let adapter_to_index = Arc::new(Mutex::new(HashMap::from([(
            AdapterParameters {
                adapter_ids: vec![BASE_MODEL_ADAPTER_ID.to_string()],
                ..Default::default()
            },
            0,
        )])));

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
        ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        Self {
            validation,
            adapter_scheduler,
            adapter_to_index,
            limit_concurrent_requests: semaphore,
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
            RecvStream<Result<InferStreamResponse, InferError>>,
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
        let (response_tx, response_rx) = flume::unbounded();

        // Process the request by sending it to the queue associated with `adapter`
        self.adapter_scheduler.process(
            adapter.clone(),
            Entry {
                request: valid_request,
                response_tx,
                span: Span::current(),
                temp_span: None,
                queue_time: Instant::now(),
                batch_time: None,
            },
        );

        // Return stream
        Ok((permit, response_rx.into_stream()))
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
            .tokenize(inputs, truncate)
            .await
            .map_err(|err| {
                tracing::error!("Error occurred during tokenization. {err}");
                err
            })?;

        // Return Encoding
        Ok(encoding.map(|(encoding, _)| encoding))
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
) {
    // Infinite loop
    loop {
        // Fire if a new request comes in or an adapter becomes ready
        adapter_event.batching_task.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) = adapter_scheduler
            .next_batch(
                HashSet::new(),
                None,
                max_batch_prefill_tokens,
                max_batch_total_tokens,
            )
            .await
        {
            let mut cached_batch = prefill(&mut client, batch, &mut entries, &generation_health)
                .instrument(span)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let batch_max_tokens = batch.max_tokens;
                let mut batches = vec![batch];
                metrics::gauge!("lorax_batch_current_size", batch_size as f64);
                metrics::gauge!("lorax_batch_current_max_tokens", batch_max_tokens as f64);

                // Cleanup any adapters that are in an errored state
                // TODO(travis): can execute this more efficiently by making it event-driven
                adapter_scheduler.remove_errored_adapters().await;

                let min_size = if waiting_tokens >= max_waiting_tokens {
                    // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                    // to add a new batch even though its size might be small
                    None
                } else {
                    // Minimum batch size
                    Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                };

                let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);

                let adapters_in_use = entries
                    .iter()
                    .map(|(_, entry)| entry.request.adapter.clone())
                    .collect::<HashSet<_>>();

                // Try to get a new batch
                if let Some((mut new_entries, new_batch, span)) = adapter_scheduler
                    .next_batch(
                        adapters_in_use,
                        min_size,
                        max_batch_prefill_tokens,
                        token_budget,
                    )
                    .await
                {
                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::increment_counter!("lorax_batch_concat", "reason" => "backpressure");
                    } else {
                        metrics::increment_counter!("lorax_batch_concat", "reason" => "wait_exceeded");
                    }

                    entries.iter_mut().for_each(|(_, entry)| {
                        // Create a new span to add the info that this entry is waiting
                        // because a new batch is being computed
                        let entry_waiting_span = info_span!(parent: &entry.span, "waiting");
                        // Add relationships
                        span.follows_from(&entry_waiting_span);
                        entry_waiting_span.follows_from(&span);
                        // Update entry
                        entry.temp_span = Some(entry_waiting_span);
                    });

                    // Generate one token for this new batch to have the attention past in cache
                    let new_cached_batch =
                        prefill(&mut client, new_batch, &mut new_entries, &generation_health)
                            .instrument(span)
                            .await;
                    // Reset waiting counter
                    waiting_tokens = 1;
                    // Extend current batch with the new batch
                    if let Some(new_cached_batch) = new_cached_batch {
                        entries.extend(new_entries);
                        batches.push(new_cached_batch);
                    }
                }

                // Create span for this batch to add context to inference calls
                let next_batch_size = entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);
                entries.iter_mut().for_each(|(_, entry)| {
                    // Create a new span to link the batch back to this entry
                    let entry_batch_span = info_span!(parent: &entry.span, "infer");
                    // Add relationships
                    next_batch_span.follows_from(&entry_batch_span);
                    entry_batch_span.follows_from(&next_batch_span);
                    // Update entry
                    entry.temp_span = Some(entry_batch_span);
                });

                cached_batch = decode(&mut client, batches, &mut entries, &generation_health)
                    .instrument(next_batch_span)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("lorax_batch_current_size", 0.0);
            metrics::gauge!("lorax_batch_current_max_tokens", 0.0);
        }
    }
}

#[instrument(skip_all)]
async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("lorax_batch_inference_count", "method" => "prefill");

    match client.prefill(batch).await {
        Ok((generations, next_batch)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

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
async fn decode(
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
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

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

/// Filter a `batch` and remove all requests not present in `entries`
#[instrument(skip_all)]
async fn filter_batch(
    client: &mut ShardedClient,
    next_batch: Option<CachedBatch>,
    entries: &IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let mut batch = next_batch?;

    // No need to filter
    if batch.size as usize == entries.len() {
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
fn filter_send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
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
        let stopped = send_responses(generation, entry).map_err(|err| {
            if let SendTimeoutError::Timeout(_) = *err {
                tracing::error!("Entry response channel timed out.")
            }

            metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
            err
        }).unwrap_or(true);
        if stopped {
            entries.remove(&id).expect("ID not found in entries. This is a bug.");
        }
    });
}

/// Send responses through the `entry` response channel
fn send_responses(
    generation: Generation,
    entry: &Entry,
) -> Result<bool, Box<SendTimeoutError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected
    if entry.response_tx.is_disconnected() {
        return Ok(true);
    }

    let mut stopped = false;

    if generation.prefill_tokens_length > 0 {
        // Send message
        entry.response_tx.send_timeout(
            Ok(InferStreamResponse::Prefill {
                tokens: generation.prefill_tokens,
                tokens_length: generation.prefill_tokens_length,
            }),
            Duration::from_millis(10),
        )?;
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
                entry.response_tx.send_timeout(
                    Ok(InferStreamResponse::End {
                        token,
                        generated_text: generated_text.clone(),
                        queued: entry.queue_time,
                        start: entry.batch_time.unwrap(),
                    }),
                    Duration::from_millis(10),
                )?;
            }
            _ => {
                // Send message
                entry.response_tx.send_timeout(
                    Ok(InferStreamResponse::Token(token)),
                    Duration::from_millis(10),
                )?;
            }
        }
    }

    Ok(stopped)
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
            .send_timeout(Err(err), Duration::from_millis(10))
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
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::GenerationError(_) => "generation",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::IncompleteGeneration => "incomplete_generation",
        }
    }
}
