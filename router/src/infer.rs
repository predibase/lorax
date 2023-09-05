/// Batching and inference logic
use crate::validation::{Validation, ValidationError};
use crate::{Entry, Queue, Token};
use crate::{GenerateRequest, PrefillToken};
use flume::r#async::RecvStream;
use flume::SendTimeoutError;
use futures::future::try_join_all;
use futures::stream::StreamExt;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use text_generation_client::{
    Batch, CachedBatch, ClientError, GeneratedText, Generation, PrefillTokens, Request, ShardedClient,
};
use thiserror::Error;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError, oneshot, Mutex};
use tokio::time::Instant;
use tracing::{info_span, instrument, Instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Manages the queues of the various adapters
    adapter_manager: AdapterManager,
    /// Shared state
    shared: Arc<Shared>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
}

/// Infer shared state
struct Shared {
    /// Batching background Tokio task notifier
    batching_task: Notify,
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

enum AdapterQueueCommand {
    Append {
        response_sender: oneshot::Sender<bool>,
        entry: Box<Entry>,
    },
    IsEmpty {
        response_sender: oneshot::Sender<bool>,
    },
    NextBatch {
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },

}

struct AdapterQueue {
    sender: flume::Sender<AdapterQueueCommand>,
}

impl AdapterQueue {
    pub(crate) fn new(requires_padding: bool, block_size: u32) -> Self {
        let (sender, receiver) = flume::unbounded();

        tokio::spawn(adapter_queue_task(
            requires_padding,
            block_size,
            receiver,
        ));
        Self {
            sender,
        }
    }

    pub(crate) async fn append(&self, entry: Box<Entry>) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterQueueCommand::Append {
                response_sender,
                entry,
            })
            .unwrap();
        response_receiver.await.unwrap();
    }

    pub(crate) async fn is_empty(&self) -> bool {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterQueueCommand::IsEmpty {
                response_sender,
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.sender
            .send(AdapterQueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span: Span::current(),
            })
            .unwrap();
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.unwrap()
    }
}

struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Paged Attention block size
    block_size: u32,
}

impl State {
    fn new(requires_padding: bool, block_size: u32) -> Self {
        Self {
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            next_batch_id: 0,
            requires_padding,
            block_size,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
    }

    // Is empty
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // Len
    fn len(&self) -> usize {
        self.entries.len()
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        let mut max_input_length = 0;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;

        // Pop entries starting from the front of the queue
        while let Some((id, mut entry)) = self.entries.pop_front() {
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_disconnected() {
                metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
                continue;
            }

            if self.requires_padding {
                // We pad to max input length in the Python shards
                // We need to take these padding tokens into the equation
                max_input_length = max_input_length.max(entry.request.input_length);
                prefill_tokens = (batch_requests.len() + 1) as u32 * max_input_length
            } else {
                // pad to block size
                prefill_tokens += ((entry.request.input_length + self.block_size - 1)
                    / self.block_size)
                    * self.block_size;
            }

            if self.requires_padding {
                decode_tokens += entry.request.stopping_parameters.max_new_tokens;
            } else {
                // pad to block size
                decode_tokens +=
                    ((entry.request.stopping_parameters.max_new_tokens + self.block_size - 1)
                        / self.block_size)
                        * self.block_size;
            }

            if prefill_tokens > prefill_token_budget
                || (prefill_tokens + decode_tokens) > token_budget
            {
                // Entry is over budget
                // Add it back to the front
                self.entries.push_front((id, entry));
                break;
            }

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                inputs: entry.request.inputs.clone(),
                truncate: entry.request.truncate,
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: Some(entry.request.stopping_parameters.clone()),
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Empty batch
        if batch_requests.is_empty() {
            return None;
        }

        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            // Batch is too small
            if batch_requests.len() < min_size {
                // Add back entries to the queue in the correct order
                for r in batch_requests.into_iter().rev() {
                    let id = r.id;
                    let entry = batch_entries.remove(&id).unwrap();
                    self.entries.push_front((id, entry));
                }

                return None;
            }
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: (prefill_tokens + decode_tokens),
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("tgi_batch_next_size", batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }
}

async fn adapter_queue_task(
    requires_padding: bool,
    block_size: u32,
    receiver: flume::Receiver<AdapterQueueCommand>,
) {
    let mut state = State::new(requires_padding, block_size);

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterQueueCommand::Append{
                response_sender, 
                entry
            } => {
                state.append(*entry);
                println!("ASDFASDF inside adapter_queue_task. queue length: {}", state.len());
                response_sender.send(true).unwrap();
            }
            AdapterQueueCommand::IsEmpty { 
                response_sender
            } => {
                let response = state.is_empty();
                response_sender.send(response).unwrap();
            }
            AdapterQueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(min_size, prefill_token_budget, token_budget);
                response_sender.send(next_batch).unwrap();
                metrics::gauge!("tgi_queue_size", state.entries.len() as f64);
            }),
        }
    }
}

enum AdapterManagerCommand {
    Append(String, Box<Entry>),
}

#[derive(Clone)]
struct AdapterManager {
    sender: flume::Sender<AdapterManagerCommand>,
}

impl AdapterManager {
    pub(crate) fn new(
        requires_padding: bool,
        block_size: u32,
        batcher_sender: flume::Sender<BatcherCommand>,
    ) -> Self {
        let (sender, receiver) = flume::unbounded();

        // receives requests from the infer struct and sends them to the appropriate adapter queue
        tokio::spawn(adapter_manager_task(
            requires_padding,
            block_size,
            receiver,
            batcher_sender
        ));

        Self {
            sender,
        }
    }

    pub(crate) fn process(&self, adapter_id: String, entry: Entry) {
        // only blocks until the message is sent
        // the adapter manager task will handle the actual processing
        self.sender.send(AdapterManagerCommand::Append(adapter_id, Box::new(entry))).unwrap();
    }
}

async fn adapter_manager_task(
    requires_padding: bool,
    block_size: u32,
    receiver: flume::Receiver<AdapterManagerCommand>,
    batcher_sender: flume::Sender<BatcherCommand>,
) {
    let mut queue_map: HashMap<String, Arc<AdapterQueue>> = HashMap::new();

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterManagerCommand::Append(adapter_id, entry) => {
                println!("ASDFASDF inside adapter_manager_task. adapter_id: {}", adapter_id);

                let queue = queue_map.entry(adapter_id.clone()).or_insert_with(|| {
                    Arc::new(AdapterQueue::new(
                        requires_padding,
                        block_size,
                    ))
                });

                // ensure that append completes before sending batcher message
                queue.append(entry).await;
                batcher_sender.send(BatcherCommand::Batch{adapter_id, queue: queue.clone()}).unwrap();
            }
        }
    }
}

enum BatcherCommand {
    Batch {
        adapter_id: String, 
        queue: Arc<AdapterQueue>
    }
}

async fn batcher_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    generation_health: Arc<AtomicBool>,
    batcher_receiver: flume::Receiver<BatcherCommand>,
) {
    while let Ok(cmd) = batcher_receiver.recv_async().await {
        match cmd {
            BatcherCommand::Batch {
                adapter_id,
                queue,
            } => {
                println!("ASDFASDF top of batcher_task: {}", adapter_id);
                loop {
                    if queue.is_empty().await {
                        println!("ASDFASDF batcher_task: queue is empty");
                        break;
                    }
                    if let Some((mut entries, batch, span)) = queue
                        .next_batch(None, max_batch_prefill_tokens, max_batch_total_tokens)
                        .await
                    {
                        println!("ASDFASDF batcher_task: next_batch called");
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
                            metrics::gauge!("tgi_batch_current_size", batch_size as f64);
                            metrics::gauge!("tgi_batch_current_max_tokens", batch_max_tokens as f64);

                            let min_size = if waiting_tokens >= max_waiting_tokens {
                                // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                                // to add a new batch even though its size might be small
                                None
                            } else {
                                // Minimum batch size
                                Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                            };

                            let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);

                            // Try to get a new batch
                            if let Some((mut new_entries, new_batch, span)) = queue
                                .next_batch(min_size, max_batch_prefill_tokens, token_budget)
                                .await
                            {
                                // Tracking metrics
                                if min_size.is_some() {
                                    metrics::increment_counter!("tgi_batch_concat", "reason" => "backpressure");
                                } else {
                                    metrics::increment_counter!("tgi_batch_concat", "reason" => "wait_exceeded");
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
                        metrics::gauge!("tgi_batch_current_size", 0.0);
                        metrics::gauge!("tgi_batch_current_max_tokens", 0.0);
                    }
                }
            }
        }
    }           
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
        requires_padding: bool,
        generation_health: Arc<AtomicBool>,
    ) -> Self {
        // Infer shared state
        // let queue = Queue::new(requires_padding, 16);
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        // Spawn batching background task that contains all the inference logic
        // tokio::spawn(batching_task(
        //     client,
        //     waiting_served_ratio,
        //     max_batch_prefill_tokens,
        //     max_batch_total_tokens,
        //     max_waiting_tokens,
        //     queue.clone(),
        //     shared.clone(),
        //     generation_health,
        // ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        let (batcher_sender, batcher_receiver) = flume::unbounded();
        let adapter_manager = AdapterManager::new(requires_padding, 16, batcher_sender);
        // receives adapter queues from the adapter manager and batches them
        tokio::spawn(batcher_task(
            client,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            generation_health,
            batcher_receiver,
        ));


        Self {
            validation,
            adapter_manager,
            shared,
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
                metrics::increment_counter!("tgi_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        let adapter_id = request.parameters.adapter_id.clone().unwrap_or("__base_model__".to_string());

        // Validate request
        let valid_request = self.validation.validate(request).await.map_err(|err| {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            err
        })?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = flume::unbounded();

        self.adapter_manager.process(adapter_id, Entry {
            request: valid_request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
        });

        // // Append the request to the queue
        // self.queue.append(Entry {
        //     request: valid_request,
        //     response_tx,
        //     span: Span::current(),
        //     temp_span: None,
        //     queue_time: Instant::now(),
        //     batch_time: None,
        // });

        // Notify the background task that we have a new entry in the queue that needs
        // to be batched
        // self.shared.batching_task.notify_one();

        // Return stream
        Ok((permit, response_rx.into_stream()))
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
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

        // Iterate on stream
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill(tokens) => {
                    // Create Token objects
                    // We do that here instead of in the Python code as Rust for loops are faster
                    result_prefill = tokens
                        .ids
                        .into_iter()
                        .zip(tokens.logprobs.into_iter())
                        .zip(tokens.texts.into_iter())
                        .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
                        .collect();
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
                generated_text,
                queued,
                start,
            })
        } else {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
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
/// Batches requests and sends them to the inference server
#[allow(clippy::too_many_arguments)]
async fn batching_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    queue: Queue,
    shared: Arc<Shared>,
    generation_health: Arc<AtomicBool>,
) {
    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        shared.batching_task.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) = queue
            .next_batch(None, max_batch_prefill_tokens, max_batch_total_tokens)
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
                metrics::gauge!("tgi_batch_current_size", batch_size as f64);
                metrics::gauge!("tgi_batch_current_max_tokens", batch_max_tokens as f64);

                let min_size = if waiting_tokens >= max_waiting_tokens {
                    // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                    // to add a new batch even though its size might be small
                    None
                } else {
                    // Minimum batch size
                    Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                };

                let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);

                // Try to get a new batch
                if let Some((mut new_entries, new_batch, span)) = queue
                    .next_batch(min_size, max_batch_prefill_tokens, token_budget)
                    .await
                {
                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::increment_counter!("tgi_batch_concat", "reason" => "backpressure");
                    } else {
                        metrics::increment_counter!("tgi_batch_concat", "reason" => "wait_exceeded");
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
            metrics::gauge!("tgi_batch_current_size", 0.0);
            metrics::gauge!("tgi_batch_current_max_tokens", 0.0);
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
    metrics::increment_counter!("tgi_batch_inference_count", "method" => "prefill");

    match client.prefill(batch).await {
        Ok((generations, next_batch)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "prefill");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "prefill");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            generation_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "prefill");
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
    metrics::increment_counter!("tgi_batch_inference_count", "method" => "decode");

    match client.decode(batches).await {
        Ok((generations, next_batch)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "decode");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "decode");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            generation_health.store(false, Ordering::SeqCst);
            for id in batch_ids {
                let _ = client.clear_cache(Some(id)).await;
            }
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "decode");
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

            metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
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

    if let Some(prefill_tokens) = generation.prefill_tokens {
        // Send message
        entry.response_tx.send_timeout(
            Ok(InferStreamResponse::Prefill(prefill_tokens)),
            Duration::from_millis(10),
        )?;
    }

    // Create last Token
    let token = Token {
        id: generation.token_id,
        text: generation.token_text,
        logprob: generation.token_logprob,
        special: generation.token_is_special,
    };

    if let Some(generated_text) = generation.generated_text {
        // Generation has ended
        stopped = true;
        // Send message
        entry.response_tx.send_timeout(
            Ok(InferStreamResponse::End {
                token,
                generated_text,
                queued: entry.queue_time,
                start: entry.batch_time.unwrap(),
            }),
            Duration::from_millis(10),
        )?;
    } else {
        // Send message
        entry.response_tx.send_timeout(
            Ok(InferStreamResponse::Token(token)),
            Duration::from_millis(10),
        )?;
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
        metrics::increment_counter!("tgi_request_failure", "err" => "generation");
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
    Prefill(PrefillTokens),
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
