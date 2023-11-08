use crate::{Entry, AdapterLoader, adapter::Adapter, queue::{AdapterEvent, AdapterQueuesState}};
use std::{collections::{HashMap, HashSet}, sync::{Arc, Mutex}, cmp::min};
use nohash_hasher::{IntMap, BuildNoHashHasher};
use text_generation_client::{ShardedClient, Batch, Request};
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{info_span, Span, instrument};


enum AdapterSchedulerCommand {
    Append(Adapter, Entry),
    RemoveErroredAdapters {},
    NextBatch {
        adapters_in_use: HashSet<Adapter>,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[derive(Clone)]
pub(crate) struct AdapterScheduler {
    sender: flume::Sender<AdapterSchedulerCommand>,
}

impl AdapterScheduler {
    pub(crate) fn new(
        client: ShardedClient,
        adapter_event: Arc<AdapterEvent>,
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        let (sender, receiver) = flume::unbounded();

        // receives requests from the infer struct and sends them to the appropriate adapter queue
        tokio::spawn(adapter_scheduler_task(
            client,
            adapter_event,
            requires_padding,
            block_size,
            window_size,
            receiver,
        ));

        Self {
            sender,
        }
    }

    pub(crate) fn process(&self, adapter: Adapter, entry: Entry) {
        // only blocks until the message is sent
        // the adapter manager task will handle the actual processing
        self.sender.send(AdapterSchedulerCommand::Append(adapter, entry)).unwrap();
    }

    pub(crate) async fn remove_errored_adapters(&self) {
        self.sender
            .send(AdapterSchedulerCommand::RemoveErroredAdapters {})
            .unwrap();
    }

    // Get the next batch
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        adapters_in_use: HashSet<Adapter>,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.sender
            .send(AdapterSchedulerCommand::NextBatch {
                adapters_in_use,
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

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

/// Background task that manages the queues of the various adapters
/// TODO(geoffrey): add tracing (span object) to the various commands
async fn adapter_scheduler_task(
    client: ShardedClient,
    adapter_event: Arc<AdapterEvent>,
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    receiver: flume::Receiver<AdapterSchedulerCommand>,
) {
    let mut state = AdapterSchedulerState::new(client, requires_padding, block_size, window_size);

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterSchedulerCommand::Append(adapter, entry) => {
                state.append(adapter, adapter_event.clone(), entry);
            }
            AdapterSchedulerCommand::RemoveErroredAdapters {} => {
                state.remove_errored_adapters();
            },
            AdapterSchedulerCommand::NextBatch {
                adapters_in_use,
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(&adapters_in_use, min_size, prefill_token_budget, token_budget);
                response_sender.send(next_batch).unwrap();
            }),
        }
    }
}


/// Scheduler State
#[derive(Debug)]
struct AdapterSchedulerState {
    /// State for the adapter queues
    queues_state: Arc<Mutex<AdapterQueuesState>>,

    /// Async adapter loader
    loader: AdapterLoader,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,
}

impl AdapterSchedulerState {
    fn new(client: ShardedClient, requires_padding: bool, block_size: u32, window_size: Option<u32>) -> Self {
        let queues_state = Arc::new(Mutex::new(AdapterQueuesState::new()));
        let loader = AdapterLoader::new(client.clone());

        Self {
            queues_state,
            loader,
            next_batch_id: 0,
            requires_padding,
            block_size,
            window_size,
        }
    }

    /// Append entry to the appropriate queue
    fn append(&mut self, adapter: Adapter, adapter_event: Arc<AdapterEvent>, entry: Entry) {
        // check if queue_map has adapter_key as key
        // if not, then add a new Queue and download the adapter
        let mut queues_state = self.queues_state.lock().unwrap();

        let download = queues_state.append(adapter.clone(), adapter_event.clone(), entry);
        if download {
            // Download the adapter async
            self.loader.download_adapter(adapter.clone(), self.queues_state.clone());
        }

        adapter_event.batching_task.notify_one();
    }

    /// Remove any queues that are in an errored state
    fn remove_errored_adapters(&mut self) {
        let mut queues_state = self.queues_state.lock().unwrap();
        let errored_adapters = queues_state.get_errored_adapters();
        for adapter in errored_adapters {
            // Start async offload process
            self.loader.terminate(adapter.clone(), self.queues_state.clone());
        }
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        adapters_in_use: &HashSet<Adapter>,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        let queues_state = &mut self.queues_state.lock().unwrap();

        let num_entries = queues_state.len();
        if num_entries == 0 {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if num_entries < min_size {
                return None;
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(num_entries);
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(num_entries, BuildNoHashHasher::default());
        
        let mut index_to_adapter = HashMap::with_capacity(queues_state.active_len());

        let mut max_input_length = 0;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;

        // Update adapters
        let loader = &mut self.loader;
        update_adapters(queues_state, loader, adapters_in_use, self.queues_state.clone());

        // Pop entries starting from the front of the queue
        while let Some((id, mut entry, adapter)) = queues_state.next_entry() {
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
                let max_new_tokens = match self.window_size {
                    None => entry.request.stopping_parameters.max_new_tokens,
                    Some(window_size) => min(
                        window_size.saturating_sub(entry.request.input_length),
                        entry.request.stopping_parameters.max_new_tokens,
                    ),
                };

                // pad to block size
                decode_tokens +=
                    ((max_new_tokens + self.block_size - 1) / self.block_size) * self.block_size;
            }

            if prefill_tokens > prefill_token_budget
                || (prefill_tokens + decode_tokens) > token_budget
            {
                // Entry is over budget
                // Add it back to the front
                queues_state.push_front(&adapter, id, entry);
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
                adapter_index: adapter.index(),
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
            // Map from adapter index back to queue in case we need to add back entries below
            // let queue = queue_map.get_mut(&adapter).unwrap();
            index_to_adapter.insert(adapter.index(), adapter);
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
                    let adapter_index = r.adapter_index;
                    let adapter = index_to_adapter.get_mut(&adapter_index).unwrap();
                    queues_state.push_front(adapter, id, entry);
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


fn update_adapters(
    queues_state: &mut AdapterQueuesState,
    loader: &mut AdapterLoader,
    adapters_in_use: &HashSet<Adapter>,
    shared_state: Arc<Mutex<AdapterQueuesState>>,
) {
    let errored_adapters = queues_state.get_errored_adapters();
    for adapter in errored_adapters {
        // Start async offload process
        loader.terminate(adapter.clone(), shared_state.clone());
        queues_state.untrack(&adapter);
    }

    let (offload_adapters, load_adapters) = queues_state.update_adapters(adapters_in_use);
    
    // Background task to offload and load adapters
    for adapter in offload_adapters {
        loader.offload_adapter(adapter.clone(), shared_state.clone());
    }
    for adapter in load_adapters {
        loader.load_adapter(adapter.clone(), shared_state.clone());
    }
}
