use crate::{Entry, AdapterLoader, adapter::Adapter, queue::{QueueState, self}};
use std::{collections::{HashMap, VecDeque, HashSet}, sync::{Arc, Mutex}, time::Duration};
use nohash_hasher::{IntMap, BuildNoHashHasher};
use text_generation_client::{ShardedClient, Batch, Request};
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{info_span, Span, instrument};


enum AdapterSchedulerCommand {
    Append(Adapter, Entry),
    RemoveQueue {
        adapter: Adapter,
    },
    NextBatch {
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
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        let (sender, receiver) = flume::unbounded();

        // receives requests from the infer struct and sends them to the appropriate adapter queue
        tokio::spawn(adapter_scheduler_task(
            client,
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

    pub(crate) async fn remove_queue(&self, adapter: Adapter) {
        self.sender
            .send(AdapterSchedulerCommand::RemoveQueue {
                adapter
            })
            .unwrap();
    }

    // Get the next batch
    #[instrument(skip(self))]
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
            .send(AdapterSchedulerCommand::NextBatch {
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
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    receiver: flume::Receiver<AdapterSchedulerCommand>,
) {
    let mut state = AdapterSchedulerState::new(client, requires_padding, block_size, window_size);

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterSchedulerCommand::Append(adapter, entry) => {
                state.append(adapter, entry);
            }
            AdapterSchedulerCommand::RemoveQueue {
                adapter
            } => {
                state.remove_queue(adapter);
            },
            AdapterSchedulerCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(min_size, prefill_token_budget, token_budget);
                response_sender.send(next_batch).unwrap();
            }),
        }
    }
}


/// Scheduler State
#[derive(Debug)]
struct AdapterSchedulerState {
    /// Sharded client
    client: ShardedClient,

    /// Async adapter loader
    loader: AdapterLoader,

    /// Map of adapter key to queue
    queue_map: Arc<Mutex<HashMap<Adapter, QueueState>>>,

    /// Adapters that are currently not in use, but have entries in queue
    pending_adapters: VecDeque<Adapter>,

    /// Adapters that are currently in use
    active_adapters: VecDeque<Adapter>,

    /// Number of adapters that can be active at a time
    max_active_adapters: usize,

    /// Maximum time an adapter is allowed to be active before exchanging out
    max_active_time: Duration,

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
        let mut queue_map = Arc::new(Mutex::new(HashMap::new()));
        let mut pending_adapters = VecDeque::new();
        let mut active_adapters = VecDeque::new();
        let mut loader = AdapterLoader::new(client);

        Self {
            client,
            loader,
            queue_map,
            pending_adapters,
            active_adapters,
            max_active_adapters: 3,
            max_active_time: Duration::from_secs(2),
            next_batch_id: 0,
            requires_padding,
            block_size,
            window_size,
        }
    }

    /// Append entry to the appropriate queue
    fn append(&mut self, adapter: Adapter, entry: Entry) {
        // check if queue_map has adapter_key as key
        // if not, then add a new Queue and download the adapter
        let mut queue_map = self.queue_map.lock().unwrap();

        if !queue_map.contains_key(&adapter) {
            queue_map.insert(adapter.clone(), QueueState::new(adapter.clone()));

            self.pending_adapters.push_back(adapter.clone());

            // Download the adapter async
            self.loader.download_adapter(adapter.clone(), self.queue_map);
        }

        // ensure that append completes before sending batcher message
        let queue = queue_map.get_mut(&adapter).unwrap();
        queue.append(entry);
    }

    /// Remove queue
    fn remove_queue(&mut self, adapter: Adapter) {
        let mut queue_map = self.queue_map.lock().unwrap();

        queue_map.remove(&adapter);
        self.active_adapters.retain(|id| id != &adapter);
        self.pending_adapters.retain(|id| id != &adapter);
    }

    /// Updates the mapping from adapter to the age of its oldest entry, then returns the oldest active adapter
    fn get_oldest_active_adapter(&mut self, queue_map: &HashMap<Adapter, QueueState>) -> Option<Adapter> {
        let mut oldest_timestamp = Instant::now();
        let mut oldest_adapter = None;
        for adapter in self.active_adapters.iter() {
            let queue = queue_map.get(adapter).unwrap().clone();
            if !queue.entries().is_empty() {
                // queue is not empty, so it must have an oldest entry
                let adapter_oldest_ts = queue.peek().unwrap();
                if adapter_oldest_ts < oldest_timestamp {
                    oldest_timestamp = adapter_oldest_ts;
                    oldest_adapter = Some(adapter.clone());
                }
            }
        }
        oldest_adapter
    }

    fn update_adapters(
        &mut self, 
        adapters_in_use: &HashSet<Adapter>, 
        queue_map: &HashMap<Adapter, QueueState>,
    ) {
        // Mark any active adapters that are Idle (have no active or pending requests) for removal
        // Additionally, move any adapters that have been activate over the limit to pending
        let now = Instant::now();
        let mut adapters_to_remove = HashSet::new();
        let mut adapters_to_pending = Vec::new();
        for adapter in self.active_adapters.iter() {
            let queue = queue_map.get(adapter).unwrap().clone();
            if adapters_in_use.contains(&queue.adapter()) {
                // Cannot modify active adapters that are in use
                continue
            }

            if self.pending_adapters.len() <= adapters_to_remove.len() + adapters_to_pending.len() {
                // Only move adapters out of active if we have pending adapters ready to take their place
                continue
            }

            if queue.entries().is_empty() {
                // queue is empty and not in use, so move to removal set
                adapters_to_remove.insert(adapter.clone());
            } else if now.duration_since(queue.activation_ts().unwrap()) > self.max_active_time {
                // queue has been active for more than the allowed time limit, so move to pending queue
                adapters_to_pending.push(adapter.clone());
            }
        }

        // Remove all adapters in the remove set
        self.active_adapters.retain(|adapter| {
            !adapters_to_remove.contains(adapter)
        });

        // Add all adapters in the pending set
        for adapter in adapters_to_pending.iter() {
            self.pending_adapters.push_back(adapter.clone());
        }

        // Add pending adapters to the active set until we reach the max
        while self.active_adapters.len() < self.max_active_adapters && self.pending_adapters.len() > 0 {
            let adapter = self.pending_adapters.pop_front().unwrap();

            // Update activation timestamp
            let mut queue = queue_map.get_mut(&adapter).unwrap();
            queue.set_activation_ts(now);

            // Start async loading process
            self.loader.load_adapter(adapter.clone(), self.queue_map);

            self.active_adapters.push_back(adapter.clone());
        }
    }

    fn next_entry(
        &mut self, 
        adapters_in_use: &HashSet<Adapter>, 
        queue_map: &HashMap<Adapter, QueueState>,
    ) -> Option<(u64, Entry, QueueState)> {
        // Update the queues of pending and active adapters based on the current state
        self.update_adapters(adapters_in_use, queue_map);

        // Get the adapter from the active set that has been waiting the longest.
        let adapter = self.get_oldest_active_adapter(queue_map);
        if adapter.is_none() {
            // No active adapter has any entries
            return None;
        }

        // TODO(travis): update adapters to remove nones and cycle in other adapters

        // Pop the oldest entry from the queue
        let adapter_key = adapter.unwrap();
        let queue = queue_map.get(&adapter_key).unwrap().clone();
        let (id, entry, next_oldest_entry) = queue.pop().unwrap();
        self.adapter_oldest_entries.insert(adapter_key.clone(), next_oldest_entry);
        Some((id, entry, *queue))
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        adapters_in_use: &HashSet<Adapter>,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        let mut queue_map = self.queue_map.lock().unwrap();

        let num_entries = queue_map.values().map(|queue| queue.entries().len()).sum();
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
        
        let mut adapter_index_to_queue = HashMap::with_capacity(self.active_adapters.len());

        let mut max_input_length = 0;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;

        // Pop entries starting from the front of the queue
        while let Some((id, mut entry, queue)) = self.next_entry(adapters_in_use, &queue_map) {
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
                queue.entries().push_front((id, entry));
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
                adapter_index: queue.adapter().index(),
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
            // Map from adapter index back to queue in case we need to add back entries below
            adapter_index_to_queue.insert(queue.adapter().index(), queue);
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
                    let queue = adapter_index_to_queue.get(&adapter_index).unwrap();
                    queue.entries().push_front((id, entry));
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
