use crate::{Entry, Queue, adapter::Adapter};
use std::{collections::{HashSet, HashMap, VecDeque}, sync::Arc};
use nohash_hasher::{IntMap, BuildNoHashHasher};
use text_generation_client::{ShardedClient, Batch, Request};
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{info_span, Span, instrument};


enum AdapterManagerCommand {
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
pub(crate) struct AdapterManager {
    sender: flume::Sender<AdapterManagerCommand>,
}

impl AdapterManager {
    pub(crate) fn new(
        client: ShardedClient,
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        let (sender, receiver) = flume::unbounded();

        // receives requests from the infer struct and sends them to the appropriate adapter queue
        tokio::spawn(adapter_manager_task(
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
        self.sender.send(AdapterManagerCommand::Append(adapter, entry)).unwrap();
    }

    pub(crate) async fn remove_queue(&self, adapter: Adapter) {
        self.sender
            .send(AdapterManagerCommand::RemoveQueue {
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
            .send(AdapterManagerCommand::NextBatch {
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
async fn adapter_manager_task(
    client: ShardedClient,
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    receiver: flume::Receiver<AdapterManagerCommand>,
) {
    let mut state = AdapterManagerState::new(client, requires_padding, block_size, window_size);

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterManagerCommand::Append(adapter, entry) => {
                state.append(adapter, entry);
            }
            AdapterManagerCommand::RemoveQueue {
                adapter
            } => {
                state.remove_queue()
            },
            AdapterManagerCommand::NextBatch {
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


/// Queue State
#[derive(Debug)]
struct AdapterManagerState {
    /// Sharded client
    client: ShardedClient,

    /// Map of adapter key to queue
    queue_map: HashMap<String, Arc<Queue>>,

    /// Adapters that are currently not in use
    inactive_adapters: VecDeque<String>,

    /// Adapters that are currently in use
    active_adapters: VecDeque<String>,

    /// Map of adapter key to the time of the oldest entry in its queue, or None if last empty
    adapter_oldest_entries: HashMap<String, Option<Instant>>,

    /// Number of adapters that can be active at a time
    max_active_adapters: usize,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,
}

impl AdapterManagerState {
    fn new(client: ShardedClient, requires_padding: bool, block_size: u32, window_size: Option<u32>) -> Self {
        let mut queue_map: HashMap<String, Arc<Queue>> = HashMap::new();
        let mut inactive_adapters: VecDeque<String> = VecDeque::new();
        let mut active_adapters: VecDeque<String> = VecDeque::new();
        let mut adapter_oldest_entries: HashMap<String, Option<Instant>> = HashMap::new();

        Self {
            client,
            queue_map,
            inactive_adapters,
            active_adapters,
            adapter_oldest_entries,
            max_active_adapters: 3,
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
        let queue;
        let adapter_key = adapter.as_string();
        if !self.queue_map.contains_key(&adapter_key) {
            queue = Arc::new(Queue::new(
                adapter.clone(),
                self.client.clone(),
                self.requires_padding,
                self.block_size,
                self.window_size,
            ));
            self.queue_map.insert(adapter_key.clone(), queue.clone());

            // add the adapter to the active set if we're below the limit, otherwise
            // add it to the inactive set
            if self.active_adapters.len() < self.max_active_adapters {
                self.active_adapters.push_back(adapter_key.clone());
            } else {
                self.inactive_adapters.push_back(adapter_key.clone());
            }
            self.adapter_oldest_entries.insert(adapter_key.clone(), None);
        } else {
            queue = self.queue_map.get(&adapter_key).unwrap().clone();
        }

        // ensure that append completes before sending batcher message
        queue.append(entry);
    }

    /// Remove queue
    async fn remove_queue(&mut self, adapter: Adapter) {
        let adapter_key = adapter.as_string();
        let queue = self.queue_map.get(&adapter_key).unwrap().clone();
        queue.terminate().await;
        self.queue_map.remove(&adapter_key);
        self.active_adapters.retain(|id| id != &adapter_key);
        self.inactive_adapters.retain(|id| id != &adapter_key);
        self.adapter_oldest_entries.remove(&adapter_key);
    }

    fn update_queue_ages(&mut self) {
        for queue in self.queue_map.values() {
            queue.update_age();
        }
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
                adapter_index: self.adapter_index,
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
