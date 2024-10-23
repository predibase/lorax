use crate::{
    adapter::Adapter,
    batch::{BatchEntries, Entry},
    block_allocator::BlockAllocator,
    queue::{AdapterEvent, AdapterQueuesState},
    AdapterLoader,
};
use lorax_client::{Batch, ShardedClient};
use std::{cmp::max, collections::HashSet, sync::Arc};
use tokio::sync::{oneshot, Mutex};
use tracing::{info_span, instrument, Instrument, Span};

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
        max_active_adapters: usize,
        adapter_cycle_time_s: u64,
        speculate: u32,
        max_batch_total_tokens: u32,
        prefix_caching: bool,
        chunked_prefill: bool,
        is_causal_lm: bool,
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
            max_active_adapters,
            adapter_cycle_time_s,
            speculate,
            max_batch_total_tokens,
            prefix_caching,
            chunked_prefill,
            is_causal_lm,
        ));

        Self { sender }
    }

    pub(crate) fn process(&self, adapter: Adapter, entry: Entry) {
        // only blocks until the message is sent
        // the adapter manager task will handle the actual processing
        self.sender
            .send(AdapterSchedulerCommand::Append(adapter, entry))
            .unwrap();
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

type NextBatch = (Box<dyn BatchEntries>, Batch, Span);

/// Background task that manages the queues of the various adapters
/// TODO(geoffrey): add tracing (span object) to the various commands
async fn adapter_scheduler_task(
    client: ShardedClient,
    adapter_event: Arc<AdapterEvent>,
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    receiver: flume::Receiver<AdapterSchedulerCommand>,
    max_active_adapters: usize,
    adapter_cycle_time_s: u64,
    speculate: u32,
    max_batch_total_tokens: u32,
    prefix_caching: bool,
    chunked_prefill: bool,
    is_causal_lm: bool,
) {
    let mut state = AdapterSchedulerState::new(
        client,
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

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterSchedulerCommand::Append(adapter, entry) => {
                state.append(adapter, adapter_event.clone(), entry).await;
            }
            AdapterSchedulerCommand::RemoveErroredAdapters {} => {
                state.remove_errored_adapters().await;
            }
            AdapterSchedulerCommand::NextBatch {
                adapters_in_use,
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => {
                let next_batch = state
                    .next_batch(
                        &adapters_in_use,
                        min_size,
                        prefill_token_budget,
                        token_budget,
                    )
                    .instrument(span)
                    .await;
                response_sender.send(next_batch).unwrap();
            }
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

    #[allow(dead_code)] // currently unused
    /// Whether the model is using padding
    requires_padding: bool,

    #[allow(dead_code)] // currently unused
    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    // window_size: Option<u32>,

    /// Speculation amount
    speculate: u32,

    /// Chunked prefill
    chunked_prefill: bool,

    /// Paged Attention Block Allocation
    block_allocator: Option<BlockAllocator>,
}

impl AdapterSchedulerState {
    fn new(
        client: ShardedClient,
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
        max_active_adapters: usize,
        adapter_cycle_time_s: u64,
        speculate: u32,
        max_batch_total_tokens: u32,
        prefix_caching: bool,
        chunked_prefill: bool,
        is_causal_lm: bool,
    ) -> Self {
        let queues_state = Arc::new(Mutex::new(AdapterQueuesState::new(
            max_active_adapters,
            adapter_cycle_time_s,
        )));
        let loader = AdapterLoader::new(client.clone());

        // Only causal LMs require the block allocator, due to paged attention
        let block_allocator = (!requires_padding && is_causal_lm).then(|| {
            BlockAllocator::new(
                max_batch_total_tokens,
                block_size,
                prefix_caching,
                window_size,
            )
        });

        Self {
            queues_state,
            loader,
            next_batch_id: 0,
            requires_padding,
            block_size,
            // window_size,
            speculate,
            chunked_prefill,
            block_allocator,
        }
    }

    /// Append entry to the appropriate queue
    async fn append(&mut self, adapter: Adapter, adapter_event: Arc<AdapterEvent>, entry: Entry) {
        // check if queue_map has adapter_key as key
        // if not, then add a new Queue and download the adapter
        let mut queues_state = self.queues_state.lock().await;

        let download = queues_state.append(adapter.clone(), adapter_event.clone(), entry);
        if download {
            // Download the adapter async
            self.loader
                .download_adapter(adapter.clone(), self.queues_state.clone());
        }

        adapter_event.batching_task.notify_one();
    }

    /// Remove any queues that are in an errored state
    async fn remove_errored_adapters(&mut self) {
        let mut queues_state = self.queues_state.lock().await;
        let errored_adapters = queues_state.get_errored_adapters();
        for adapter in errored_adapters {
            // Start async offload process
            self.loader
                .terminate(adapter.clone(), self.queues_state.clone());
        }
    }

    async fn next_entry(&mut self) -> Option<(u64, Entry, Adapter)> {
        self.queues_state.lock().await.next_entry()
    }

    // Get the next batch
    async fn next_batch(
        &mut self,
        adapters_in_use: &HashSet<Adapter>,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if prefill_token_budget == 0 || token_budget == 0 {
            return None;
        };

        let num_entries = self.queues_state.lock().await.len();
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

        let mut max_input_length = 0;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;
        let mut max_blocks = 0;

        // Update adapters
        {
            let queues_state = &mut self.queues_state.lock().await;

            let loader = &mut self.loader;
            update_adapters(
                queues_state,
                loader,
                adapters_in_use,
                self.queues_state.clone(),
            );
        }

        // Pop entries starting from the front of the queue
        let mut batch_entries: Option<Box<dyn BatchEntries>> = None;
        'entry_loop: while let Some((id, mut entry, adapter)) = self.next_entry().await {
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_disconnected() {
                metrics::increment_counter!("lorax_request_failure", "err" => "dropped");
                continue;
            }

            let mut batch_requests_len = 0;
            if let Some(batch_entries) = batch_entries.as_ref() {
                batch_requests_len = batch_entries.len();
            }

            let mut should_break = false;
            let mut chunk_len = None;
            let block_allocation = match &self.block_allocator {
                None => {
                    // We pad to max input length in the Python shards
                    // We need to take these padding tokens into the equation
                    max_input_length = max_input_length.max(entry.request.input_length());
                    prefill_tokens = (batch_requests_len + 1) as u32 * max_input_length;

                    decode_tokens += entry.request.max_new_tokens();
                    let total_tokens = prefill_tokens + decode_tokens + self.speculate;

                    if prefill_tokens > prefill_token_budget || total_tokens > token_budget {
                        // Entry is over budget
                        // Add it back to the front
                        tracing::debug!("Over budget: prefill_tokens={prefill_tokens} > {prefill_token_budget} || {prefill_tokens} + {decode_tokens} + {} > {token_budget}", self.speculate);
                        self.queues_state
                            .lock()
                            .await
                            .push_front(&adapter, id, entry);
                        break 'entry_loop;
                    }
                    None
                }
                Some(block_allocator) => {
                    let tokens = entry.request.input_length()
                        + entry.request.max_new_tokens()
                        + self.speculate
                        - 1;

                    tracing::trace!(
                        "Scheduling {} tokens ({} input, {} output, {} speculate)",
                        tokens,
                        entry.request.input_length(),
                        entry.request.max_new_tokens(),
                        self.speculate
                    );

                    let block_allocation = match block_allocator
                        .allocate(adapter.index(), tokens, entry.request.input_ids())
                        .await
                    {
                        None => {
                            // Entry is over budget
                            // Add it back to the front
                            tracing::debug!("Over budget: not enough free blocks");
                            self.queues_state
                                .lock()
                                .await
                                .push_front(&adapter, id, entry);
                            break 'entry_loop;
                        }
                        Some(mut block_allocation) => {
                            tracing::debug!("Allocation: {block_allocation:?}");
                            max_blocks = max(max_blocks, block_allocation.blocks.len() as u32);

                            if block_allocation.prefix_len == entry.request.input_length() {
                                // The whole request was found in the radix trie
                                // However, for the transformer forward to work, we need to
                                // have at least one token of postfix.
                                block_allocation.prefix_len -= 1;
                            }

                            block_allocation
                        }
                    };

                    let postfix_len = entry.request.input_length() - block_allocation.prefix_len;
                    if prefill_tokens + postfix_len > prefill_token_budget {
                        // Entry is over budget
                        if self.chunked_prefill {
                            // We support chunking, just set postfix_len to exactly match prefill_token_budget
                            let entry_chunk_len =
                                prefill_token_budget.saturating_sub(prefill_tokens);
                            if entry_chunk_len > 0 {
                                chunk_len = Some(entry_chunk_len);
                            } else {
                                // We cannot prefill even one token for this entry
                                // Add it back to the queue
                                self.queues_state
                                    .lock()
                                    .await
                                    .push_front(&adapter, id, entry);
                                break 'entry_loop;
                            }
                            tracing::debug!(
                                "Matched budget: prefill_tokens={} == {prefill_token_budget}",
                                prefill_tokens + postfix_len
                            );
                            should_break = true;
                        } else {
                            // We don't support chunking, this entry needs to go back to the buffer
                            // Add it back to the front
                            tracing::debug!(
                                "Over budget: prefill_tokens={} > {prefill_token_budget}",
                                prefill_tokens + postfix_len
                            );
                            self.queues_state
                                .lock()
                                .await
                                .push_front(&adapter, id, entry);
                            break 'entry_loop;
                        }
                    }

                    prefill_tokens += postfix_len;

                    Some(block_allocation)
                }
            };

            if batch_entries.is_none() {
                batch_entries = Some(
                    entry
                        .request
                        .to_batch(num_entries, self.queues_state.lock().await.active_len()),
                );
            }

            if !batch_entries.as_ref().unwrap().can_add(&entry) {
                // Incompatible entry for this batch. Reinsert and break
                self.queues_state
                    .lock()
                    .await
                    .push_front(&adapter, id, entry);
                break 'entry_loop;
            }

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            let (blocks, slots, prefix_len) = match &block_allocation {
                None => (Vec::new(), Vec::new(), 0),
                Some(block_allocation) => (
                    block_allocation.blocks.clone(),
                    block_allocation.slots.clone(),
                    block_allocation.prefix_len,
                ),
            };

            entry.block_allocation = block_allocation;

            batch_entries
                .as_mut()
                .unwrap()
                .add(id, entry, adapter, blocks, slots, prefix_len, chunk_len);

            if should_break {
                break 'entry_loop;
            }
        }

        if batch_entries.is_none() {
            return None;
        }

        let mut batch_entries = batch_entries.unwrap();

        // Empty batch
        if batch_entries.is_empty() {
            return None;
        }

        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            let queues_state = &mut self.queues_state.lock().await;

            // Batch is too small
            if batch_entries.len() < min_size {
                // Add back entries to the queue in the correct order
                for (adapter, id, entry) in batch_entries.drain() {
                    queues_state.push_front(&adapter, id, entry);
                }

                return None;
            }
        }

        next_batch_span.record("batch_size", batch_entries.len() as u32);
        let max_tokens = prefill_tokens + decode_tokens;
        let batch = batch_entries.create_batch_data(self.next_batch_id, max_tokens, max_blocks);

        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("lorax_batch_next_size", batch_entries.len() as f64);

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
