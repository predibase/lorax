use core::fmt::Debug;
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
};

use async_trait::async_trait;

use lorax_client::{
    Batch, CachedBatch, NextTokenChooserParameters, Request, ShardedClient,
    StoppingCriteriaParameters,
};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use tokio::time::Instant;
use tracing::{info_span, Instrument, Span};

use crate::{
    adapter::Adapter,
    infer::{decode, embed, prefill, InferError, InferStreamResponse},
};

pub(crate) trait ValidRequest: Sync + Send + Debug + Any {
    fn input_length(&self) -> u32;
    fn max_new_tokens(&self) -> u32;
    fn adapter(&self) -> Adapter;
    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Box<dyn BatchEntries>;
    fn as_any(&self) -> &dyn Any;
}

impl ValidRequest for ValidGenerateRequest {
    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        self.stopping_parameters.max_new_tokens
    }

    fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }

    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Box<dyn BatchEntries> {
        Box::new(GenerateBatchEntries::new(num_entries, queue_len))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub(crate) struct ValidEmbedRequest {
    pub inputs: String,
    pub input_length: u32,
    pub adapter: Adapter,
}

impl ValidRequest for ValidEmbedRequest {
    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        1
    }

    fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }

    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Box<dyn BatchEntries> {
        Box::new(EmbedBatchEntries::new(num_entries, queue_len))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub adapter: Adapter,
    pub apply_chat_template: bool,
}

/// AdapterLoader entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: Arc<dyn ValidRequest>,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: flume::Sender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
}

#[derive(Debug)]
pub(crate) struct BatchEntriesState {
    pub(crate) batch_requests: Vec<Request>,
    pub(crate) batch_entries: HashMap<u64, Entry, BuildNoHashHasher<u64>>,
    pub(crate) index_to_adapter: HashMap<u32, Adapter>,
    next_batch_span: Span,
}

impl BatchEntriesState {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let batch_requests = Vec::with_capacity(num_entries);
        let batch_entries =
            IntMap::with_capacity_and_hasher(num_entries, BuildNoHashHasher::default());

        let index_to_adapter = HashMap::with_capacity(queue_len);

        Self {
            batch_requests,
            batch_entries,
            index_to_adapter,
            next_batch_span,
        }
    }

    fn add(&mut self, id: u64, mut entry: Entry, adapter: Adapter, request: Request) {
        // Create a new span to link the batch back to this entry
        let entry_batch_span = info_span!(parent: &entry.span, "infer");
        // Add relationships
        self.next_batch_span.follows_from(&entry_batch_span);
        entry_batch_span.follows_from(&self.next_batch_span);
        // Update entry
        entry.temp_span = Some(entry_batch_span);

        self.batch_requests.push(request);

        // Set batch_time
        entry.batch_time = Some(Instant::now());
        // Insert in batch_entries IntMap
        self.batch_entries.insert(id, entry);
        // Map from adapter index back to queue in case we need to add back entries below
        // let queue = queue_map.get_mut(&adapter).unwrap();
        self.index_to_adapter.insert(adapter.index(), adapter);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        let mut entries = Vec::with_capacity(self.batch_requests.len());

        for r in self.batch_requests.into_iter().rev() {
            let id = r.id;
            let entry = self.batch_entries.remove(&id).unwrap();
            let adapter_index = r.adapter_index;
            let adapter = self.index_to_adapter.get_mut(&adapter_index).unwrap();
            entries.push((adapter.clone(), id, entry));
        }

        entries
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32) -> Batch {
        // Final batch size
        let size = self.len() as u32;
        self.next_batch_span.record("batch_size", size);

        Batch {
            id: batch_id,
            requests: self.batch_requests,
            size,
            max_tokens,
        }
    }

    fn adapters_in_use(&self) -> HashSet<Adapter> {
        self.batch_entries
            .iter()
            .map(|(_, entry)| entry.request.adapter())
            .collect::<HashSet<_>>()
    }

    fn is_empty(&self) -> bool {
        self.batch_requests.is_empty()
    }

    fn len(&self) -> usize {
        self.batch_requests.len()
    }
}

#[async_trait]
pub(crate) trait BatchEntries: Sync + Send + Debug {
    fn can_add(&self, entry: &Entry) -> bool;
    fn add(&mut self, id: u64, entry: Entry, adapter: Adapter);
    fn extend(&mut self, entries: Box<dyn BatchEntries>);
    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)>;
    fn create_batch_data(&self, batch_id: u64, max_tokens: u32) -> Batch;
    fn adapters_in_use(&self) -> HashSet<Adapter>;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn state(&self) -> &BatchEntriesState;
    fn set_span(&mut self, span: Span);
    fn update_entries_span(&mut self, create_span_fn: Box<dyn Fn(&Span) -> Span>);

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch>;

    async fn process_next(
        &mut self,
        client: &mut ShardedClient,
        batches: Vec<CachedBatch>,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch>;
}

#[derive(Debug)]
pub(crate) struct GenerateBatchEntries {
    pub(crate) state: BatchEntriesState,
}

impl GenerateBatchEntries {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        Self {
            state: BatchEntriesState::new(num_entries, queue_len),
        }
    }
}

#[async_trait]
impl BatchEntries for GenerateBatchEntries {
    fn can_add(&self, entry: &Entry) -> bool {
        // return false if the entry.request is not of type ValidGenerateRequest
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidGenerateRequest>();

        let result = valid_request.is_some();
        result
    }

    fn add(&mut self, id: u64, entry: Entry, adapter: Adapter) {
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidGenerateRequest>();

        let request = valid_request.unwrap();
        let request_proto = Request {
            id,
            prefill_logprobs: request.decoder_input_details,
            inputs: request.inputs.clone(),
            truncate: request.truncate,
            parameters: Some(request.parameters.clone()),
            stopping_parameters: Some(request.stopping_parameters.clone()),
            adapter_index: adapter.index(),
            apply_chat_template: request.apply_chat_template,
        };

        self.state.add(id, entry, adapter, request_proto);
    }

    fn extend(&mut self, entries: Box<dyn BatchEntries>) {
        self.state()
            .batch_entries
            .extend(entries.state().batch_entries);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        self.state.drain()
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32) -> Batch {
        self.state.create_batch_data(batch_id, max_tokens)
    }

    fn adapters_in_use(&self) -> HashSet<Adapter> {
        self.state.adapters_in_use()
    }

    fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    fn len(&self) -> usize {
        self.state.len()
    }

    fn state(&self) -> &BatchEntriesState {
        &self.state
    }

    fn set_span(&mut self, span: Span) {
        self.state.next_batch_span = span;
    }

    fn update_entries_span(&mut self, create_span_fn: Box<dyn Fn(&Span) -> Span>) {
        let state = self.state();
        for (_, entry) in self.state.batch_entries.iter_mut() {
            // Create a new span to link the batch back to this entry
            let entry_batch_span = create_span_fn(&entry.span);
            // Add relationships
            state.next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&state.next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);
        }
    }

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        prefill(
            &mut client,
            batch,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(self.state.next_batch_span)
        .await
    }

    async fn process_next(
        &mut self,
        client: &mut ShardedClient,
        batches: Vec<CachedBatch>,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        decode(
            &mut client,
            batches,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(self.state.next_batch_span)
        .await
    }
}

#[derive(Debug)]
pub(crate) struct EmbedBatchEntries {
    pub(crate) state: BatchEntriesState,
}

impl EmbedBatchEntries {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        Self {
            state: BatchEntriesState::new(num_entries, queue_len),
        }
    }
}

#[async_trait]
impl BatchEntries for EmbedBatchEntries {
    fn can_add(&self, entry: &Entry) -> bool {
        // return false if the entry.request is not of type ValidEmbedRequest
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidEmbedRequest>();

        let result = valid_request.is_some();
        result
    }

    fn add(&mut self, id: u64, entry: Entry, adapter: Adapter) {
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidEmbedRequest>();

        let request = valid_request.unwrap();
        let request_proto = Request {
            id,
            prefill_logprobs: false,
            inputs: request.inputs.clone(),
            truncate: 0,
            parameters: None,
            stopping_parameters: None,
            adapter_index: adapter.index(),
            apply_chat_template: false,
        };

        self.state.add(id, entry, adapter, request_proto);
    }

    fn extend(&mut self, entries: Box<dyn BatchEntries>) {
        self.state()
            .batch_entries
            .extend(entries.state().batch_entries);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        self.state.drain()
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32) -> Batch {
        self.state.create_batch_data(batch_id, max_tokens)
    }

    fn adapters_in_use(&self) -> HashSet<Adapter> {
        self.state.adapters_in_use()
    }

    fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    fn len(&self) -> usize {
        self.state.len()
    }

    fn state(&self) -> &BatchEntriesState {
        &self.state
    }

    fn set_span(&mut self, span: Span) {
        self.state.next_batch_span = span;
    }

    fn update_entries_span(&mut self, create_span_fn: Box<dyn Fn(&Span) -> Span>) {
        let state = self.state();
        for (_, entry) in self.state.batch_entries.iter_mut() {
            // Create a new span to link the batch back to this entry
            let entry_batch_span = create_span_fn(&entry.span);
            // Add relationships
            state.next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&state.next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);
        }
    }

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        embed(
            client,
            batch,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(self.state.next_batch_span)
        .await
    }

    async fn process_next(
        &mut self,
        client: &mut ShardedClient,
        batches: Vec<CachedBatch>,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        // TODO(travis): send error (programming eroor) if we get here
        None
    }
}
