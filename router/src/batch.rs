use core::fmt::Debug;
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
};

use async_trait::async_trait;

use lorax_client::{
    Batch, CachedBatch, NextTokenChooserParameters, Request, ShardedClient,
    StoppingCriteriaParameters, TokenizedInputs,
};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use tokio::{sync::mpsc, time::Instant};
use tracing::{Instrument, Span};

use crate::{
    adapter::Adapter,
    block_allocator::BlockAllocation,
    infer::{classify, decode, embed, prefill, InferError, InferStreamResponse},
};

pub(crate) trait ValidRequest: Sync + Send + Debug + Any {
    fn decoder_input_details(&self) -> bool;
    fn input_length(&self) -> u32;
    fn input_ids(&self) -> Option<Arc<Vec<u32>>>;
    fn max_new_tokens(&self) -> u32;
    fn adapter(&self) -> Adapter;
    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Box<dyn BatchEntries>;
    fn as_any(&self) -> &dyn Any;
}

impl ValidRequest for ValidGenerateRequest {
    fn decoder_input_details(&self) -> bool {
        self.decoder_input_details
    }

    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn input_ids(&self) -> Option<Arc<Vec<u32>>> {
        if let Some(tokenized_inputs) = &self.tokenized_inputs {
            Some(Arc::new(tokenized_inputs.ids.clone()))
        } else {
            None
        }
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
    pub tokenized_inputs: Option<TokenizedInputs>,
    pub input_length: u32,
    pub adapter: Adapter,
}

impl ValidRequest for ValidEmbedRequest {
    fn decoder_input_details(&self) -> bool {
        false
    }

    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn input_ids(&self) -> Option<Arc<Vec<u32>>> {
        if let Some(tokenized_inputs) = &self.tokenized_inputs {
            Some(Arc::new(tokenized_inputs.ids.clone()))
        } else {
            None
        }
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
pub(crate) struct ValidClassifyRequest {
    pub inputs: String,
    pub tokenized_inputs: Option<TokenizedInputs>,
    pub input_length: u32,
    pub adapter: Adapter,
}

impl ValidRequest for ValidClassifyRequest {
    fn decoder_input_details(&self) -> bool {
        false
    }

    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn input_ids(&self) -> Option<Arc<Vec<u32>>> {
        if let Some(tokenized_inputs) = &self.tokenized_inputs {
            Some(Arc::new(tokenized_inputs.ids.clone()))
        } else {
            None
        }
    }

    fn max_new_tokens(&self) -> u32 {
        1
    }

    fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }

    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Box<dyn BatchEntries> {
        Box::new(ClassifyBatchEntries::new(num_entries, queue_len))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub tokenized_inputs: Option<TokenizedInputs>,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub adapter: Adapter,
}

/// AdapterLoader entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: Arc<dyn ValidRequest>,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Block Allocation
    pub block_allocation: Option<BlockAllocation>,
    /// Optional entry id
    pub id: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct BatchEntriesState {
    pub(crate) batch_requests: Vec<Request>,
    pub(crate) batch_entries: HashMap<u64, Entry, BuildNoHashHasher<u64>>,
    pub(crate) index_to_adapter: HashMap<u32, Adapter>,
}

impl BatchEntriesState {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        let batch_requests = Vec::with_capacity(num_entries);
        let batch_entries =
            IntMap::with_capacity_and_hasher(num_entries, BuildNoHashHasher::default());

        let index_to_adapter = HashMap::with_capacity(queue_len);

        Self {
            batch_requests,
            batch_entries,
            index_to_adapter,
        }
    }

    fn add(&mut self, id: u64, mut entry: Entry, adapter: Adapter, request: Request) {
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

        // TODO(travis): clone is not ideal, find a way to do this cleanly in place
        for r in self.batch_requests.clone().into_iter().rev() {
            let id = r.id;
            let entry = self.batch_entries.remove(&id).unwrap();
            let adapter_index = r.adapter_index;
            let adapter = self.index_to_adapter.get_mut(&adapter_index).unwrap();
            entries.push((adapter.clone(), id, entry));
        }

        entries
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32, max_blocks: u32) -> Batch {
        // Final batch size
        let size = self.len() as u32;

        // TODO(travis): clone is not ideal, find a way to do this cleanly in place
        Batch {
            id: batch_id,
            requests: self.batch_requests.clone(),
            size,
            max_tokens,
            max_blocks,
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
    fn add(
        &mut self,
        id: u64,
        entry: Entry,
        adapter: Adapter,
        blocks: Vec<u32>,
        slots: Vec<u32>,
        prefix_len: u32,
        chunk_len: Option<u32>,
    );
    fn extend(&mut self, entries: Box<dyn BatchEntries>);
    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)>;
    fn create_batch_data(&self, batch_id: u64, max_tokens: u32, max_blocks: u32) -> Batch;
    fn adapters_in_use(&self) -> HashSet<Adapter>;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    #[allow(dead_code)]
    fn state(&self) -> &BatchEntriesState;
    fn mut_state(&mut self) -> &mut BatchEntriesState;

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        cached_batch: Option<CachedBatch>,
        span: Span,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch>;

    async fn process_next(
        &mut self,
        client: &mut ShardedClient,
        batches: Vec<CachedBatch>,
        span: Span,
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

    fn add(
        &mut self,
        id: u64,
        entry: Entry,
        adapter: Adapter,
        blocks: Vec<u32>,
        slots: Vec<u32>,
        prefix_len: u32,
        chunk_len: Option<u32>,
    ) {
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
            tokenized_inputs: request.tokenized_inputs.clone(),
            truncate: request.truncate,
            parameters: Some(request.parameters.clone()),
            stopping_parameters: Some(request.stopping_parameters.clone()),
            adapter_index: adapter.index(),
            blocks,
            slots,
            cache_len: prefix_len,
            chunk_len: chunk_len,
        };

        self.state.add(id, entry, adapter, request_proto);
    }

    fn extend(&mut self, mut entries: Box<dyn BatchEntries>) {
        let new_batch_entries = std::mem::take(&mut entries.mut_state().batch_entries);
        self.state.batch_entries.extend(new_batch_entries);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        self.state.drain()
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32, max_blocks: u32) -> Batch {
        self.state
            .create_batch_data(batch_id, max_tokens, max_blocks)
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

    fn mut_state(&mut self) -> &mut BatchEntriesState {
        &mut self.state
    }

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        cached_batch: Option<CachedBatch>,
        span: Span,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        prefill(
            client,
            batch,
            cached_batch,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(span)
        .await
    }

    async fn process_next(
        &mut self,
        client: &mut ShardedClient,
        batches: Vec<CachedBatch>,
        span: Span,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        decode(
            client,
            batches,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(span)
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

    fn add(
        &mut self,
        id: u64,
        entry: Entry,
        adapter: Adapter,
        blocks: Vec<u32>,
        slots: Vec<u32>,
        prefix_len: u32,
        chunk_len: Option<u32>,
    ) {
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
            tokenized_inputs: request.tokenized_inputs.clone(),
            truncate: 0,
            parameters: None,
            stopping_parameters: None,
            adapter_index: adapter.index(),
            blocks,
            slots,
            cache_len: prefix_len,
            chunk_len: chunk_len,
        };

        self.state.add(id, entry, adapter, request_proto);
    }

    fn extend(&mut self, mut entries: Box<dyn BatchEntries>) {
        let new_batch_entries = std::mem::take(&mut entries.mut_state().batch_entries);
        self.state.batch_entries.extend(new_batch_entries);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        self.state.drain()
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32, max_blocks: u32) -> Batch {
        self.state
            .create_batch_data(batch_id, max_tokens, max_blocks)
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

    fn mut_state(&mut self) -> &mut BatchEntriesState {
        &mut self.state
    }

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        _cached_batch: Option<CachedBatch>,
        span: Span,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        embed(
            client,
            batch,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(span)
        .await
    }

    async fn process_next(
        &mut self,
        _client: &mut ShardedClient,
        _batches: Vec<CachedBatch>,
        _span: Span,
        _generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        // TODO(travis): send error (programming eroor) if we get here
        None
    }
}

#[derive(Debug)]
pub(crate) struct ClassifyBatchEntries {
    pub(crate) state: BatchEntriesState,
}

impl ClassifyBatchEntries {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        Self {
            state: BatchEntriesState::new(num_entries, queue_len),
        }
    }
}

#[async_trait]
impl BatchEntries for ClassifyBatchEntries {
    fn can_add(&self, entry: &Entry) -> bool {
        // return false if the entry.request is not of type ValidEmbedRequest
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidClassifyRequest>();

        let result = valid_request.is_some();
        result
    }

    fn add(
        &mut self,
        id: u64,
        entry: Entry,
        adapter: Adapter,
        blocks: Vec<u32>,
        slots: Vec<u32>,
        prefix_len: u32,
        chunk_len: Option<u32>,
    ) {
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidClassifyRequest>();

        let request = valid_request.unwrap();
        let request_proto = Request {
            id,
            prefill_logprobs: false,
            inputs: request.inputs.clone(),
            tokenized_inputs: request.tokenized_inputs.clone(),
            truncate: 0,
            parameters: None,
            stopping_parameters: None,
            adapter_index: adapter.index(),
            blocks,
            slots,
            cache_len: prefix_len,
            chunk_len: chunk_len,
        };

        self.state.add(id, entry, adapter, request_proto);
    }

    fn extend(&mut self, mut entries: Box<dyn BatchEntries>) {
        let new_batch_entries = std::mem::take(&mut entries.mut_state().batch_entries);
        self.state.batch_entries.extend(new_batch_entries);
    }

    fn drain(&mut self) -> Vec<(Adapter, u64, Entry)> {
        self.state.drain()
    }

    fn create_batch_data(&self, batch_id: u64, max_tokens: u32, max_blocks: u32) -> Batch {
        self.state
            .create_batch_data(batch_id, max_tokens, max_blocks)
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

    fn mut_state(&mut self) -> &mut BatchEntriesState {
        &mut self.state
    }

    async fn process_first(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        _cached_batch: Option<CachedBatch>,
        span: Span,
        generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        classify(
            client,
            batch,
            &mut self.state.batch_entries,
            &generation_health,
        )
        .instrument(span)
        .await
    }

    async fn process_next(
        &mut self,
        _client: &mut ShardedClient,
        _batches: Vec<CachedBatch>,
        _span: Span,
        _generation_health: &Arc<AtomicBool>,
    ) -> Option<CachedBatch> {
        // TODO(magdy): send error (programming eroor) if we get here
        None
    }
}
