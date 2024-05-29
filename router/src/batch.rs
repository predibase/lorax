use core::fmt::Debug;
use std::{any::Any, collections::HashMap, sync::Arc};

use lorax_client::{NextTokenChooserParameters, Request, StoppingCriteriaParameters};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use tokio::time::Instant;
use tracing::{info_span, Span};

use crate::{
    adapter::Adapter,
    infer::{InferError, InferStreamResponse},
};

pub(crate) trait ValidRequest: Sync + Send + Debug + Any {
    fn input_length(&self) -> u32;
    fn max_new_tokens(&self) -> u32;
    fn adapter(&self) -> Adapter;
    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Arc<dyn BatchEntries>;
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
        self.adapter
    }

    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Arc<dyn BatchEntries> {
        Arc::new(GenerateBatchEntries::new(num_entries, queue_len))
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
        self.adapter
    }

    fn to_batch(&self, num_entries: usize, queue_len: usize) -> Arc<dyn BatchEntries> {
        Arc::new(EmbedBatchEntries::new())
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

pub(crate) trait BatchEntries {
    fn add(&mut self, id: u64, entry: Entry, adapter: Adapter) -> bool;
}

#[derive(Debug)]
pub(crate) struct GenerateBatchEntries {
    pub(crate) batch_requests: Vec<Request>,
    pub(crate) batch_entries: HashMap<u64, Entry, BuildNoHashHasher<u64>>,
    pub(crate) index_to_adapter: HashMap<u32, Adapter>,
    next_batch_span: Span,
}

impl GenerateBatchEntries {
    pub(crate) fn new(num_entries: usize, queue_len: usize) -> Self {
        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(num_entries);
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(num_entries, BuildNoHashHasher::default());

        let mut index_to_adapter = HashMap::with_capacity(queue_len);

        Self {
            batch_requests,
            batch_entries,
            index_to_adapter,
            next_batch_span,
        }
    }
}

impl BatchEntries for GenerateBatchEntries {
    fn add(&mut self, id: u64, mut entry: Entry, adapter: Adapter) -> bool {
        // return false if the entry.request is not of type ValidGenerateRequest
        let valid_request = entry
            .request
            .as_ref()
            .as_any()
            .downcast_ref::<ValidGenerateRequest>();

        if valid_request.is_none() {
            return false;
        }

        let request = valid_request.unwrap();

        // self.entries.push(entry);

        // Create a new span to link the batch back to this entry
        let entry_batch_span = info_span!(parent: &entry.span, "infer");
        // Add relationships
        self.next_batch_span.follows_from(&entry_batch_span);
        entry_batch_span.follows_from(&self.next_batch_span);
        // Update entry
        entry.temp_span = Some(entry_batch_span);

        self.batch_requests.push(Request {
            id,
            prefill_logprobs: request.decoder_input_details,
            inputs: request.inputs.clone(),
            truncate: request.truncate,
            parameters: Some(request.parameters.clone()),
            stopping_parameters: Some(request.stopping_parameters.clone()),
            adapter_index: adapter.index(),
            apply_chat_template: request.apply_chat_template,
        });
        // Set batch_time
        entry.batch_time = Some(Instant::now());
        // Insert in batch_entries IntMap
        self.batch_entries.insert(id, entry);
        // Map from adapter index back to queue in case we need to add back entries below
        // let queue = queue_map.get_mut(&adapter).unwrap();
        self.index_to_adapter.insert(adapter.index(), adapter);
        return true;
    }
}

#[derive(Debug)]
pub(crate) struct EmbedBatchEntries {
    pub(crate) entries: Vec<Entry>,
}

impl EmbedBatchEntries {
    pub(crate) fn new() -> Self {
        Self { entries: vec![] }
    }
}

impl BatchEntries for EmbedBatchEntries {
    fn add(&mut self, id: u64, mut entry: Entry, adapter: Adapter) -> bool {
        return false;
    }
}
