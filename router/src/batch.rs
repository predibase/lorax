use core::fmt::Debug;
use std::sync::Arc;

use lorax_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use tokio::time::Instant;
use tracing::Span;

use crate::{
    adapter::Adapter,
    infer::{InferError, InferStreamResponse},
};

pub(crate) trait ValidRequest {
    fn input_length(&self) -> u32;
    fn max_new_tokens(&self) -> u32;
    fn to_batch(&self) -> Arc<dyn BatchEntries>;
}

impl Debug for dyn ValidRequest {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ValidRequest")
            .field("input_length", &self.input_length())
            .field("max_new_tokens", &self.max_new_tokens())
            .finish()
    }
}

impl ValidRequest for ValidGenerateRequest {
    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        self.stopping_parameters.max_new_tokens
    }

    fn to_batch(&self) -> Arc<dyn BatchEntries> {
        Arc::new(GenerateBatchEntries::new())
    }
}

#[derive(Debug)]
pub(crate) struct ValidEmbedRequest {
    pub inputs: String,
    pub input_length: u32,
}

impl ValidRequest for ValidEmbedRequest {
    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        1
    }

    fn to_batch(&self) -> Arc<dyn BatchEntries> {
        Arc::new(EmbedBatchEntries::new())
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
    fn add(&mut self, entry: Entry);
}

#[derive(Debug)]
pub(crate) struct GenerateBatchEntries {
    pub(crate) entries: Vec<Entry>,
}

impl GenerateBatchEntries {
    pub(crate) fn new() -> Self {
        Self { entries: vec![] }
    }
}

impl BatchEntries for GenerateBatchEntries {
    fn add(&mut self, entry: Entry) {
        self.entries.push(entry);
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
    fn add(&mut self, entry: Entry) {
        self.entries.push(entry);
    }
}
