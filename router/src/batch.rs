use lorax_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use tokio::time::Instant;
use tracing::Span;

use crate::{
    adapter::Adapter,
    infer::{InferError, InferStreamResponse},
};

pub(crate) trait ValidRequest: Sized {
    type BatchEntries: BatchEntries<Self>
    where
        Self: Sized;

    fn input_length(&self) -> u32;
    fn max_new_tokens(&self) -> u32;
    fn to_batch(&self, entry: Entry<Self>) -> Self::BatchEntries;
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

impl ValidRequest for ValidGenerateRequest {
    type BatchEntries = GenerateBatchEntries;

    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        self.stopping_parameters.max_new_tokens
    }

    fn to_batch(&self, entry: Entry<ValidGenerateRequest>) -> Self::BatchEntries {
        GenerateBatchEntries::new(entry)
    }
}

#[derive(Debug)]
pub(crate) struct ValidEmbedRequest {
    pub inputs: String,
    pub input_length: u32,
}

impl ValidRequest for ValidEmbedRequest {
    type BatchEntries = EmbedBatchEntries;

    fn input_length(&self) -> u32 {
        self.input_length
    }

    fn max_new_tokens(&self) -> u32 {
        1
    }

    fn to_batch(&self, entry: Entry<ValidEmbedRequest>) -> Self::BatchEntries {
        EmbedBatchEntries::new(entry)
    }
}

/// AdapterLoader entry
#[derive(Debug, Clone)]
pub(crate) struct Entry<T: ValidRequest> {
    /// Request
    pub request: T,
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

pub(crate) trait BatchEntries<T: ValidRequest>: Sized {
    fn push(&mut self, entry: Entry<T>);
    fn pop(&mut self) -> Option<Entry<T>>;
    fn len(&self) -> usize;
}

#[derive(Debug)]
pub(crate) struct GenerateBatchEntries {
    pub(crate) entries: Vec<Entry<ValidGenerateRequest>>,
}

impl GenerateBatchEntries {
    pub(crate) fn new(entry: Entry<ValidGenerateRequest>) -> Self {
        Self {
            entries: vec![entry],
        }
    }
}

impl BatchEntries<ValidGenerateRequest> for GenerateBatchEntries {
    fn push(&mut self, entry: Entry<ValidGenerateRequest>) {
        self.entries.push(entry);
    }

    fn pop(&mut self) -> Option<Entry<ValidGenerateRequest>> {
        self.entries.pop()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug)]
pub(crate) struct EmbedBatchEntries {
    pub(crate) entries: Vec<Entry<ValidEmbedRequest>>,
}

impl EmbedBatchEntries {
    pub(crate) fn new(entry: Entry<ValidEmbedRequest>) -> Self {
        Self {
            entries: vec![entry],
        }
    }
}

impl BatchEntries<ValidEmbedRequest> for EmbedBatchEntries {
    fn push(&mut self, entry: Entry<ValidEmbedRequest>) {
        self.entries.push(entry);
    }

    fn pop(&mut self) -> Option<Entry<ValidEmbedRequest>> {
        self.entries.pop()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}
