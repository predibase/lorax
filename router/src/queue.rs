use std::{collections::VecDeque, sync::Arc};

use tokio::{time::Instant, sync::Notify};
use tracing::{info_span, Span};

use crate::{adapter::Adapter, validation::ValidGenerateRequest, infer::{InferStreamResponse, InferError}};

/// AdapterLoader entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
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

#[derive(Debug, PartialEq)]
pub(crate) enum AdapterStatus {
    Downloading,
    Downloaded,
    Loading,
    Ready,
    Errored,
}

#[derive(Debug)]
pub(crate) struct AdapterEvent {
    /// Adapter readniess task notifier
    pub batching_task: Notify,
}

/// Queue State
#[derive(Debug)]
pub(crate) struct QueueState {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Adapter index
    adapter: Adapter,

    /// Adapter status
    status: AdapterStatus,

    /// Timestamp when the adapter was last activated
    activation_ts: Option<Instant>,

    /// Adapter event
    event: Arc<AdapterEvent>,
}

impl QueueState {
    pub(crate) fn new(adapter: Adapter, event: Arc<AdapterEvent>) -> Self {
        let status = AdapterStatus::Downloading;
        Self {
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            adapter,
            status,
            activation_ts: None,
            event,
        }
    }

    /// Append an entry to the queue
    pub(crate) fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
    }

    // Is empty
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // Peeks at the front of the queue and returns timestamp of the oldest entry, if present
    pub(crate) fn peek(&self) -> Option<Instant> {
        self.entries.front().map(|(_, entry)| entry.queue_time)
    }

    // Pops the front of the queue and returns the oldest entry, if present
    pub(crate) fn pop(&mut self) -> Option<(u64, Entry, Option<Instant>)> {
        self.entries.pop_front().map(|(id, mut entry)| (id, entry, self.peek()))
    }

    pub(crate) fn entries(&self) -> &VecDeque<(u64, Entry)> {
        &self.entries
    }

    pub(crate) fn drain(&mut self) -> std::collections::vec_deque::Drain<(u64, Entry)> {
        self.entries.drain(..)
    }

    pub(crate) fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub(crate) fn set_status(&mut self, status: AdapterStatus) {
        self.status = status;
        if self.status == AdapterStatus::Ready {
            self.event.batching_task.notify_one();
        }
    }

    pub(crate) fn status(&self) -> &AdapterStatus {
        &self.status
    }

    pub(crate) fn set_activation_ts(&mut self, ts: Instant) {
        self.activation_ts = Some(ts);
    }

    pub(crate) fn activation_ts(&self) -> Option<Instant> {
        self.activation_ts
    }
}
