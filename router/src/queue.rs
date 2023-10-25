use crate::adapter::Adapter;
use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::min;
use text_generation_client::ShardedClient;
use std::collections::VecDeque;
use text_generation_client::{Batch, Request};
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
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

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// adapter associated with this queue
    adapter: Adapter,
    /// Channel to communicate with the background queue task
    queue_sender: flume::Sender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new(
        adapter: Adapter, 
        client: ShardedClient, 
        requires_padding: bool, 
        block_size: u32, 
        window_size: Option<u32>
    ) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = flume::unbounded();

        // Launch background queue task
        tokio::spawn(queue_task(
            adapter.clone(),
            client,
            requires_padding,
            block_size,
            window_size,
            queue_receiver,
        ));
        Self { adapter, queue_sender }
    }
    
    /// Return adapter ID
    pub(crate) fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    pub(crate) async fn load_adapter(&self) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::LoadAdapter {
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn is_empty(&self) -> bool {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::IsEmpty {
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn is_errored(&self) -> bool {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::IsErrored {
                response_sender,
                span: Span::current()
            }).unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn terminate(&self) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::Terminate {
                response_sender,
                span: Span::current()
            }).unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn peek(&self) -> Option<Instant> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::Peek {
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn pop(&self) -> Option<(u64, Entry, Option<Instant>)> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::Pop {
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }
}

// Background task responsible of the queue state
async fn queue_task(
    adapter: Adapter,
    mut client: ShardedClient,
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    receiver: flume::Receiver<QueueCommand>,
) {
    let mut state = State::new(requires_padding, block_size, window_size, adapter.index());
    let mut err_msg: Option<String> = None;

    // download the adapter
    match client.download_adapter(
        adapter.id().to_string(), 
        adapter.source().to_string(),
    ).await {
        Ok(_) => {
            tracing::info!("adapter {} downloaded", adapter.id());
        }
        // if we have a download error, we send an error to the entry response
        Err(error) => {
            metrics::increment_counter!("tgi_request_failure", "err" => "download_adapter");
            err_msg = Some(error.to_string());
        }
    }

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            QueueCommand::Append(entry, span) => span.in_scope(|| {
                // no-op if adapter errored
                if err_msg.is_some() {
                    return;
                }
                state.append(*entry);
            }),
            QueueCommand::LoadAdapter {
                response_sender,
                span: _  // TODO(geoffrey): not sure how to use 'span' with async fn
            } => {
                if err_msg.is_some() {
                    response_sender.send(()).unwrap();
                    continue;
                }
                match client.load_adapter(
                    adapter.id().to_string(),
                    adapter.source().to_string(),
                    adapter.index(),
                ).await {
                    Ok(_) => {
                        tracing::info!("adapter {} loaded", adapter.id());
                        response_sender.send(()).unwrap();
                    }
                    // If we have a load error, we send an error to the entry response
                    Err(error) => {
                        metrics::increment_counter!("tgi_request_failure", "err" => "load_adapter");
                        err_msg = Some(error.to_string());
                        response_sender.send(()).unwrap();
                    }
                }
            }
            QueueCommand::IsEmpty { 
                response_sender,
                span
            } => span.in_scope(|| {
                if err_msg.is_some() {
                    response_sender.send(true).unwrap();
                } else {
                    let response = state.is_empty();
                    response_sender.send(response).unwrap();
                }
            }),
            QueueCommand::IsErrored{
                response_sender,
                span
            } => span.in_scope(|| {
                response_sender.send(err_msg.is_some()).unwrap();
            }),
            QueueCommand::Peek { 
                response_sender,
                span
            } => span.in_scope(|| {
                if err_msg.is_some() {
                    response_sender.send(None).unwrap();
                    return;
                } else {
                    let response = state.peek();
                    response_sender.send(response).unwrap();
                }
            }),
            QueueCommand::Pop { 
                response_sender,
                span
            } => span.in_scope(|| {
                if err_msg.is_some() {
                    response_sender.send(None).unwrap();
                    return;
                } else {
                    let response = state.pop();
                    response_sender.send(response).unwrap();
                    metrics::gauge!("tgi_queue_size", state.entries.len() as f64);
                }
            }),
            QueueCommand::Terminate {
                response_sender,
                span,
            } => {
                tracing::info!("terminating adapter queue for {}", adapter.id());

                // Create an asynchronous closure
                let span_closure = async move {
                    span.in_scope(|| {
                        for entry in state.entries.drain(..) {
                            let (_, entry) = entry;
                            if let Some(err_msg) = err_msg.clone() {
                                entry.response_tx.send(Err(InferError::GenerationError(err_msg))).unwrap();
                            }
                        }
                    });
                };

                // Await the closure and break the loop
                tokio::spawn(span_closure).await.expect("spawn failed");
                response_sender.send(()).unwrap();
                break;
            }
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,

    /// Adapter index
    adapter_index: u32,
}

impl State {
    fn new(requires_padding: bool, block_size: u32, window_size: Option<u32>, adapter_index: u32) -> Self {
        Self {
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            requires_padding,
            block_size,
            window_size,
            adapter_index,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
    }

    // Is empty
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // Peeks at the front of the queue and returns timestamp of the oldest entry, if present
    fn peek(&self) -> Option<Instant> {
        self.entries.front().map(|(_, entry)| entry.queue_time)
    }

    // Pops the front of the queue and returns the oldest entry, if present
    fn pop(&mut self) -> Option<(u64, Entry, Option<Instant>)> {
        self.entries.pop_front().map(|(id, mut entry)| (id, entry, self.peek()))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    LoadAdapter {
        response_sender: oneshot::Sender<()>,
        span: Span,
    },
    IsEmpty {
        response_sender: oneshot::Sender<bool>,
        span: Span,
    },
    IsErrored {
        response_sender: oneshot::Sender<bool>,
        span: Span,
    },
    Terminate {
        response_sender: oneshot::Sender<()>,
        span: Span,
    },
    Peek {
        response_sender: oneshot::Sender<Option<Instant>>,
        span: Span,
    },
    Pop {
        response_sender: oneshot::Sender<Option<(u64, Entry, Option<Instant>)>>,
        span: Span,
    },
}

// TODO(geoffrey): revisit unit tests. They should work given the minimal changes
// to existing functionality. The issue is that Queue now takes a ShardedClient,
// which I'm not yet sure how to mock out. They should also be extended to test
// the new functionality.
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
//     use tracing::info_span;

//     fn default_entry() -> (
//         Entry,
//         flume::Receiver<Result<InferStreamResponse, InferError>>,
//     ) {
//         let (response_tx, receiver_tx) = flume::unbounded();

//         let entry = Entry {
//             request: ValidGenerateRequest {
//                 inputs: "".to_string(),
//                 input_length: 0,
//                 truncate: 0,
//                 decoder_input_details: false,
//                 parameters: NextTokenChooserParameters {
//                     temperature: 0.0,
//                     top_k: 0,
//                     top_p: 0.0,
//                     typical_p: 0.0,
//                     do_sample: false,
//                     seed: 0,
//                     repetition_penalty: 0.0,
//                     watermark: false,
//                 },
//                 stopping_parameters: StoppingCriteriaParameters {
//                     ignore_eos_token: false,
//                     max_new_tokens: 1,
//                     stop_sequences: vec![],
//                 },
//             },
//             response_tx,
//             span: info_span!("entry"),
//             temp_span: None,
//             queue_time: Instant::now(),
//             batch_time: None,
//         };
//         (entry, receiver_tx)
//     }

//     #[test]
//     fn test_append() {
//         let mut state = State::new(false, 1);
//         let (entry, _guard) = default_entry();

//         assert_eq!(state.next_id, 0);
//         assert_eq!(state.entries.len(), 0);

//         state.append(entry);

//         assert_eq!(state.next_id, 1);
//         assert_eq!(state.entries.len(), 1);
//         let (id, _) = state.entries.remove(0).unwrap();
//         assert_eq!(id, 0);
//     }

//     #[test]
//     fn test_next_batch_empty() {
//         let mut state = State::new(false, 1);

//         assert!(state.next_batch(None, 1, 1).is_none());
//         assert!(state.next_batch(Some(1), 1, 1).is_none());
//     }

//     #[test]
//     fn test_next_batch_min_size() {
//         let mut state = State::new(false, 1);
//         let (entry1, _guard1) = default_entry();
//         let (entry2, _guard2) = default_entry();
//         state.append(entry1);
//         state.append(entry2);

//         let (entries, batch, _) = state.next_batch(None, 2, 2).unwrap();
//         assert_eq!(entries.len(), 2);
//         assert!(entries.contains_key(&0));
//         assert!(entries.contains_key(&1));
//         assert!(entries.get(&0).unwrap().batch_time.is_some());
//         assert!(entries.get(&1).unwrap().batch_time.is_some());
//         assert_eq!(batch.id, 0);
//         assert_eq!(batch.size, 2);

//         assert_eq!(state.next_id, 2);
//         assert_eq!(state.entries.len(), 0);
//         assert_eq!(state.next_batch_id, 1);

//         let (entry3, _guard3) = default_entry();
//         state.append(entry3);

//         assert!(state.next_batch(Some(2), 2, 2).is_none());

//         assert_eq!(state.next_id, 3);
//         assert_eq!(state.entries.len(), 1);
//         let (id, _) = state.entries.remove(0).unwrap();
//         assert_eq!(id, 2);
//     }

//     #[test]
//     fn test_next_batch_token_budget() {
//         let mut state = State::new(false, 1);
//         let (entry1, _guard1) = default_entry();
//         let (entry2, _guard2) = default_entry();
//         state.append(entry1);
//         state.append(entry2);

//         let (entries, batch, _) = state.next_batch(None, 1, 1).unwrap();
//         assert_eq!(entries.len(), 1);
//         assert!(entries.contains_key(&0));
//         assert_eq!(batch.id, 0);
//         assert_eq!(batch.size, 1);

//         assert_eq!(state.next_id, 2);
//         assert_eq!(state.entries.len(), 1);
//         assert_eq!(state.next_batch_id, 1);

//         let (entry3, _guard3) = default_entry();
//         state.append(entry3);

//         let (entries, batch, _) = state.next_batch(None, 3, 3).unwrap();
//         assert_eq!(entries.len(), 2);
//         assert!(entries.contains_key(&1));
//         assert!(entries.contains_key(&2));
//         assert_eq!(batch.id, 1);
//         assert_eq!(batch.size, 2);

//         assert_eq!(state.next_id, 3);
//         assert_eq!(state.entries.len(), 0);
//         assert_eq!(state.next_batch_id, 2);
//     }

//     #[tokio::test]
//     async fn test_queue_append() {
//         let queue = Queue::new(false, 1);
//         let (entry, _guard) = default_entry();
//         queue.append(entry);
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_empty() {
//         let queue = Queue::new(false, 1);

//         assert!(queue.next_batch(None, 1, 1).await.is_none());
//         assert!(queue.next_batch(Some(1), 1, 1).await.is_none());
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_min_size() {
//         let queue = Queue::new(false, 1);
//         let (entry1, _guard1) = default_entry();
//         let (entry2, _guard2) = default_entry();
//         queue.append(entry1);
//         queue.append(entry2);

//         let (entries, batch, _) = queue.next_batch(None, 2, 2).await.unwrap();
//         assert_eq!(entries.len(), 2);
//         assert!(entries.contains_key(&0));
//         assert!(entries.contains_key(&1));
//         assert!(entries.get(&0).unwrap().batch_time.is_some());
//         assert!(entries.get(&1).unwrap().batch_time.is_some());
//         assert_eq!(batch.id, 0);
//         assert_eq!(batch.size, 2);

//         let (entry3, _guard3) = default_entry();
//         queue.append(entry3);

//         // Not enough requests pending
//         assert!(queue.next_batch(Some(2), 2, 2).await.is_none());
//         // Not enough token budget
//         assert!(queue.next_batch(Some(1), 0, 0).await.is_none());
//         // Ok
//         let (entries2, batch2, _) = queue.next_batch(Some(1), 2, 2).await.unwrap();
//         assert_eq!(entries2.len(), 1);
//         assert!(entries2.contains_key(&2));
//         assert!(entries2.get(&2).unwrap().batch_time.is_some());
//         assert_eq!(batch2.id, 1);
//         assert_eq!(batch2.size, 1);
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_token_budget() {
//         let queue = Queue::new(false, 1);
//         let (entry1, _guard1) = default_entry();
//         let (entry2, _guard2) = default_entry();
//         queue.append(entry1);
//         queue.append(entry2);

//         let (entries, batch, _) = queue.next_batch(None, 1, 1).await.unwrap();
//         assert_eq!(entries.len(), 1);
//         assert!(entries.contains_key(&0));
//         assert_eq!(batch.id, 0);
//         assert_eq!(batch.size, 1);

//         let (entry3, _guard3) = default_entry();
//         queue.append(entry3);

//         let (entries, batch, _) = queue.next_batch(None, 3, 3).await.unwrap();
//         assert_eq!(entries.len(), 2);
//         assert!(entries.contains_key(&1));
//         assert!(entries.contains_key(&2));
//         assert_eq!(batch.id, 1);
//         assert_eq!(batch.size, 2);
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_dropped_receiver() {
//         let queue = Queue::new(false, 1);
//         let (entry, _) = default_entry();
//         queue.append(entry);

//         assert!(queue.next_batch(None, 1, 1).await.is_none());
//     }
// }
