use crate::adapter;
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

/// Request AdapterLoader
#[derive(Debug, Clone)]
pub(crate) struct AdapterLoader {
    /// Channel to communicate with the background task
    sender: flume::Sender<AdapterLoaderCommand>,
}

impl AdapterLoader {
    pub(crate) fn new(client: ShardedClient) -> Self {
        // Create channel
        let (sender, receiver) = flume::unbounded();

        // Launch background queue task
        tokio::spawn(loader_task(client, receiver));
        Self { sender }
    }

    pub(crate) async fn download_adapter(&self, adapter: Adapter) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::DownloadAdapter {
                adapter,
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn load_adapter(&self, adapter: Adapter) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::LoadAdapter {
                adapter,
                response_sender,
                span: Span::current()
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn is_errored(&self) -> bool {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::IsErrored {
                response_sender,
                span: Span::current()
            }).unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) async fn terminate(&self) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::Terminate {
                response_sender,
                span: Span::current()
            }).unwrap();
        response_receiver.await.unwrap()
    }
}

// Background task responsible of the loader state
async fn loader_task(
    mut client: ShardedClient,
    receiver: flume::Receiver<AdapterLoaderCommand>,
) {
    let mut err_msg: Option<String> = None;

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            AdapterLoaderCommand::DownloadAdapter {
                adapter,
                response_sender,
                span: _  // TODO(geoffrey): not sure how to use 'span' with async fn
            } => {
                if err_msg.is_some() {
                    response_sender.send(()).unwrap();
                    continue;
                }
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
            },
            AdapterLoaderCommand::LoadAdapter {
                adapter,
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
            AdapterLoaderCommand::IsErrored{
                response_sender,
                span
            } => span.in_scope(|| {
                response_sender.send(err_msg.is_some()).unwrap();
            }),
            AdapterLoaderCommand::Terminate {
                response_sender,
                span,
            } => {
                tracing::info!("terminating loader");

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

/// AdapterLoader State
#[derive(Debug)]
struct State {
    /// AdapterLoader entries organized in a Vec
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

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum AdapterLoaderCommand {
    DownloadAdapter {
        adapter: Adapter,
        response_sender: oneshot::Sender<()>,
        span: Span,
    },
    LoadAdapter {
        adapter: Adapter,
        response_sender: oneshot::Sender<()>,
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
}

// TODO(geoffrey): revisit unit tests. They should work given the minimal changes
// to existing functionality. The issue is that AdapterLoader now takes a ShardedClient,
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
//         let queue = AdapterLoader::new(false, 1);
//         let (entry, _guard) = default_entry();
//         queue.append(entry);
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_empty() {
//         let queue = AdapterLoader::new(false, 1);

//         assert!(queue.next_batch(None, 1, 1).await.is_none());
//         assert!(queue.next_batch(Some(1), 1, 1).await.is_none());
//     }

//     #[tokio::test]
//     async fn test_queue_next_batch_min_size() {
//         let queue = AdapterLoader::new(false, 1);
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
//         let queue = AdapterLoader::new(false, 1);
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
//         let queue = AdapterLoader::new(false, 1);
//         let (entry, _) = default_entry();
//         queue.append(entry);

//         assert!(queue.next_batch(None, 1, 1).await.is_none());
//     }
// }
