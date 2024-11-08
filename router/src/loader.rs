use crate::adapter::Adapter;
use crate::infer::InferError;
use crate::queue::{AdapterQueuesState, AdapterStatus};
use lorax_client::ShardedClient;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::Span;

/// Request AdapterLoader
#[derive(Debug, Clone)]
pub(crate) struct AdapterLoader {
    /// Channel to communicate with the background task
    sender: mpsc::UnboundedSender<AdapterLoaderCommand>,
}

impl AdapterLoader {
    pub(crate) fn new(client: ShardedClient) -> Self {
        // Create channel
        let (sender, receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(loader_task(client, receiver));
        Self { sender }
    }

    pub(crate) fn download_adapter(
        &self,
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
    ) {
        // Create response channel
        let (response_sender, _response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::DownloadAdapter {
                adapter,
                queues_state,
                response_sender,
                span: Span::current(),
            })
            .unwrap();

        // Don't block the caller
        // response_receiver.await.unwrap()
    }

    pub(crate) fn load_adapter(
        &self,
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
    ) {
        // Create response channel
        let (response_sender, _response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::LoadAdapter {
                adapter,
                queues_state,
                response_sender,
                span: Span::current(),
            })
            .unwrap();

        // Don't block the caller
        // response_receiver.await.unwrap()
    }

    pub(crate) fn offload_adapter(
        &self,
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
    ) {
        // Create response channel
        let (response_sender, _response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::OffloadAdapter {
                adapter,
                queues_state,
                response_sender,
                span: Span::current(),
            })
            .unwrap();

        // Don't block the caller
        // response_receiver.await.unwrap()
    }

    #[allow(dead_code)] // cuurently unused
    pub(crate) async fn is_errored(&self, adapter: Adapter) -> bool {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::IsErrored {
                adapter,
                response_sender,
                span: Span::current(),
            })
            .unwrap();
        response_receiver.await.unwrap()
    }

    pub(crate) fn terminate(&self, adapter: Adapter, queues_state: Arc<Mutex<AdapterQueuesState>>) {
        // Create response channel
        let (response_sender, _response_receiver) = oneshot::channel();
        self.sender
            .send(AdapterLoaderCommand::Terminate {
                adapter,
                queues_state,
                response_sender,
                span: Span::current(),
            })
            .unwrap();

        // Don't block the caller
        // response_receiver.await.unwrap()
    }
}

// Background task responsible of the loader state
async fn loader_task(
    mut client: ShardedClient,
    mut receiver: mpsc::UnboundedReceiver<AdapterLoaderCommand>,
) {
    let mut err_msgs: HashMap<Adapter, String> = HashMap::new();

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            AdapterLoaderCommand::DownloadAdapter {
                adapter,
                queues_state,
                response_sender: _,
                span: _, // TODO(geoffrey): not sure how to use 'span' with async fn
            } => {
                if err_msgs.contains_key(&adapter) {
                    metrics::increment_counter!("lorax_request_failure", "err" => "download_adapter");
                    let mut locked_state = queues_state.lock().await;
                    if locked_state.has_adapter(&adapter) {
                        // Above check guards against the case where the adapter was terminated between the initial
                        // time of request and the time of adapter download
                        locked_state.set_status(&adapter, AdapterStatus::Errored);
                    }
                    continue;
                }

                match client
                    .download_adapter(
                        adapter.params().clone().into(),
                        adapter.source().to_string(),
                        adapter.api_token().clone(),
                    )
                    .await
                {
                    Ok(resp) => {
                        tracing::info!("adapter {} downloaded", adapter.as_string());
                        let mut locked_state = queues_state.lock().await;
                        if locked_state.has_adapter(&adapter) {
                            // Above check guards against the case where the adapter was terminated between the initial
                            // time of request and the time of adapter download
                            locked_state.set_cost(&adapter, resp.memory_fraction);
                            locked_state.set_status(&adapter, AdapterStatus::Downloaded);
                        }
                    }
                    // if we have a download error, we send an error to the entry response
                    Err(error) => {
                        tracing::info!("FAILED downloading adapter {}", adapter.as_string());
                        metrics::increment_counter!("lorax_request_failure", "err" => "download_adapter");
                        let mut locked_state = queues_state.lock().await;
                        if locked_state.has_adapter(&adapter) {
                            // Above check guards against the case where the adapter was terminated between the initial
                            // time of request and the time of adapter download
                            locked_state.set_status(&adapter, AdapterStatus::Errored);
                        }
                        err_msgs.insert(adapter, error.to_string());
                    }
                }
            }
            AdapterLoaderCommand::LoadAdapter {
                adapter,
                queues_state,
                response_sender: _,
                span: _, // TODO(geoffrey): not sure how to use 'span' with async fn
            } => {
                if err_msgs.contains_key(&adapter) {
                    metrics::increment_counter!("lorax_request_failure", "err" => "load_adapter");
                    let mut locked_state = queues_state.lock().await;
                    if locked_state.has_adapter(&adapter) {
                        locked_state.set_status(&adapter, AdapterStatus::Errored);
                    }
                    // response_sender.send(()).unwrap();
                    continue;
                }
                match client
                    .load_adapter(
                        adapter.params().clone().into(),
                        adapter.source().to_string(),
                        adapter.index(),
                        adapter.api_token().clone(),
                    )
                    .await
                {
                    Ok(_) => {
                        tracing::info!("adapter {} loaded", adapter.as_string());
                        queues_state
                            .lock()
                            .await
                            .set_status(&adapter, AdapterStatus::Ready);
                        // response_sender.send(()).unwrap();
                    }
                    // If we have a load error, we send an error to the entry response
                    Err(error) => {
                        tracing::info!("FAILED loading adapter {}", adapter.as_string());
                        metrics::increment_counter!("lorax_request_failure", "err" => "load_adapter");
                        queues_state
                            .lock()
                            .await
                            .set_status(&adapter, AdapterStatus::Errored);
                        err_msgs.insert(adapter, error.to_string());
                        // response_sender.send(()).unwrap();
                    }
                }
            }
            AdapterLoaderCommand::OffloadAdapter {
                adapter,
                queues_state,
                response_sender: _,
                span: _, // TODO(geoffrey): not sure how to use 'span' with async fn
            } => {
                if err_msgs.contains_key(&adapter) {
                    metrics::increment_counter!("lorax_request_failure", "err" => "offload_adapter");
                    let mut locked_state = queues_state.lock().await;
                    if locked_state.has_adapter(&adapter) {
                        locked_state.set_status(&adapter, AdapterStatus::Errored);
                    }
                    // response_sender.send(()).unwrap();
                    continue;
                }
                match client
                    .offload_adapter(
                        adapter.params().clone().into(),
                        adapter.source().to_string(),
                        adapter.index(),
                    )
                    .await
                {
                    Ok(_) => {
                        tracing::info!("adapter {} offloaded", adapter.as_string());
                        queues_state
                            .lock()
                            .await
                            .set_status(&adapter, AdapterStatus::Downloaded);
                        // response_sender.send(()).unwrap();
                    }
                    // If we have a load error, we send an error to the entry response
                    Err(error) => {
                        tracing::info!("FAILED offloading adapter {}", adapter.as_string());
                        metrics::increment_counter!("lorax_request_failure", "err" => "offload_adapter");
                        queues_state
                            .lock()
                            .await
                            .set_status(&adapter, AdapterStatus::Errored);
                        err_msgs.insert(adapter, error.to_string());
                        // response_sender.send(()).unwrap();
                    }
                }
            }
            AdapterLoaderCommand::IsErrored {
                adapter,
                response_sender,
                span,
            } => span.in_scope(|| {
                response_sender
                    .send(err_msgs.contains_key(&adapter))
                    .unwrap();
            }),
            AdapterLoaderCommand::Terminate {
                adapter,
                queues_state,
                response_sender: _,
                span: _,
            } => {
                tracing::info!("terminating adapter {} loader", adapter.as_string());

                let mut locked_state = queues_state.lock().await;
                if !locked_state.has_adapter(&adapter) {
                    err_msgs.remove(&adapter);
                    continue;
                }

                for entry in locked_state.drain(&adapter) {
                    let (_, entry) = entry;
                    if let Some(err_msg) = err_msgs.get(&adapter) {
                        entry
                            .response_tx
                            .send(Err(InferError::GenerationError(err_msg.clone())))
                            .unwrap();
                    }
                }
                err_msgs.remove(&adapter);
                locked_state.remove(&adapter);
            }
        }
    }
}

#[derive(Debug)]
enum AdapterLoaderCommand {
    DownloadAdapter {
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
        #[allow(dead_code)] // currently unused
        response_sender: oneshot::Sender<()>,
        #[allow(dead_code)] // currently unused
        span: Span,
    },
    LoadAdapter {
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
        #[allow(dead_code)] // currently unused
        response_sender: oneshot::Sender<()>,
        #[allow(dead_code)] // currently unused
        span: Span,
    },
    OffloadAdapter {
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
        #[allow(dead_code)] // currently unused
        response_sender: oneshot::Sender<()>,
        #[allow(dead_code)] // currently unused
        span: Span,
    },
    #[allow(dead_code)]
    IsErrored {
        adapter: Adapter,
        response_sender: oneshot::Sender<bool>,
        span: Span,
    },
    Terminate {
        adapter: Adapter,
        queues_state: Arc<Mutex<AdapterQueuesState>>,
        #[allow(dead_code)] // currently unused
        response_sender: oneshot::Sender<()>,
        #[allow(dead_code)] // currently unused
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
//     use lorax_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
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
//                     return_k_alternatives: 0,
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
