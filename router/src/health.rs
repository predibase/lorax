use lorax_client::{
    Batch, NextTokenChooserParameters, Request, ShardInfo, ShardedClient,
    StoppingCriteriaParameters, Cl
};
use crate::{
  ClassifyRequest, EmbedRequest
}
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Note: Request ids and batch ids cannot collide.
#[allow(dead_code)]
const LIVENESS_ID: u64 = u64::MAX;
const BATCH_ID: u64 = u64::MAX;

#[derive(Clone, Debug)]
pub(crate) struct Health {
    client: ShardedClient,
    generation_health: Arc<AtomicBool>,
    shard_info: ShardInfo,
}

impl Health {
    pub(crate) fn new(
        client: ShardedClient,
        generation_health: Arc<AtomicBool>,
        shard_info: ShardInfo,
    ) -> Self {
        Self {
            #[allow(dead_code)]
            client,
            #[allow(dead_code)]
            generation_health,
            shard_info,
        }
    }

    pub(crate) fn shard_info(&self) -> &ShardInfo {
        &self.shard_info
    }

    #[allow(dead_code)]
    pub(crate) async fn check(&mut self) -> bool {
        if self.generation_health.load(Ordering::SeqCst) {
            // Generation is healthy, we only check that the shards are answering gRPC calls
            self.client.health().await.is_ok()
        } else {
            // Generation is unhealthy or have not sent any generation request yet
            // Default to generation 
            let mut liveness_request = Request {
              id: LIVENESS_ID,
              inputs: "liveness".to_string(),
              tokenized_inputs: None,
              truncate: 10,
              prefill_logprobs: false,
              parameters: Some(NextTokenChooserParameters {
                  temperature: 1.0,
                  top_k: 0,
                  top_p: 1.0,
                  typical_p: 1.0,
                  do_sample: false,
                  seed: 0,
                  repetition_penalty: 1.0,
                  watermark: false,
                  adapter_id: "".to_string(),
                  schema: None,
                  return_k_alternatives: 0,
              }),
              stopping_parameters: Some(StoppingCriteriaParameters {
                  max_new_tokens: 1,
                  stop_sequences: vec![],
                  ignore_eos_token: false,
              }),
              adapter_index: 0,
              // Block 0 is reserved for health checks
              blocks: vec![0],
              slots: (0..16).collect(),
              cache_len: 0,
              chunk_len: None,
          };
            // Create different requestas based on the type of model this is 
            if self.shard_info().supports_embeddings {
              liveness_request = EmbedRequest {
                inputs: "San Francisco".to_string(),
                parameters: EmbedParameters {
                    adapter_id: None,
                    adapter_source: None,
                    adapter_parameters: None,
                    api_token: None,
                },
              }
            };
            
            if self.shard_info().supports_classification {
              liveness_request = ClassifyRequest {
                inputs: "San Francisco".to_string(),
              };
            }

            // Dummy batch of 1 token and 1 generated token
            let batch = Batch {
                id: BATCH_ID,
                requests: vec![liveness_request],
                size: 1,
                max_tokens: 2,
            };

            // Skips the queue
            if self.shard_info().supports_generation {
              let value = self.client.prefill(batch, None).await.is_ok();
              // Update generation health
              self.generation_health.store(value, Ordering::SeqCst);
              return value
            }

            if self.shard_info().supports_embeddings {
              let value = self.client.embed(batch).await.is_ok();
              // Update generation health
              self.generation_health.store(value, Ordering::SeqCst);
              return value
            }

            if self.shard_info().supports_classification {
              let value = self.client.classify(batch).await.is_ok();
              // Update generation health
              self.generation_health.store(value, Ordering::SeqCst);
              return value
            }
            // Return false - need to implement that shard type.
            return false
          }
        }
}
