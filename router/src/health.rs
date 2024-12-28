use lorax_client::{
    Batch, NextTokenChooserParameters, Request, ShardInfo, ShardedClient,
    StoppingCriteriaParameters,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Note: Request ids and batch ids cannot collide.
#[allow(dead_code)]
const LIVENESS_ID: u64 = u64::MAX;
const BATCH_ID: u64 = u64::MAX;

#[derive(Clone, Debug)]
pub(crate) struct Health {
    client: ShardedClient,
    inference_health: Arc<AtomicBool>,
    shard_info: ShardInfo,
}

impl Health {
    pub(crate) fn new(
        client: ShardedClient,
        inference_health: Arc<AtomicBool>,
        shard_info: ShardInfo,
    ) -> Self {
        Self {
            #[allow(dead_code)]
            client,
            #[allow(dead_code)]
            inference_health,
            shard_info,
        }
    }

    pub(crate) fn shard_info(&self) -> &ShardInfo {
        &self.shard_info
    }

    pub(crate) async fn check_generation(&mut self) -> bool {
        let generation_liveness_request = Request {
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
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
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
        let batch = Batch {
            id: BATCH_ID,
            requests: vec![generation_liveness_request],
            size: 1,
            max_tokens: 2,
            max_blocks: 1,
        };
        // Skips the queue
        self.client.prefill(batch, None).await.is_ok()
    }

    pub(crate) async fn check_classification(&mut self) -> bool {
        let classify_request = Request {
            id: LIVENESS_ID,
            inputs: "San Francisco".to_string(),
            tokenized_inputs: None,
            truncate: 10,
            prefill_logprobs: false,
            parameters: None,
            stopping_parameters: None,
            adapter_index: 0,
            blocks: vec![0],
            slots: (0..16).collect(),
            cache_len: 0,
            chunk_len: None,
        };
        let batch = Batch {
            id: BATCH_ID,
            requests: vec![classify_request],
            size: 1,
            max_tokens: 2,
            max_blocks: 1,
        };
        self.client.classify(batch).await.is_ok()
    }

    pub(crate) async fn check_embeddings(&mut self) -> bool {
        let embed_request = Request {
            id: LIVENESS_ID,
            inputs: "San Francisco".to_string(),
            tokenized_inputs: None,
            truncate: 10,
            prefill_logprobs: false,
            parameters: None,
            stopping_parameters: None,
            adapter_index: 0,
            blocks: vec![0],
            slots: (0..16).collect(),
            cache_len: 0,
            chunk_len: None,
        };
        let batch = Batch {
            id: BATCH_ID,
            requests: vec![embed_request],
            size: 1,
            max_tokens: 2,
            max_blocks: 1,
        };
        self.client.embed(batch).await.is_ok()
    }

    #[allow(dead_code)]
    pub(crate) async fn check(&mut self) -> bool {
        if self.inference_health.load(Ordering::SeqCst) {
            // Generation is healthy, we only check that the shards are answering gRPC calls
            self.client.health().await.is_ok()
        } else {
            // Generation is unhealthy or have not sent any generation request yet
            let mut value = false;
            // Dummy batch of 1 token and 1 generated token
            if self.shard_info().supports_generation {
                value = self.check_generation().await;
                // Update generation health
                self.inference_health.store(value, Ordering::SeqCst);
            }

            if self.shard_info().supports_classification {
                value = self.check_classification().await;
                // Update generation health
                self.inference_health.store(value, Ordering::SeqCst);
            }

            if self.shard_info().supports_embeddings {
                value = self.check_embeddings().await;
                // Update generation health
                self.inference_health.store(value, Ordering::SeqCst);
            }

            value
        }
    }
}
