/// Single shard Client
use crate::pb::generate::v1::lorax_service_client::LoraxServiceClient;
use crate::pb::generate::v1::*;
use crate::ClientError;
use crate::Result;
use crate::WARMUP_IMAGE_BASE64;
use base64::{engine::general_purpose::STANDARD, Engine};
use grpc_metadata::InjectTelemetryContext;
use std::cmp::min;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

use self::input_chunk::Chunk;

/// LoRAX gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: LoraxServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: LoraxServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: LoraxServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.stub.service_discovery(request).await?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            // Remove unix socket prefix
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    /// Get model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let request = tonic::Request::new(InfoRequest {}).inject_context();
        let response = self.stub.info(request).await?.into_inner();
        Ok(response)
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest { id: batch_id }).inject_context();
        self.stub.clear_cache(request).await?;
        Ok(())
    }

    /// Filter a cached batch
    #[instrument(skip(self))]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<Option<CachedBatch>> {
        let request = tonic::Request::new(FilterBatchRequest {
            batch_id,
            request_ids,
        })
        .inject_context();
        let filtered_batch = self.stub.filter_batch(request).await?.into_inner();
        Ok(filtered_batch.batch)
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip_all)]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
    ) -> Result<Option<u32>> {
        let mut n_tokens = 0;
        let mut requests = Vec::new();

        // Create requests
        while n_tokens < max_prefill_tokens {
            // We truncate the input on the server side to be sure that it has the correct size
            let truncate_length = min(max_input_length, max_prefill_tokens - n_tokens);

            let mut input_chunks = Vec::new();
            input_chunks
                .push(Chunk::Text("_test ".to_string().repeat(max_input_length as usize)).into());
            if n_tokens == 0 {
                input_chunks.push(
                    Chunk::Image(Image {
                        // Safe unwrap, because we control the data.
                        data: STANDARD.decode(WARMUP_IMAGE_BASE64).unwrap(),
                        mimetype: "image/jpeg;base64".to_string(),
                    })
                    .into(),
                );
            }

            requests.push(Request {
                id: 0,
                inputs: "_test ".to_string().repeat(max_input_length as usize),
                tokenized_inputs: Some(TokenizedInputs {
                    ids: vec![],
                    input_chunks: input_chunks,
                }),
                truncate: truncate_length,
                // Blocks and slots will be set on the server side if we use paged attention
                blocks: vec![],
                slots: vec![],
                cache_len: 0,
                chunk_len: None,
                // Set sampling parameters to also take these ops into account in the max memory
                parameters: Some(NextTokenChooserParameters {
                    temperature: 0.9,
                    top_k: 10,
                    top_p: 0.9,
                    typical_p: 0.9,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 1.2,
                    frequency_penalty: 0.5,
                    presence_penalty: 0.5,
                    watermark: true,
                    adapter_id: "".to_string(),
                    schema: None,
                    return_k_alternatives: 0,
                }),
                stopping_parameters: Some(StoppingCriteriaParameters {
                    max_new_tokens: max_total_tokens - truncate_length,
                    stop_sequences: vec![],
                    ignore_eos_token: false,
                }),
                adapter_index: 0,
                prefill_logprobs: true,
            });
            n_tokens += max_input_length;
        }

        let batch = Batch {
            id: 0,
            size: requests.len() as u32,
            requests,
            max_tokens: max_input_length,
            max_blocks: 0,
        };

        let max_new_tokens = max_total_tokens - max_input_length;
        let request = tonic::Request::new(WarmupRequest {
            batch: Some(batch),
            max_input_length,
            max_prefill_tokens,
            max_new_tokens,
        })
        .inject_context();
        let response = self.stub.warmup(request).await?.into_inner();
        Ok(response.max_supported_total_tokens)
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
        cached_batch: Option<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let request = tonic::Request::new(PrefillRequest {
            batch: Some(batch),
            cached_batch,
        })
        .inject_context();
        let response = self.stub.prefill(request).await?.into_inner();
        Ok((response.generations, response.batch))
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = batches.iter().map(|batch|{batch.size}).sum::<u32>()))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let request = tonic::Request::new(DecodeRequest { batches }).inject_context();
        let response = self.stub.decode(request).await?.into_inner();
        Ok((response.generations, response.batch))
    }

    /// Embed
    #[instrument(skip(self))]
    pub async fn embed(&mut self, batch: Batch) -> Result<Vec<Embedding>> {
        let request = tonic::Request::new(EmbedRequest { batch: Some(batch) }).inject_context();
        let response = self.stub.embed(request).await?.into_inner();
        Ok(response.embeddings)
    }

    /// Classify
    #[instrument(skip(self))]
    pub async fn classify(&mut self, batch: Batch) -> Result<Vec<ClassifyPredictionList>> {
        let request = tonic::Request::new(ClassifyRequest { batch: Some(batch) }).inject_context();
        let response = self.stub.classify(request).await?.into_inner();
        Ok(response.classify_prediction_lists)
    }

    /// Downloads the weights for an adapter.
    pub async fn download_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        api_token: Option<String>,
    ) -> Result<DownloadAdapterResponse> {
        if let Some(adapter_source_enum) =
            AdapterSource::from_str_name(adapter_source.to_uppercase().as_str())
        {
            let request = tonic::Request::new(DownloadAdapterRequest {
                adapter_parameters: Some(adapter_parameters),
                adapter_source: adapter_source_enum.into(),
                api_token: api_token,
            })
            .inject_context();
            let response = self.stub.download_adapter(request).await?.into_inner();
            Ok(response)
        } else {
            let err_string = format!(
                "Invalid source '{}' when downloading adapter '{}'",
                adapter_source,
                adapter_parameters.adapter_ids.join(",")
            );
            tracing::error!(err_string);
            Err(ClientError::Generation(err_string).into())
        }
    }

    /// Physically loads the weights into the model for an adapter
    pub async fn load_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        adapter_index: u32,
        api_token: Option<String>,
    ) -> Result<bool> {
        if let Some(adapter_source_enum) =
            AdapterSource::from_str_name(adapter_source.to_uppercase().as_str())
        {
            let request = tonic::Request::new(LoadAdapterRequest {
                adapter_parameters: Some(adapter_parameters),
                adapter_source: adapter_source_enum.into(),
                adapter_index,
                api_token: api_token,
            })
            .inject_context();
            let response = self.stub.load_adapter(request).await?.into_inner();
            Ok(response.loaded)
        } else {
            let err_string = format!(
                "Invalid source '{}' when loading adapter '{}'",
                adapter_source,
                adapter_parameters.adapter_ids.join(",")
            );
            tracing::error!(err_string);
            Err(ClientError::Generation(err_string).into())
        }
    }

    /// Offloads adapter the weights from GPU to CPU or disk
    pub async fn offload_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        adapter_index: u32,
    ) -> Result<bool> {
        if let Some(adapter_source_enum) =
            AdapterSource::from_str_name(adapter_source.to_uppercase().as_str())
        {
            let request = tonic::Request::new(OffloadAdapterRequest {
                adapter_parameters: Some(adapter_parameters),
                adapter_source: adapter_source_enum.into(),
                adapter_index,
            })
            .inject_context();
            let response = self.stub.offload_adapter(request).await?.into_inner();
            Ok(response.offloaded)
        } else {
            let err_string = format!(
                "Invalid source '{}' when loading adapter '{}'",
                adapter_source,
                adapter_parameters.adapter_ids.join(",")
            );
            tracing::error!(err_string);
            Err(ClientError::Generation(err_string).into())
        }
    }
}
