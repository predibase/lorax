use crate::pb::generate::v1::{
    ClassifyPredictionList, EmbedResponse, Embedding, Entity, EntityList,
};
/// Multi shard Client
use crate::{
    AdapterParameters, Batch, CachedBatch, Client, DownloadAdapterResponse, Generation,
    HealthResponse, ShardInfo,
};
use crate::{ClientError, Result};
use futures::future::join_all;
use tokenizers::tokenizer::Tokenizer;
use tonic::transport::Uri;
use tracing::instrument;

#[derive(Debug, Clone)]
/// LoRAX gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
    tokenizer: Option<Tokenizer>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>, tokenizer: Option<Tokenizer>) -> Self {
        Self { clients, tokenizer }
        // Self { clients }
    }

    /// Create a new ShardedClient from a master client. The master client will communicate with
    /// the other shards and returns all uris/unix sockets with the `service_discovery` gRPC method.
    async fn from_master_client(
        mut master_client: Client,
        tokenizer: Option<Tokenizer>,
    ) -> Result<Self> {
        // Get all uris/unix sockets from the master client
        let uris = master_client.service_discovery().await?;
        let futures = uris.into_iter().map(Client::connect_uds);
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?, tokenizer))
    }

    /// Returns a client connected to the given uri
    pub async fn connect(uri: Uri, tokenizer: Option<Tokenizer>) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client, tokenizer).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String, tokenizer: Option<Tokenizer>) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client, tokenizer).await
    }

    /// Get the model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<ShardInfo> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.info())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// GRPC health check
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.health())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache(batch_id))
            .collect();
        join_all(futures).await.into_iter().collect()
    }

    /// Filter a cached batch
    #[instrument(skip(self))]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<Option<CachedBatch>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.filter_batch(batch_id, request_ids.clone())))
            .collect();
        // all shards return the same message
        join_all(futures).await.pop().unwrap()
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip(self))]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
    ) -> Result<Option<u32>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| {
                Box::pin(client.warmup(max_input_length, max_prefill_tokens, max_total_tokens))
            })
            .collect();
        // Take the minimum value
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<Option<u32>>>>()?;
        Ok(results.into_iter().flatten().min())
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.prefill(batch.clone())))
            .collect();
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>)>> =
            join_all(futures).await.into_iter().collect();
        merge_generations(results?)
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
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.decode(batches.clone())))
            .collect();
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>)>> =
            join_all(futures).await.into_iter().collect();
        merge_generations(results?)
    }

    /// Embed the given batch
    #[instrument(skip(self))]
    pub async fn embed(&mut self, batch: Batch) -> Result<Vec<Embedding>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.embed(batch.clone())))
            .collect();
        let results: Result<Vec<Vec<Embedding>>> = join_all(futures).await.into_iter().collect();
        Ok(results?.into_iter().flatten().collect())
    }

    /// Classify the given batch
    #[instrument(skip(self))]
    pub async fn classify(&mut self, batch: Batch) -> Result<Vec<EntityList>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.classify(batch.clone())))
            .collect();
        let results: Result<Vec<Vec<ClassifyPredictionList>>> =
            join_all(futures).await.into_iter().collect();

        let flat_results: Vec<ClassifyPredictionList> = results?.into_iter().flatten().collect();
        let entity_lists: Vec<EntityList> = flat_results
            .into_iter()
            .map(|prediction| {
                let entities =
                    format_ner_output(prediction.clone(), self.tokenizer.as_ref().unwrap());
                EntityList {
                    request_id: prediction.request_id,
                    entities,
                    input_ids: prediction.input_ids,
                }
            })
            .collect();
        Ok(entity_lists)
    }

    pub async fn download_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        api_token: Option<String>,
    ) -> Result<DownloadAdapterResponse> {
        // Only download the adapter with one client, since they share a single disk
        self.clients[0]
            .download_adapter(adapter_parameters, adapter_source, api_token)
            .await
    }

    pub async fn load_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        adapter_index: u32,
        api_token: Option<String>,
    ) -> Result<bool> {
        // Load the adapter in all clients since there is sharding done between them
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| {
                Box::pin(client.load_adapter(
                    adapter_parameters.clone(),
                    adapter_source.clone(),
                    adapter_index,
                    api_token.clone(),
                ))
            })
            .collect();

        match join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<bool>>>()
        {
            Ok(mut results) => {
                // Return the first adapter id
                Ok(results.pop().unwrap())
            }
            Err(err) => Err(err),
        }
    }

    pub async fn offload_adapter(
        &mut self,
        adapter_parameters: AdapterParameters,
        adapter_source: String,
        adapter_index: u32,
    ) -> Result<bool> {
        // Load the adapter in all clients since there is sharding done between them
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| {
                Box::pin(client.offload_adapter(
                    adapter_parameters.clone(),
                    adapter_source.clone(),
                    adapter_index,
                ))
            })
            .collect();

        match join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<bool>>>()
        {
            Ok(mut results) => {
                // Return the first adapter id
                Ok(results.pop().unwrap())
            }
            Err(err) => Err(err),
        }
    }
}

/// Merge generations from the different model shards
fn merge_generations(
    mut results: Vec<(Vec<Generation>, Option<CachedBatch>)>,
) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
    let (mut generations, next_batch) = results.pop().ok_or(ClientError::EmptyResults)?;

    for (mut shard_generations, _) in results.into_iter() {
        generations.append(&mut shard_generations);
    }
    Ok((generations, next_batch))
}

fn decode_tokens(input_ids: Vec<u32>, tokenizer: &Tokenizer) -> Result<String> {
    let tokens = tokenizer.decode(&input_ids, false).unwrap();
    Ok(tokens)
}

fn format_ner_output(
    classify_prediction_list: ClassifyPredictionList,
    tokenizer: &Tokenizer,
) -> Vec<Entity> {
    let input_ids =
        &classify_prediction_list.input_ids[1..classify_prediction_list.input_ids.len() - 1];
    let predicted_token_class =
        &classify_prediction_list.predictions[1..classify_prediction_list.predictions.len() - 1];
    let scores = &classify_prediction_list.scores[1..classify_prediction_list.scores.len() - 1];
    let tokens: Vec<String> = tokenizer
        .decode(input_ids, true)
        .unwrap()
        .split_whitespace()
        .map(String::from)
        .collect();

    let mut ner_results = Vec::new();
    let mut current_entity: Option<Entity> = None;
    for (i, ((token, token_class), score)) in tokens
        .iter()
        .zip(predicted_token_class.iter())
        .zip(scores.iter())
        .enumerate()
    {
        if token_class != "O" {
            let (bi, tag) = get_tag(token_class);
            if bi == "B"
                || (current_entity.is_some() && tag != current_entity.as_ref().unwrap().entity)
            {
                if let Some(entity) = current_entity {
                    ner_results.push(entity);
                }
                current_entity = Some(Entity {
                    entity: tag,
                    score: *score,
                    index: i as u32,
                    word: token.to_string(),
                    start: tokenizer.decode(&input_ids[..i], false).unwrap().len() as u32,
                    end: tokenizer.decode(&input_ids[..=i], false).unwrap().len() as u32,
                });
            } else if bi == "I" && current_entity.is_some() {
                let entity = current_entity.as_mut().unwrap();
                entity.word += &token.replace("##", "");
                entity.end = tokenizer.decode(&input_ids[..=i], false).unwrap().len() as u32;
            }
        } else if let Some(entity) = current_entity.take() {
            ner_results.push(entity);
        }
    }
    if let Some(entity) = current_entity {
        ner_results.push(entity);
    }
    ner_results
}

fn get_tag(token_class: &str) -> (String, String) {
    let parts: Vec<&str> = token_class.split('-').collect();
    if parts.len() == 2 {
        (parts[0].to_string(), parts[1].to_string())
    } else {
        ("O".to_string(), "O".to_string())
    }
}
