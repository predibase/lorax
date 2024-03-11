/// LoRAX Webserver
mod adapter;
mod health;
mod infer;
mod loader;
mod queue;
mod scheduler;
pub mod server;
mod validation;
use lorax_client::{AdapterParameters as AdapterParametersMessage, AlternativeTokens};
use lorax_client::{MajoritySignMethod, MergeStrategy};

use infer::Infer;
use loader::AdapterLoader;
use queue::Entry;
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;
use validation::Validation;

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "bigscience/blomm-560m")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(nullable = true, example = "lorax")]
    pub model_pipeline_tag: Option<String>,
    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "2")]
    pub max_best_of: usize,
    #[schema(example = "4")]
    pub max_stop_sequences: usize,
    #[schema(example = "1024")]
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_total_tokens: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(example = "2")]
    pub validation_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema, Default)]
pub(crate) struct AdapterParameters {
    #[serde(rename(deserialize = "ids"))]
    #[schema(inline, example = json ! (["arnavgrg/codealpaca-qlora"]))]
    pub adapter_ids: Vec<String>,
    #[serde(default)]
    #[schema(inline, example = json ! ([0.25, 0.75]))]
    pub weights: Vec<f32>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "linear")]
    pub merge_strategy: Option<String>,
    #[serde(default)]
    #[schema(nullable = false, default = 0.0, example = 0.5)]
    pub density: f32,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "total")]
    pub majority_sign_method: Option<String>,
}

impl Into<AdapterParametersMessage> for AdapterParameters {
    fn into(self) -> AdapterParametersMessage {
        AdapterParametersMessage {
            adapter_ids: self.adapter_ids,
            weights: self.weights,
            merge_strategy: MergeStrategy::from_str_name(
                self.merge_strategy
                    .unwrap_or("linear".to_string())
                    .to_uppercase()
                    .as_str(),
            )
            .unwrap()
            .into(),
            density: self.density,
            majority_sign_method: MajoritySignMethod::from_str_name(
                self.majority_sign_method
                    .unwrap_or("total".to_string())
                    .to_uppercase()
                    .as_str(),
            )
            .unwrap()
            .into(),
        }
    }
}

impl std::hash::Hash for AdapterParameters {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.adapter_ids.len() == 1 {
            self.adapter_ids[0].hash(state);
            return;
        }

        self.adapter_ids.hash(state);

        // Convert weights vec into vec of u32 bits
        let weights: Vec<u32> = self.weights.iter().map(|x| x.to_bits()).collect();
        weights.hash(state);

        self.merge_strategy.hash(state);

        // Hash the raw bits of the float, acknowledging that this
        // can cause issues with different representations of the same value.
        self.density.to_bits().hash(state);

        self.majority_sign_method.hash(state);
    }
}

impl PartialEq for AdapterParameters {
    fn eq(&self, other: &Self) -> bool {
        if self.adapter_ids.len() == 1 {
            return self.adapter_ids[0] == other.adapter_ids[0];
        }

        // In this implementation, we assume that adapter order matters
        self.adapter_ids == other.adapter_ids
            && self.weights == other.weights
            && self.merge_strategy == other.merge_strategy
            && self.density == other.density // direct comparison of f32
            && self.majority_sign_method == other.majority_sign_method
    }
}

impl Eq for AdapterParameters {}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateParameters {
    #[serde(default)]
    #[schema(
        nullable = true,
        default = "null",
        example = "arnavgrg/codealpaca-qlora"
    )]
    pub adapter_id: Option<String>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "hub")]
    pub adapter_source: Option<String>,
    #[serde(rename(deserialize = "merged_adapters"))]
    #[schema(nullable = true, default = "null")]
    pub adapter_parameters: Option<AdapterParameters>,
    #[serde(default)]
    #[schema(
        nullable = true,
        default = "null",
        example = "<token for private adapters>"
    )]
    pub api_token: Option<String>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 1)]
    pub best_of: Option<usize>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 0.5
    )]
    pub temperature: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 1.03
    )]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub top_k: Option<i32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub top_p: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub typical_p: Option<f32>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    #[schema(exclusive_minimum = 0, exclusive_maximum = 512, default = "20")]
    pub max_new_tokens: u32,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = false)]
    pub return_full_text: Option<bool>,
    #[serde(default)]
    #[schema(inline, max_items = 4, example = json ! (["photographer"]))]
    pub stop: Vec<String>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub truncate: Option<usize>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub watermark: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub decoder_input_details: bool,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub return_k_alternatives: Option<i32>,
    #[serde(default)]
    #[schema(default = "false")]
    pub apply_chat_template: bool,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0,
        nullable = true,
        default = "null",
        example = "null"
    )]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schema(
        nullable = true,
        default = "null",
        example = json!(r#"{"type": "json_object", "schema": {type": "string", "title": "response"}}"#)
    )]
    pub response_format: Option<ResponseFormat>,
}

fn default_max_new_tokens() -> u32 {
    20
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        adapter_id: None,
        adapter_source: None,
        adapter_parameters: None,
        api_token: None,
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: false,
        max_new_tokens: default_max_new_tokens(),
        return_full_text: None,
        stop: Vec::new(),
        truncate: None,
        watermark: false,
        details: false,
        decoder_input_details: false,
        return_k_alternatives: None,
        apply_chat_template: false,
        seed: None,
        response_format: None,
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    #[serde(default)]
    #[schema(default = "false")]
    pub stream: bool,
}

impl From<CompatGenerateRequest> for GenerateRequest {
    fn from(req: CompatGenerateRequest) -> Self {
        Self {
            inputs: req.inputs,
            parameters: req.parameters,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PrefillToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct AlternativeToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Token {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
    #[schema(example = "false")]
    special: bool,
    #[schema(nullable = true)]
    #[serde(skip_serializing_if = "Option::is_none")]
    alternative_tokens: Option<Vec<AlternativeToken>>,
}

#[derive(Serialize, ToSchema)]
#[serde(rename_all(serialize = "snake_case"))]
pub(crate) enum FinishReason {
    #[schema(rename = "length")]
    Length,
    #[serde(rename = "eos_token")]
    #[schema(rename = "eos_token")]
    EndOfSequenceToken,
    #[schema(rename = "stop_sequence")]
    StopSequence,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BestOfSequence {
    #[schema(example = "test")]
    pub generated_text: String,
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Details {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub prompt_tokens: u32,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of_sequences: Option<Vec<BestOfSequence>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct GenerateResponse {
    #[schema(example = "test")]
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamDetails {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub prompt_tokens: u32,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamResponse {
    pub token: Token,
    #[schema(nullable = true, default = "null", example = "test")]
    pub generated_text: Option<String>,
    #[schema(nullable = true, default = "null")]
    pub details: Option<StreamDetails>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

// OpenAI compatible structs

#[derive(Serialize, ToSchema)]
struct UsageInfo {
    prompt_tokens: u32,
    total_tokens: u32,
    completion_tokens: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
enum ResponseFormatType {
    #[serde(alias = "json_object")]
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct ResponseFormat {
    r#type: ResponseFormatType,
    schema: serde_json::Value, // TODO: make this optional once arbitrary JSON object is supported in Outlines
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<std::collections::HashMap<String, String>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<i32>,
    max_tokens: Option<i32>,
    #[serde(default)]
    stop: Vec<String>,
    stream: Option<bool>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    user: Option<String>,
    // Additional parameters
    // TODO(travis): add other LoRAX params here
    response_format: Option<ResponseFormat>,
    repetition_penalty: Option<f32>,
    top_k: Option<i32>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct CompletionRequest {
    model: String,
    prompt: String,
    suffix: Option<String>,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<i32>,
    stream: Option<bool>,
    logprobs: Option<i32>,
    echo: Option<bool>,
    #[serde(default)]
    stop: Vec<String>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    best_of: Option<i32>,
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    user: Option<String>,
    // Additional parameters
    // TODO(travis): add other LoRAX params here
    repetition_penalty: Option<f32>,
    top_k: Option<i32>,
}

#[derive(Serialize, ToSchema)]
struct LogProbs {
    text_offset: Vec<i32>,
    token_logprobs: Vec<Option<f32>>,
    tokens: Vec<String>,
    top_logprobs: Option<Vec<Option<std::collections::HashMap<i32, f32>>>>,
}

#[derive(Serialize, ToSchema)]
struct CompletionResponseChoice {
    index: i32,
    text: String,
    logprobs: Option<LogProbs>,
    finish_reason: Option<CompletionFinishReason>,
}

#[derive(Serialize, ToSchema)]
struct CompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<CompletionResponseChoice>,
    usage: UsageInfo,
}

#[derive(Serialize, ToSchema)]
struct CompletionResponseStreamChoice {
    index: i32,
    text: String,
    logprobs: Option<LogProbs>,
    finish_reason: Option<CompletionFinishReason>,
}

#[derive(Serialize, ToSchema)]
struct CompletionStreamResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<CompletionResponseStreamChoice>,
    usage: Option<UsageInfo>,
}

#[derive(Serialize, ToSchema)]
struct ChatMessage {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct ChatCompletionResponseChoice {
    index: i32,
    message: ChatMessage,
    finish_reason: Option<CompletionFinishReason>,
}

#[derive(Serialize, ToSchema)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionResponseChoice>,
    usage: UsageInfo,
}

#[derive(Serialize, ToSchema)]
struct ChatCompletionStreamResponseChoice {
    index: i32,
    delta: ChatMessage,
    finish_reason: Option<CompletionFinishReason>,
}

#[derive(Serialize, ToSchema)]
struct ChatCompletionStreamResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionStreamResponseChoice>,
}

#[derive(Serialize, ToSchema, PartialEq)]
#[serde(rename_all(serialize = "snake_case"))]
pub(crate) enum CompletionFinishReason {
    #[schema(rename = "stop")]
    Stop,
    #[schema(rename = "length")]
    Length,
    #[schema(rename = "content_filter")]
    ContentFilter,
    #[schema(rename = "tool_calls")]
    ToolCalls,
}

impl From<CompletionRequest> for CompatGenerateRequest {
    fn from(req: CompletionRequest) -> Self {
        CompatGenerateRequest {
            inputs: req.prompt,
            parameters: GenerateParameters {
                adapter_id: req.model.parse().ok(),
                adapter_source: None,
                adapter_parameters: None,
                api_token: None,
                best_of: req.best_of.map(|x| x as usize),
                temperature: req.temperature,
                repetition_penalty: req.repetition_penalty,
                top_k: req.top_k,
                top_p: req.top_p,
                typical_p: None,
                do_sample: !req.n.is_none(),
                max_new_tokens: req
                    .max_tokens
                    .map(|x| x as u32)
                    .unwrap_or(default_max_new_tokens()),
                return_full_text: req.echo,
                stop: req.stop,
                truncate: None,
                watermark: false,
                details: true,
                decoder_input_details: req.logprobs.is_some(),
                return_k_alternatives: None,
                apply_chat_template: false,
                seed: None,
                response_format: None,
            },
            stream: req.stream.unwrap_or(false),
        }
    }
}

impl From<ChatCompletionRequest> for CompatGenerateRequest {
    fn from(req: ChatCompletionRequest) -> Self {
        CompatGenerateRequest {
            inputs: serde_json::to_string(&req.messages).unwrap(),
            parameters: GenerateParameters {
                adapter_id: req.model.parse().ok(),
                adapter_source: None,
                adapter_parameters: None,
                api_token: None,
                best_of: req.n.map(|x| x as usize),
                temperature: req.temperature,
                repetition_penalty: req.repetition_penalty,
                top_k: req.top_k,
                top_p: req.top_p,
                typical_p: None,
                do_sample: !req.n.is_none(),
                max_new_tokens: req
                    .max_tokens
                    .map(|x| x as u32)
                    .unwrap_or(default_max_new_tokens()),
                return_full_text: None,
                stop: req.stop,
                truncate: None,
                watermark: false,
                details: true,
                decoder_input_details: false,
                return_k_alternatives: None,
                apply_chat_template: true,
                seed: None,
                response_format: req.response_format,
            },
            stream: req.stream.unwrap_or(false),
        }
    }
}

impl From<GenerateResponse> for CompletionResponse {
    fn from(resp: GenerateResponse) -> Self {
        let prompt_tokens = resp.details.as_ref().map(|x| x.prompt_tokens).unwrap_or(0);
        let completion_tokens = resp
            .details
            .as_ref()
            .map(|x| x.generated_tokens)
            .unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        CompletionResponse {
            id: "null".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "null".to_string(),
            choices: vec![CompletionResponseChoice {
                index: 0,
                text: resp.generated_text,
                logprobs: None,
                finish_reason: resp
                    .details
                    .map(|x| CompletionFinishReason::from(x.finish_reason)),
            }],
            usage: UsageInfo {
                prompt_tokens: prompt_tokens,
                total_tokens: total_tokens,
                completion_tokens: Some(completion_tokens),
            },
        }
    }
}

impl From<StreamResponse> for CompletionStreamResponse {
    fn from(resp: StreamResponse) -> Self {
        let prompt_tokens = resp.details.as_ref().map(|x| x.prompt_tokens).unwrap_or(0);
        let completion_tokens = resp
            .details
            .as_ref()
            .map(|x| x.generated_tokens)
            .unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        let finish_reason = resp
            .details
            .map(|x| CompletionFinishReason::from(x.finish_reason));

        let is_stop = finish_reason
            .as_ref()
            .is_some_and(|x| x == &CompletionFinishReason::Stop);

        CompletionStreamResponse {
            id: "null".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "null".to_string(),
            choices: vec![CompletionResponseStreamChoice {
                index: 0,
                text: if is_stop {
                    "".to_string()
                } else {
                    resp.token.text
                },
                logprobs: None,
                finish_reason: finish_reason,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: prompt_tokens,
                total_tokens: total_tokens,
                completion_tokens: Some(completion_tokens),
            }),
        }
    }
}

impl From<GenerateResponse> for ChatCompletionResponse {
    fn from(resp: GenerateResponse) -> Self {
        let prompt_tokens = resp.details.as_ref().map(|x| x.prompt_tokens).unwrap_or(0);
        let completion_tokens = resp
            .details
            .as_ref()
            .map(|x| x.generated_tokens)
            .unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        ChatCompletionResponse {
            id: "null".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "null".to_string(),
            choices: vec![ChatCompletionResponseChoice {
                index: 0,
                message: ChatMessage {
                    role: Some("assistant".to_string()),
                    content: Some(resp.generated_text),
                },
                finish_reason: resp
                    .details
                    .map(|x| CompletionFinishReason::from(x.finish_reason)),
            }],
            usage: UsageInfo {
                prompt_tokens: prompt_tokens,
                total_tokens: total_tokens,
                completion_tokens: Some(completion_tokens),
            },
        }
    }
}

impl From<StreamResponse> for ChatCompletionStreamResponse {
    fn from(resp: StreamResponse) -> Self {
        let finish_reason = resp
            .details
            .map(|x| CompletionFinishReason::from(x.finish_reason));

        let is_stop = finish_reason
            .as_ref()
            .is_some_and(|x| x == &CompletionFinishReason::Stop);

        ChatCompletionStreamResponse {
            id: "null".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "null".to_string(),
            choices: vec![ChatCompletionStreamResponseChoice {
                index: 0,
                delta: ChatMessage {
                    role: if is_stop {
                        None
                    } else {
                        Some("assistant".to_string())
                    },
                    content: if is_stop { None } else { Some(resp.token.text) },
                },
                finish_reason: finish_reason,
            }],
        }
    }
}

impl From<FinishReason> for CompletionFinishReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::Length => CompletionFinishReason::Length,
            FinishReason::EndOfSequenceToken => CompletionFinishReason::Stop,
            FinishReason::StopSequence => CompletionFinishReason::ContentFilter,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tokenizers::Tokenizer;

    pub(crate) async fn get_tokenizer() -> Tokenizer {
        if !std::path::Path::new("tokenizer.json").exists() {
            let content = reqwest::get("https://huggingface.co/gpt2/raw/main/tokenizer.json")
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let mut file = std::fs::File::create("tokenizer.json").unwrap();
            file.write_all(&content).unwrap();
        }
        Tokenizer::from_file("tokenizer.json").unwrap()
    }
}
