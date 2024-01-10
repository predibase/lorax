/// LoRAX Webserver
mod adapter;
mod health;
mod infer;
mod loader;
mod queue;
mod scheduler;
pub mod server;
mod validation;

use infer::Infer;
use loader::AdapterLoader;
use queue::Entry;
use serde::{Deserialize, Serialize};
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
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "<token from predibase>")]
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
}

fn default_max_new_tokens() -> u32 {
    20
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        adapter_id: None,
        adapter_source: None,
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
        apply_chat_template: false,
        seed: None,
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
pub struct Token {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
    #[schema(example = "false")]
    special: bool,
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
    finish_reason: Option<String>,
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
    finish_reason: Option<String>,
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
    role: String,
    content: String,
}

#[derive(Serialize, ToSchema)]
struct ChatCompletionResponseChoice {
    index: i32,
    message: ChatMessage,
    finish_reason: Option<String>,
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

impl From<CompletionRequest> for CompatGenerateRequest {
    fn from(req: CompletionRequest) -> Self {
        CompatGenerateRequest {
            inputs: req.prompt,
            parameters: GenerateParameters {
                adapter_id: req.model.parse().ok(),
                adapter_source: None,
                api_token: None,
                best_of: req.best_of.map(|x| x as usize),
                temperature: req.temperature,
                repetition_penalty: None,
                top_k: None,
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
                apply_chat_template: false,
                seed: None,
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
                api_token: None,
                best_of: req.n.map(|x| x as usize),
                temperature: req.temperature,
                repetition_penalty: None,
                top_k: None,
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
                apply_chat_template: true,
                seed: None,
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
                finish_reason: None,
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

        CompletionStreamResponse {
            id: "null".to_string(),
            object: "text_completion".to_string(),
            created: 0,
            model: "null".to_string(),
            choices: vec![CompletionResponseStreamChoice {
                index: 0,
                text: resp.generated_text.unwrap_or_default(),
                logprobs: None,
                finish_reason: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: prompt_tokens,
                total_tokens: total_tokens,
                completion_tokens: Some(completion_tokens),
            }),
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
