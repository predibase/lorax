/// LoRAX Webserver
mod adapter;
mod batch;
mod block_allocator;
pub mod config;
mod health;
mod infer;
mod loader;
mod queue;
mod radix;
mod scheduler;
pub mod server;
mod tool_grammar;

mod validation;
use lorax_client::{AdapterParameters as AdapterParametersMessage, Entity as EntityMessage};
use lorax_client::{MajoritySignMethod, MergeStrategy};

use batch::Entry;
use infer::{Infer, InferError};
use loader::AdapterLoader;
use serde::{Deserialize, Serialize};
use serde_json::json;
use server::prepare_chat_input;
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
    #[schema(example = false)]
    pub eager_prefill: bool,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
    #[schema(nullable = true, example = "http://localhost:8899")]
    pub request_logger_url: Option<String>,
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
    #[schema(
        exclusive_minimum = -2.0,
        exclusive_maximum = 2.0,
        nullable = true,
        default = "null",
        example = 0.1
    )]
    pub frequency_penalty: Option<f32>,

    #[serde(default)]
    #[schema(
        exclusive_minimum = -2.0,
        exclusive_maximum = 2.0,
        nullable = true,
        default = "null",
        example = 0.1
    )]
    pub presence_penalty: Option<f32>,

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
    #[serde(default)]
    #[schema(exclusive_minimum = 0, default = "null")]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub ignore_eos_token: bool,
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
    #[allow(dead_code)] // For now allow this field even though it is unused
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

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        adapter_id: None,
        adapter_source: None,
        adapter_parameters: None,
        api_token: None,
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        frequency_penalty: None,
        presence_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: false,
        max_new_tokens: None,
        ignore_eos_token: false,
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

    /// This is used internally because some requests
    /// already contain the templated input therefore
    /// we shouldn't add the special tokens.
    #[serde(default = "default_true", skip)]
    pub add_special_tokens: bool,
}

fn default_true() -> bool {
    true
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

    /// This is used internally because some requests
    /// already contain the templated input therefore
    /// we shouldn't add the special tokens.
    #[serde(default = "default_true", skip)]
    pub add_special_tokens: bool,
}

impl From<CompatGenerateRequest> for GenerateRequest {
    fn from(req: CompatGenerateRequest) -> Self {
        Self {
            inputs: req.inputs,
            parameters: req.parameters,
            add_special_tokens: req.add_special_tokens,
        }
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct TokenizeRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[schema(nullable = true, example = true)]
    pub add_special_tokens: Option<bool>,
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

#[derive(Debug, Serialize, ToSchema)]
pub struct SimpleToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(example = 0)]
    start: usize,
    #[schema(example = 2)]
    stop: usize,
}

#[derive(Serialize, ToSchema, Clone)]
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
#[serde(transparent)]
pub(crate) struct TokenizeResponse(Vec<SimpleToken>);

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
    #[serde(alias = "text")]
    Text,
    #[serde(alias = "json_object")]
    JsonObject,
    #[serde(alias = "json_schema")]
    JsonSchema,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct ResponseFormat {
    #[allow(dead_code)] // For now allow this field even though it is unused
    r#type: ResponseFormatType,

    #[serde(default = "default_json_schema")]
    schema: Option<serde_json::Value>,
}

// Default schema to be used when no value is provided
fn default_json_schema() -> Option<serde_json::Value> {
    Some(serde_json::json!({
        "additionalProperties": {
            "type": ["object", "string", "integer", "number", "boolean", "null"]
        },
        "title": "ArbitraryJsonModel",
        "type": "object"
    }))
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
struct JsonSchema {
    #[allow(dead_code)] // For now allow this field even though it is unused
    description: Option<String>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    name: String,
    schema: Option<serde_json::Value>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    strict: Option<bool>,
}

// TODO check if json_schema field is required if type is json_schema
#[derive(Clone, Debug, Deserialize, ToSchema)]
struct OpenAiResponseFormat {
    #[serde(rename(deserialize = "type"))]
    response_format_type: ResponseFormatType,
    json_schema: Option<JsonSchema>,

    // For backwards compatibility
    #[serde(default = "default_json_schema")]
    schema: Option<serde_json::Value>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct Url {
    url: String,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default, Debug, PartialEq)]
pub(crate) struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum MessageChunk {
    Text { text: String },
    ImageUrl { image_url: Url },
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct Message {
    #[schema(example = "user")]
    role: String,
    #[schema(example = "My name is David and I")]
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "\"David\"")]
    name: Option<String>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    SingleText(String),
    MultipleChunks(Vec<MessageChunk>),
}

// Pushing a chunk to a single text message will convert it to a multiple chunks message
impl MessageContent {
    pub fn push(&mut self, chunk: MessageChunk) {
        match self {
            MessageContent::SingleText(text) => {
                *self =
                    MessageContent::MultipleChunks(vec![MessageChunk::Text { text: text.clone() }]);
            }
            MessageContent::MultipleChunks(chunks) => {
                chunks.push(chunk);
            }
        }
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct TextMessage {
    #[schema(example = "user")]
    pub role: String,
    #[schema(example = "My name is David and I")]
    pub content: String,
}

impl From<Message> for TextMessage {
    fn from(value: Message) -> Self {
        TextMessage {
            role: value.role,
            content: match value.content {
                MessageContent::SingleText(text) => text,
                MessageContent::MultipleChunks(chunks) => chunks
                    .into_iter()
                    .map(|chunk| match chunk {
                        MessageChunk::Text { text } => text,
                        MessageChunk::ImageUrl { image_url } => format!("![]({})", image_url.url),
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<i32>,
    max_tokens: Option<i32>,
    #[serde(default)]
    stop: Vec<String>,
    stream: Option<bool>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    presence_penalty: Option<f32>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    frequency_penalty: Option<f32>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    user: Option<String>,
    seed: Option<u64>,
    response_format: Option<OpenAiResponseFormat>,

    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of
    /// functions the model may generate JSON inputs for.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub tools: Option<Vec<Tool>>,

    /// A specific tool to use. If not provided, the model will default to use any of the tools provided in the tools parameter.
    /// A specific tool to use. If not provided, the model will default to use any of the tools provided in the tools parameter.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub tool_choice: ToolChoice,
    // Additional parameters
    // TODO(travis): add other LoRAX params here
    repetition_penalty: Option<f32>,
    top_k: Option<i32>,
    ignore_eos_token: Option<bool>,
    adapter_source: Option<String>,
    api_token: Option<String>,

    /// A prompt to be appended before the tools
    #[serde(default)]
    #[schema(
        nullable = true,
        example = "Given the functions available, please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {name: function name, parameters: dictionary of argument name and its value}.Do not use variables."
    )]
    pub tool_prompt: Option<String>,

    /// A guideline to be used in the chat_template
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub guideline: Option<String>,
}

impl ChatCompletionRequest {
    fn try_into_generate(self, infer: &Infer) -> Result<(CompatGenerateRequest, bool), InferError> {
        let ChatCompletionRequest {
            model,
            max_tokens,
            messages,
            seed,
            stop,
            stream,
            tools,
            tool_choice,
            tool_prompt,
            temperature,
            response_format,
            guideline,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            top_p,
            top_k,
            n,
            adapter_source,
            api_token,
            ignore_eos_token,
            ..
        } = self;

        let mut adapter_id = Some(model.clone());
        if model == "" {
            adapter_id = None;
        }

        // Modify input values to ResponseFormat to be OpenAI API compatible
        let response_format: Option<ResponseFormat> = match response_format {
            None => None,
            Some(openai_format) => {
                let response_format_type = openai_format.response_format_type.clone();
                match response_format_type {
                    // Ignore when type is text
                    ResponseFormatType::Text => None,

                    // For json_object, use the fixed schema.
                    // For backwards compatibility, also support non-standard `schema` field
                    ResponseFormatType::JsonObject => openai_format.schema.map_or_else(
                        || {
                            Some(ResponseFormat {
                                r#type: response_format_type.clone(),
                                schema: default_json_schema(),
                            })
                        },
                        |schema_value: serde_json::Value| {
                            Some(ResponseFormat {
                                r#type: response_format_type.clone(),
                                schema: Some(schema_value),
                            })
                        },
                    ),

                    // For json_schema, use schema_value if available, otherwise fallback to the fixed schema
                    ResponseFormatType::JsonSchema => openai_format
                        .json_schema
                        .and_then(|schema| schema.schema)
                        .map_or_else(
                            || {
                                Some(ResponseFormat {
                                    r#type: response_format_type.clone(),
                                    schema: default_json_schema(),
                                })
                            },
                            |schema_value: serde_json::Value| {
                                Some(ResponseFormat {
                                    r#type: response_format_type.clone(),
                                    schema: Some(schema_value),
                                })
                            },
                        ),
                }
            }
        };

        let tool_prompt = tool_prompt
            .filter(|s| !s.is_empty())
            .unwrap_or_else(default_tool_prompt);

        // enable greedy only when temperature is 0
        let (do_sample, temperature) = match temperature {
            Some(temperature) if temperature == 0.0 => (false, None),
            other => (true, other),
        };

        let (inputs, response_format, using_tools) = prepare_chat_input(
            &infer,
            response_format,
            tools,
            tool_choice,
            &tool_prompt,
            guideline,
            messages,
        )?;

        Ok((
            CompatGenerateRequest {
                inputs: inputs.to_string(),
                add_special_tokens: false,
                parameters: GenerateParameters {
                    adapter_id,
                    adapter_source,
                    adapter_parameters: None,
                    api_token,
                    best_of: n.map(|x| x as usize),
                    temperature,
                    repetition_penalty,
                    frequency_penalty,
                    presence_penalty,
                    top_k,
                    top_p,
                    typical_p: None,
                    do_sample,
                    max_new_tokens: max_tokens.map(|x| x as u32),
                    return_full_text: None,
                    stop,
                    truncate: None,
                    watermark: false,
                    details: true,
                    decoder_input_details: false,
                    seed,
                    ignore_eos_token: ignore_eos_token.unwrap_or(false),
                    return_k_alternatives: None,
                    apply_chat_template: false,
                    response_format,
                },
                stream: stream.unwrap_or(false),
            },
            using_tools,
        ))
    }
}

pub fn default_tool_prompt() -> String {
    "\nGiven the functions available, please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {name: function name, parameters: dictionary of argument name and its value}.Do not use variables.\n".to_string()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ToSchema)]
#[schema(example = "auto")]
/// Controls which (if any) tool is called by the model.
pub enum ToolType {
    /// Means the model can pick between generating a message or calling one or more tools.
    #[schema(rename = "auto")]
    OneOf,
    /// Means the model will not call any tool and instead generates a message.
    #[schema(rename = "none")]
    NoTool,
    /// Forces the model to call a specific tool.
    #[schema(rename = "function")]
    Function(FunctionName),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, ToSchema)]
#[serde(from = "ToolTypeDeserializer")]
pub struct ToolChoice(pub Option<ToolType>);

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolTypeDeserializer {
    Null,
    String(String),
    ToolType(ToolType),
}

impl From<ToolTypeDeserializer> for ToolChoice {
    fn from(value: ToolTypeDeserializer) -> Self {
        match value {
            ToolTypeDeserializer::Null => ToolChoice(None),
            ToolTypeDeserializer::String(s) => match s.as_str() {
                "none" => ToolChoice(Some(ToolType::NoTool)),
                "auto" => ToolChoice(Some(ToolType::OneOf)),
                _ => ToolChoice(Some(ToolType::Function(FunctionName { name: s }))),
            },
            ToolTypeDeserializer::ToolType(tool_type) => ToolChoice(Some(tool_type)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, ToSchema, PartialEq)]
pub struct JsonSchemaTool {
    #[serde(flatten)]
    functions_map: FunctionsMap,
    properties: Properties,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct FunctionsMap {
    #[serde(rename = "$functions")]
    functions: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct FunctionRef {
    #[serde(rename = "$ref")]
    ref_path: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Properties {
    #[serde(serialize_with = "serialize_function")]
    function: Vec<FunctionRef>,
}

fn serialize_function<S>(functions: &Vec<FunctionRef>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeStruct;
    let mut state = serializer.serialize_struct("Function", 1)?;
    state.serialize_field("anyOf", functions)?;
    state.end()
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, Default, PartialEq)]
pub(crate) struct FunctionDefinition {
    #[serde(default)]
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "parameters")]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Tool {
    // The type of the tool. Currently, only 'function' is supported.
    #[schema(example = "function")]
    pub r#type: String,
    // Grab the tool as generic JSON for debugging purposes.
    pub function: FunctionDefinition,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[allow(dead_code)] // For now allow this field even though it is unused
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
    #[allow(dead_code)] // For now allow this field even though it is unused
    presence_penalty: Option<f32>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    frequency_penalty: Option<f32>,
    best_of: Option<i32>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[allow(dead_code)] // For now allow this field even though it is unused
    user: Option<String>,
    seed: Option<u64>,
    // Additional parameters
    // TODO(travis): add other LoRAX params here
    repetition_penalty: Option<f32>,
    top_k: Option<i32>,
    ignore_eos_token: Option<bool>,
    adapter_source: Option<String>,
    api_token: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
pub(crate) struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
pub(crate) struct Function {
    pub name: Option<String>,
    pub arguments: String,
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
    system_fingerprint: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageInfo>,
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
    #[allow(dead_code)] // For now allow this field even though it is unused
    ToolCalls,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct EmbedParameters {
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
}

fn default_embed_parameters() -> EmbedParameters {
    EmbedParameters {
        adapter_id: None,
        adapter_source: None,
        adapter_parameters: None,
        api_token: None,
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct EmbedRequest {
    inputs: String,
    #[serde(default = "default_embed_parameters")]
    pub parameters: EmbedParameters,
}

#[derive(Serialize, ToSchema)]
struct EmbedResponse {
    embeddings: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct ClassifyRequest {
    inputs: String,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
struct BatchClassifyRequest {
    inputs: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Entity {
    entity_group: String,
    score: f32,
    word: String,
    start: usize,
    end: usize,
}

impl From<EntityMessage> for Entity {
    fn from(entity: EntityMessage) -> Self {
        Entity {
            entity_group: entity.entity,
            score: entity.score,
            word: entity.word,
            start: entity.start as usize,
            end: entity.end as usize,
        }
    }
}

impl From<CompletionRequest> for CompatGenerateRequest {
    fn from(req: CompletionRequest) -> Self {
        CompatGenerateRequest {
            inputs: req.prompt,
            add_special_tokens: true,
            parameters: GenerateParameters {
                adapter_id: req.model.parse().ok(),
                adapter_source: req.adapter_source,
                adapter_parameters: None,
                api_token: req.api_token,
                best_of: req.best_of.map(|x| x as usize),
                temperature: req.temperature,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                presence_penalty: req.presence_penalty,
                top_k: req.top_k,
                top_p: req.top_p,
                typical_p: None,
                do_sample: !req.n.is_none(),
                max_new_tokens: req.max_tokens.map(|x| x as u32),
                ignore_eos_token: req.ignore_eos_token.unwrap_or(false),
                return_full_text: req.echo,
                stop: req.stop,
                truncate: None,
                watermark: false,
                details: true,
                decoder_input_details: req.logprobs.is_some(),
                return_k_alternatives: None,
                apply_chat_template: false,
                seed: req.seed,
                response_format: None,
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

impl ChatCompletionResponse {
    pub(crate) fn new(
        resp: &GenerateResponse,
        model: String,
        system_fingerprint: String,
        choice_content: Vec<(Option<Vec<ToolCall>>, Option<String>)>,
        created: u64,
        // return_logprobs: bool,  // TODO: Implement logprobs
    ) -> Self {
        let mut choices = vec![];
        for (index, (tool_calls, output)) in choice_content.into_iter().enumerate() {
            let message = match (output, tool_calls) {
                (Some(content), None) => ChatMessage {
                    role: Some("assistant".to_string()),
                    content: Some(content),
                    tool_calls: None,
                },
                (None, Some(tool_calls)) => ChatMessage {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: Some(tool_calls),
                },
                (Some(output), Some(_)) => {
                    tracing::warn!("Received both chat and tool call");
                    ChatMessage {
                        role: Some("assistant".to_string()),
                        content: Some(output),
                        tool_calls: None,
                    }
                }
                (None, None) => {
                    tracing::warn!("Didn't receive an answer");
                    ChatMessage {
                        role: Some("assistant".to_string()),
                        content: None,
                        tool_calls: None,
                    }
                }
            };

            choices.push(ChatCompletionResponseChoice {
                index: index as i32,
                message,
                finish_reason: Some(CompletionFinishReason::from(
                    resp.details.as_ref().unwrap().finish_reason.clone(),
                )),
            });
        }

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
            created: created as i64,
            system_fingerprint: system_fingerprint,
            model: model,
            choices: choices,
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
        let prompt_tokens = resp.details.as_ref().map(|x| x.prompt_tokens).unwrap_or(0);
        let completion_tokens = resp.details.as_ref().map(|x| x.generated_tokens);
        let total_tokens = prompt_tokens + completion_tokens.unwrap_or(0);

        let usage: Option<UsageInfo> = if completion_tokens.is_some() {
            Some(UsageInfo {
                prompt_tokens: prompt_tokens,
                total_tokens: total_tokens,
                completion_tokens: completion_tokens,
            })
        } else {
            None
        };

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
                    tool_calls: None, // TODO(travis): support tool_calls in stram response
                },
                finish_reason: finish_reason,
            }],
            usage: usage,
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatTemplate {
    name: String,
    template: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatTemplateVersions {
    Single(String),
    Multiple(Vec<ChatTemplate>),
}

use std::path::Path;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HubTokenizerConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub completion_template: Option<String>,
    pub bos_token: Option<TokenizerConfigToken>,
    pub eos_token: Option<TokenizerConfigToken>,
    pub tokenizer_class: Option<String>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
}

impl HubTokenizerConfig {
    pub fn from_file<P: AsRef<Path>>(filename: P) -> Option<Self> {
        std::fs::read_to_string(filename)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TokenizerConfigToken {
    String(String),
    Object { content: String },
}

impl TokenizerConfigToken {
    pub fn as_str(&self) -> &str {
        match self {
            TokenizerConfigToken::String(s) => s,
            TokenizerConfigToken::Object { content } => content,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "processor_class")]
pub enum HubPreprocessorConfig {
    Idefics2Processor(Idefics2Preprocessor),
}

impl HubPreprocessorConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(filename: P) -> Option<Self> {
        let content = std::fs::read_to_string(filename).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Idefics2Preprocessor {
    #[serde(default)]
    do_image_splitting: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HubProcessorConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub image_seq_len: usize,
    pub processor_class: Option<String>,
}

impl HubProcessorConfig {
    pub fn from_file<P: AsRef<Path>>(filename: P) -> Option<Self> {
        std::fs::read_to_string(filename)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
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
