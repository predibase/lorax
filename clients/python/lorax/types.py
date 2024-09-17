from enum import Enum
from pydantic import BaseModel, field_validator, model_validator, Field, ConfigDict
from typing import Optional, List, Dict, Any, OrderedDict, Union

from lorax.errors import ValidationError


ADAPTER_SOURCES = ["hub", "local", "s3", "pbase"]
MERGE_STRATEGIES = ["linear", "ties", "dare_linear", "dare_ties"]
MAJORITY_SIGN_METHODS = ["total", "frequency"]


class MergedAdapters(BaseModel):
    # IDs of the adapters to merge
    ids: List[str]
    # Weights of the adapters to merge
    weights: List[float]
    # Merge strategy
    merge_strategy: Optional[str] = None
    # Density
    density: float
    # Majority sign method
    majority_sign_method: Optional[str] = None

    @field_validator("ids")
    def validate_ids(cls, v):
        if not v:
            raise ValidationError("`ids` cannot be empty")
        return v

    @field_validator("weights")
    def validate_weights(cls, v, values):
        ids = values.data["ids"]
        if not v:
            raise ValidationError("`weights` cannot be empty")
        if len(ids) != len(v):
            raise ValidationError("`ids` and `weights` must have the same length")
        return v

    @field_validator("merge_strategy")
    def validate_merge_strategy(cls, v):
        if v is not None and v not in MERGE_STRATEGIES:
            raise ValidationError(f"`merge_strategy` must be one of {MERGE_STRATEGIES}")
        return v

    @field_validator("density")
    def validate_density(cls, v):
        if v < 0 or v > 1.0:
            raise ValidationError("`density` must be >= 0.0 and <= 1.0")
        return v

    @field_validator("majority_sign_method")
    def validate_majority_sign_method(cls, v):
        if v is not None and v not in MAJORITY_SIGN_METHODS:
            raise ValidationError(f"`majority_sign_method` must be one of {MAJORITY_SIGN_METHODS}")
        return v


class ResponseFormatType(str, Enum):
    json_object = "json_object"


class ResponseFormat(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    type: ResponseFormatType
    schema_spec: Union[Dict[str, Any], OrderedDict] = Field(alias="schema")


class Parameters(BaseModel):
    # The ID of the adapter to use
    adapter_id: Optional[str] = None
    # The source of the adapter to use
    adapter_source: Optional[str] = None
    # Adapter merge parameters
    merged_adapters: Optional[MergedAdapters] = None
    # API token for accessing private adapters
    api_token: Optional[str] = None
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: Optional[int] = None
    # Whether to ignore the EOS token during generation
    ignore_eos_token: bool = False
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int] = None
    # The value used to module the logits distribution.
    temperature: Optional[float] = None
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int] = None
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float] = None
    # truncate inputs tokens to the given size
    truncate: Optional[int] = None
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float] = None
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int] = None
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False
    # The number of highest probability vocabulary tokens to return as alternative tokens in the generation result
    return_k_alternatives: Optional[int] = None
    # Optional response format specification to constrain the generated text
    response_format: Optional[ResponseFormat] = None

    @model_validator(mode="after")
    def valid_adapter_id(self):
        adapter_id = self.adapter_id
        merged_adapters = self.merged_adapters
        if adapter_id is not None and merged_adapters is not None:
            raise ValidationError("you must specify at most one of `adapter_id` or `merged_adapters`")
        return self

    @field_validator("adapter_source")
    def valid_adapter_source(cls, v):
        if v is not None and v not in ADAPTER_SOURCES:
            raise ValidationError(f"`adapter_source={v}` must be one of {ADAPTER_SOURCES}")
        return v

    @field_validator("best_of")
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValidationError("`best_of` must be strictly positive")
            if field_value > 1 and values.data["seed"] is not None:
                raise ValidationError("`seed` must not be set when `best_of` is > 1")
            sampling = (
                values.data["do_sample"]
                | (values.data["temperature"] is not None)
                | (values.data["top_k"] is not None)
                | (values.data["top_p"] is not None)
                | (values.data["typical_p"] is not None)
            )
            if field_value > 1 and not sampling:
                raise ValidationError("you must use sampling when `best_of` is > 1")

        return field_value

    @field_validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`repetition_penalty` must be strictly positive")
        return v

    @field_validator("seed")
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`seed` must be positive")
        return v

    @field_validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v < 0:
            raise ValidationError("`temperature` must be non-negative")
        return v

    @field_validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`top_k` must be strictly positive")
        return v

    @field_validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`top_p` must be > 0.0 and < 1.0")
        return v

    @field_validator("truncate")
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`truncate` must be strictly positive")
        return v

    @field_validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValidationError("`typical_p` must be > 0.0 and < 1.0")
        return v

    @field_validator("return_k_alternatives")
    def valid_return_k_alternatives(cls, v):
        if v is not None and v <= 0:
            raise ValidationError("`return_k_alternatives` must be strictly positive")
        return v


class Request(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False

    @field_validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v

    @field_validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values.data["parameters"]
        if parameters is not None and parameters.best_of is not None and parameters.best_of > 1 and field_value:
            raise ValidationError("`best_of` != 1 is not supported when `stream` == True")
        return field_value


class BatchRequest(BaseModel):
    # Prompt
    inputs: List[str]
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False

    @field_validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v

    @field_validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values.data["parameters"]
        if parameters is not None and parameters.best_of is not None and parameters.best_of > 1 and field_value:
            raise ValidationError("`best_of` != 1 is not supported when `stream` == True")
        return field_value


# Decoder input tokens
class InputToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float] = None


# Alternative Tokens
class AlternativeToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float


# Generated tokens
class Token(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool
    # Alternative tokens
    alternative_tokens: Optional[List[AlternativeToken]] = None


# Generation finish reason
class FinishReason(str, Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
class BestOfSequence(BaseModel):
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]


# `generate` details
class Details(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of prompt tokens
    prompt_tokens: int
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken]
    # Generated tokens
    tokens: List[Token]
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]] = None


# `generate` return value
class Response(BaseModel):
    # Generated text
    generated_text: str
    # Generation details
    details: Optional[Details] = None


# `generate_stream` details
class StreamDetails(BaseModel):
    # Generation finish reason
    finish_reason: FinishReason
    # Number of prompt tokens
    prompt_tokens: int
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None


# `generate_stream` return value
class StreamResponse(BaseModel):
    # Generated token
    token: Token
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str] = None
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails] = None


# Inference API currently deployed model
class DeployedModel(BaseModel):
    model_id: str
    sha: str
    # Suppress pydantic warning over `model_id` field
    model_config = ConfigDict(protected_namespaces=())


class EmbedResponse(BaseModel):
    # Embeddings
    embeddings: Optional[List[float]]

class ClassifyResponse(BaseModel):
    # Classifications
    entities: Optional[List[dict]]