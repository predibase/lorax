use crate::adapter::Adapter;
/// Payload validation logic
use crate::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{GenerateParameters, GenerateRequest};
use lorax_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use rand::{thread_rng, Rng};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::TruncationDirection;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_best_of: usize,
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    /// Channel to communicate with the background tokenization task
    sender: Option<flume::Sender<TokenizerRequest>>,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<Tokenizer>,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_input_length: usize,
        max_total_tokens: usize,
    ) -> Self {
        // If we have a fast tokenizer
        let sender = if let Some(tokenizer) = tokenizer {
            // Create channel
            let (validation_sender, validation_receiver) = flume::unbounded();

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let receiver_clone = validation_receiver.clone();

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(tokenizer_clone, receiver_clone)
                });
            }
            Some(validation_sender)
        } else {
            None
        };

        Self {
            max_best_of,
            sender,
            max_stop_sequences,
            max_input_length,
            max_total_tokens,
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn tokenize(
        &self,
        inputs: String,
        truncate: Option<usize>,
    ) -> Result<Option<(tokenizers::Encoding, String)>, ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, truncate), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let encoding = response_receiver.await.unwrap()?;
            Ok(Some(encoding))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self, inputs))]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(String, usize), ValidationError> {
        // If we have a fast tokenizer
        if let Some((encoding, inputs)) = self.tokenize(inputs.clone(), truncate).await? {
            // Create response channel
            let input_length = encoding.len();

            if let Some(max_new_tokens) = max_new_tokens {
                // Get total tokens
                let total_tokens = input_length + max_new_tokens as usize;

                // Validate MaxTotalTokens
                if total_tokens > self.max_total_tokens {
                    return Err(ValidationError::MaxTotalTokens(
                        self.max_total_tokens,
                        input_length,
                        max_new_tokens,
                    ));
                }
            }

            // Validate InputLength
            if input_length > self.max_input_length {
                return Err(ValidationError::InputLength(
                    self.max_input_length,
                    input_length,
                ));
            }

            metrics::histogram!("lorax_request_input_length", input_length as f64);
            Ok((inputs, input_length))
        }
        // Return inputs without validation
        else {
            // In this case, we don't know the real length in tokens of the inputs
            // However, the inputs will be truncated by the python servers
            // We make sure that truncate + max_new_tokens <= self.max_total_tokens
            let input_length = truncate.unwrap_or(self.max_input_length);

            if let Some(max_new_tokens) = max_new_tokens {
                if (input_length as u32 + max_new_tokens) > self.max_total_tokens as u32 {
                    return Err(ValidationError::MaxNewTokens(
                        self.max_total_tokens - self.max_input_length,
                        max_new_tokens,
                    ));
                }
            }

            Ok((inputs, input_length))
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
        adapter: Adapter,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            ignore_eos_token,
            stop: stop_sequences,
            truncate,
            seed,
            watermark,
            adapter_id,
            adapter_parameters,
            decoder_input_details,
            return_k_alternatives,
            apply_chat_template,
            response_format,
            ..
        } = request.parameters;

        // adapter validation
        // cannot specify both adapter_id and adapter_parameters
        if adapter_parameters.is_some() && adapter_id.is_some() {
            return Err(ValidationError::AdapterIdConflict);
        }

        if adapter_parameters.is_some() {
            let nadapters = adapter_parameters.as_ref().unwrap().adapter_ids.len();
            let nweights = adapter_parameters.as_ref().unwrap().weights.len();
            if nadapters < 1 {
                return Err(ValidationError::AdapterIdMissing);
            }

            if nadapters != nweights {
                return Err(ValidationError::AdapterWeightMismatch);
            }
        }

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            return Err(BestOfSampling);
        }

        let temperature = temperature.unwrap_or(1.0);
        if temperature < 0.0 {
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }

        // Different because the proto default value is not a valid value
        // for the user
        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k: u32 = top_k
            .map(|value| {
                if value <= 0 {
                    return Err(ValidationError::TopK);
                }
                Ok(value as u32)
            })
            .unwrap_or(Ok(0))?;

        let return_k_alternatives: u32 = return_k_alternatives
            .map(|value| {
                if value <= 0 {
                    return Err(ValidationError::ReturnKAlternatives);
                }
                Ok(value as u32)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens.is_some() && max_new_tokens.unwrap() == 0 {
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        // If seed is None, assign a random one
        let seed = match seed {
            None => thread_rng().gen(),
            Some(seed) => {
                if best_of > 1 {
                    return Err(BestOfSeed);
                }
                seed
            }
        };

        // Check if inputs is empty
        if request.inputs.is_empty() {
            return Err(EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        let adapter_id = adapter_id.unwrap_or_else(|| "".to_string());

        // Validate inputs
        let (inputs, input_length) = self
            .validate_input(request.inputs, truncate, max_new_tokens)
            .await?;

        let mut schema: Option<String> = None;
        if response_format.is_some() {
            let response_format_val = response_format.unwrap();
            schema = Some(response_format_val.schema.to_string())
        }

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            seed,
            watermark,
            adapter_id,
            schema,
            return_k_alternatives,
        };

        let effective_max_new_tokens =
            max_new_tokens.unwrap_or((self.max_total_tokens - input_length) as u32);
        let stopping_parameters = StoppingCriteriaParameters {
            max_new_tokens: effective_max_new_tokens,
            stop_sequences,
            ignore_eos_token,
        };

        metrics::histogram!(
            "lorax_request_max_new_tokens",
            effective_max_new_tokens as f64
        );

        Ok(ValidGenerateRequest {
            inputs,
            decoder_input_details,
            input_length: input_length as u32,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            adapter,
            apply_chat_template,
        })
    }

    /// Validate the best_of parameter
    #[instrument(skip_all)]
    pub(crate) fn validate_best_of(&self, best_of: usize) -> Result<usize, ValidationError> {
        if self.max_best_of == 1 && best_of != 1 {
            return Err(ValidationError::BestOfDisabled);
        }

        if best_of > self.max_best_of {
            return Err(ValidationError::BestOf(self.max_best_of, best_of));
        }

        Ok(best_of)
    }
}

/// Start tokenization workers
fn tokenizer_worker(tokenizer: Tokenizer, receiver: flume::Receiver<TokenizerRequest>) {
    // Loop over requests
    while let Ok(((inputs, truncate), response_tx, parent_span)) = receiver.recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(inputs, truncate, &tokenizer))
                .unwrap_or(())
        })
    }
}

/// Get input length and optionally truncate it
fn prepare_input(
    mut inputs: String,
    truncate: Option<usize>,
    tokenizer: &Tokenizer,
) -> Result<(tokenizers::Encoding, String), ValidationError> {
    // Get the number of tokens in the input
    let mut encoding = tokenizer
        .encode(inputs.clone(), true)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    // Optionally truncate
    if let Some(truncate) = truncate {
        if truncate < encoding.len() {
            encoding.truncate(truncate, 0, TruncationDirection::Left);
            inputs = tokenizer
                .decode(encoding.get_ids(), false)
                .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;
        }
    }

    Ok((encoding, inputs))
}

type TokenizerRequest = (
    (String, Option<usize>),
    oneshot::Sender<Result<(tokenizers::Encoding, String), ValidationError>>,
    Span,
);

#[derive(Debug)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub adapter: Adapter,
    pub apply_chat_template: bool,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("`best_of` must be > 0 and <= {0}. Given: {1}")]
    BestOf(usize, usize),
    #[error("`best_of` != 1 is not allowed for this endpoint")]
    BestOfDisabled,
    #[error("you must use sampling when `best_of` is > 1")]
    BestOfSampling,
    #[error("`seed` must not be set when `best_of` > 1")]
    BestOfSeed,
    #[error("`best_of` != 1 is not supported when streaming tokens")]
    BestOfStream,
    #[error("`decoder_input_details` == true is not supported when streaming tokens")]
    PrefillDetailsStream,
    #[error("`temperature` must be non-negative")]
    Temperature,
    #[error("`repetition_penalty` must be strictly positive")]
    RepetitionPenalty,
    #[error("`top_p` must be > 0.0 and < 1.0")]
    TopP,
    #[error("`top_k` must be strictly positive")]
    TopK,
    #[error("`return_k_alternatives` must be strictly positive")]
    ReturnKAlternatives,
    #[error("`truncate` must be strictly positive and less than {0}. Given: {1}")]
    Truncate(usize, usize),
    #[error("`typical_p` must be > 0.0 and < 1.0")]
    TypicalP,
    #[error("`max_new_tokens` must be strictly positive")]
    NegativeMaxNewTokens,
    #[error("`max_new_tokens` must be <= {0}. Given: {1}")]
    MaxNewTokens(usize, u32),
    #[error("`inputs` tokens + `max_new_tokens` must be <= {0}. Given: {1} `inputs` tokens and {2} `max_new_tokens`")]
    MaxTotalTokens(usize, usize, u32),
    #[error("`inputs` must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("`inputs` cannot be empty")]
    EmptyInput,
    #[error("`stop` supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("at most one of `adapter_id` or `adapters` may be provided")]
    AdapterIdConflict,
    #[error("at least one adapter ID must be provided when setting `adapters`")]
    AdapterIdMissing,
    #[error("number of adapter IDs must match number of adapter weights")]
    AdapterWeightMismatch,
    #[error("Embedding models don't support text generation")]
    EmbeddingModel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::get_tokenizer;
    use crate::{default_parameters, AdapterParameters};

    #[tokio::test]
    async fn test_validation_max_new_tokens() {
        let tokenizer = None;
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_input_length = 4;
        let max_total_tokens = 5;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_input_length,
            max_total_tokens,
        );

        let max_new_tokens = Some(10);
        match validation
            .validate_input("Hello".to_string(), None, max_new_tokens)
            .await
        {
            Err(ValidationError::MaxNewTokens(1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_input_length() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_input_length = 4;
        let max_total_tokens = 5;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_input_length,
            max_total_tokens,
        );

        let max_new_tokens = Some(10);
        match validation
            .validate_input("Hello".to_string(), None, max_new_tokens)
            .await
        {
            Err(ValidationError::MaxTotalTokens(5, 1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_best_of_sampling() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_input_length = 4;
        let max_total_tokens = 5;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_input_length,
            max_total_tokens,
        );
        match validation
            .validate(
                GenerateRequest {
                    inputs: "Hello".to_string(),
                    parameters: GenerateParameters {
                        best_of: Some(2),
                        do_sample: false,
                        ..default_parameters()
                    },
                },
                Adapter::new(
                    AdapterParameters {
                        adapter_ids: vec!["".to_string()],
                        ..Default::default()
                    },
                    "hf".to_string(),
                    0,
                    None,
                ),
            )
            .await
        {
            Err(ValidationError::BestOfSampling) => (),
            _ => panic!("Unexpected not best of sampling"),
        }
    }

    #[tokio::test]
    async fn test_validation_top_p() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_input_length = 4;
        let max_total_tokens = 5;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_input_length,
            max_total_tokens,
        );
        match validation
            .validate(
                GenerateRequest {
                    inputs: "Hello".to_string(),
                    parameters: GenerateParameters {
                        top_p: Some(1.0),
                        ..default_parameters()
                    },
                },
                Adapter::new(
                    AdapterParameters {
                        adapter_ids: vec!["".to_string()],
                        ..Default::default()
                    },
                    "hf".to_string(),
                    0,
                    None,
                ),
            )
            .await
        {
            Err(ValidationError::TopP) => (),
            _ => panic!("Unexpected top_p"),
        }

        match validation
            .validate(
                GenerateRequest {
                    inputs: "Hello".to_string(),
                    parameters: GenerateParameters {
                        top_p: Some(0.99),
                        max_new_tokens: Some(1),
                        ..default_parameters()
                    },
                },
                Adapter::new(
                    AdapterParameters {
                        adapter_ids: vec!["".to_string()],
                        ..Default::default()
                    },
                    "hf".to_string(),
                    0,
                    None,
                ),
            )
            .await
        {
            Ok(_) => (),
            _ => panic!("Unexpected top_p error"),
        }

        let valid_request = validation
            .validate(
                GenerateRequest {
                    inputs: "Hello".to_string(),
                    parameters: GenerateParameters {
                        top_p: None,
                        max_new_tokens: Some(1),
                        ..default_parameters()
                    },
                },
                Adapter::new(
                    AdapterParameters {
                        adapter_ids: vec!["".to_string()],
                        ..Default::default()
                    },
                    "hf".to_string(),
                    0,
                    None,
                ),
            )
            .await
            .unwrap();
        // top_p == 1.0 is invalid for users to ask for but it's the default resolved value.
        assert_eq!(valid_request.parameters.top_p, 1.0);
    }
}
