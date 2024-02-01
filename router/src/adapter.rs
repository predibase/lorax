use std::hash;

use crate::AdapterParameters;

/// "adapter ID" for the base model. The base model does not have an adapter ID,
/// but we reason about it in the same way. This must match the base model ID
/// used in the Python server.
pub const BASE_MODEL_ADAPTER_ID: &str = "__base_model__";

/// default adapter source. One TODO is to figure out how to do this
/// from within the proto definition, or lib.rs
pub const DEFAULT_ADAPTER_SOURCE: &str = "hub";

#[derive(Debug, Clone)]
pub(crate) struct Adapter {
    /// adapter parameters
    params: AdapterParameters,
    /// source (enforced at proto level)
    source: String,
    /// index of the adapter
    index: u32,
    /// Optional - External api token
    api_token: Option<String>,
}

impl Adapter {
    pub(crate) fn new(
        params: AdapterParameters,
        source: String,
        index: u32,
        api_token: Option<String>,
    ) -> Self {
        Self {
            params,
            source,
            index,
            api_token,
        }
    }

    pub(crate) fn params(&self) -> &AdapterParameters {
        &self.params
    }

    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn api_token(&self) -> &std::option::Option<std::string::String> {
        &self.api_token
    }

    pub(crate) fn index(&self) -> u32 {
        self.index
    }

    pub(crate) fn as_string(&self) -> String {
        // format "<source>:<id>"
        format!("{}:{}", self.source, self.params.adapter_ids.join(","))
    }
}

impl hash::Hash for Adapter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl Eq for Adapter {}

impl PartialEq for Adapter {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
