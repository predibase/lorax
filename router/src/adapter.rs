/// Adapter utils

/// "adapter ID" for the base model. The base model does not have an adapter ID,
/// but we reason about it in the same way. This must match the base model ID
/// used in the Python server.
pub const BASE_MODEL_ADAPTER_ID: &str = "__base_model__";

/// default adapter source. One TODO is to figure out how to do this
/// from within the proto definition, or lib.rs
pub const DEFAULT_ADAPTER_SOURCE: &str = "hub";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Adapter {
    /// name of adapter
    id: String,
    /// source (enforced at proto level)
    source: String,
    /// index of the adapter
    index: u32,
    /// Optional - Api token from predibase
    predibase_api_token: Option<String>,
}

impl Adapter {
    pub(crate) fn new(id: String, source: String, index: u32, predibase_api_token: Option<String>) -> Self {   
        Self { id, source, index, predibase_api_token}
    }

    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn predibase_api_token(&self) -> &std::option::Option<std::string::String> {
        &self.predibase_api_token
    }

    pub(crate) fn index(&self) -> u32 {
        self.index
    }

    pub(crate) fn as_string(&self) -> String {
        // format "<source>:<id>"
        format!("{}:{}", self.source, self.id)
    }
}
