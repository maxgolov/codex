//! Anthropic Messages API wire types.
//!
//! These types model the request and response shapes used by the Anthropic
//! Messages API (<https://docs.anthropic.com/en/api/messages>). They are used
//! exclusively by the Anthropic streaming provider in [`super::anthropic`] and
//! are intentionally kept private to this crate.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

// ── Request types ───────────────────────────────────────────────────────

/// Top-level request body for `POST /v1/messages`.
#[derive(Debug, Serialize)]
pub(crate) struct MessagesRequest<'a> {
    pub model: &'a str,
    pub max_tokens: u32,
    pub messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    /// Must be `true` for streaming.
    pub stream: bool,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MessageParam {
    pub role: String,
    pub content: MessageContent,
}

/// Content can be a plain string or an array of typed content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlockParam>),
}

/// Input content block variants (for request messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ContentBlockParam {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ImageSource {
    pub r#type: String,
    pub media_type: String,
    pub data: String,
}

// ── Tool types ──────────────────────────────────────────────────────────

/// Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Anthropic tool_choice parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

// ── Response types ──────────────────────────────────────────────────────

/// Response content block variants.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

// ── SSE streaming event types ───────────────────────────────────────────

/// Top-level SSE event envelope. The `event:` line in the SSE stream
/// determines which variant to deserialize into.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StreamMessageStart },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        #[allow(dead_code)]
        usage: MessageDeltaUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: StreamError },
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamMessageStart {
    pub id: String,
    #[allow(dead_code)]
    pub role: String,
    #[allow(dead_code)]
    pub model: String,
    #[allow(dead_code)]
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessageDelta {
    pub stop_reason: Option<StopReason>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MessageDeltaUsage {
    #[allow(dead_code)]
    pub output_tokens: u32,
}

/// Content block delta variants.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
pub(crate) enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta {
        #[allow(dead_code)]
        signature: String,
    },
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamError {
    pub message: String,
    #[allow(dead_code)]
    pub r#type: String,
}
