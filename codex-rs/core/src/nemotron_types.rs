//! Nemotron/vLLM Chat Completions API wire types.
//!
//! These types model the request and response shapes used by vLLM's
//! OpenAI-compatible Chat Completions endpoint (`/v1/chat/completions`).
//! They include Nemotron-specific extensions such as `chat_template_kwargs`
//! for thinking budget control.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

// ── Request types ───────────────────────────────────────────────────────

/// Top-level request body for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'a str>,
    /// Nemotron-specific: passes `thinking_budget` to the vLLM chat template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Request SSE streaming.
    pub stream: bool,
    /// Include token usage in the final SSE chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

/// Nemotron `chat_template_kwargs` for thinking budget control.
#[derive(Debug, Serialize)]
pub(crate) struct ChatTemplateKwargs {
    pub thinking_budget: u32,
}

/// Request streaming options.
#[derive(Debug, Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

/// A single message in the chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call emitted by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolCallMessage {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

/// The function name and arguments within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition sent in the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatTool {
    pub r#type: String,
    pub function: ChatToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ── Response / SSE types ────────────────────────────────────────────────

/// A single SSE chunk from `POST /v1/chat/completions` with `stream: true`.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<ChunkUsage>,
}

/// A choice within a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct ChunkChoice {
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// The delta within a streaming choice.
#[derive(Debug, Deserialize)]
pub(crate) struct ChunkDelta {
    #[serde(default)]
    #[allow(dead_code)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// A tool call delta in a streaming chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct ChunkToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkFunction>,
}

/// Function delta within a tool call chunk.
#[derive(Debug, Deserialize)]
pub(crate) struct ChunkFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Token usage reported in the final SSE chunk (when `include_usage: true`).
#[derive(Debug, Deserialize)]
pub(crate) struct ChunkUsage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    #[allow(dead_code)]
    pub total_tokens: i64,
}

// ── Model listing types (`GET /v1/models`) ──────────────────────────────

/// Response from `GET /v1/models`.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelListResponse {
    pub data: Vec<ModelEntry>,
}

/// A single model entry returned by the vLLM models endpoint.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelEntry {
    pub id: String,
}
