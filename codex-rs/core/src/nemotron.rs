//! Nemotron/vLLM Chat Completions API streaming provider.
//!
//! This module implements the translation layer between Codex's internal
//! [`Prompt`] / [`ResponseEvent`] types and the vLLM OpenAI-compatible
//! Chat Completions API wire format. It handles Nemotron-specific quirks:
//!
//! - **`<think>` tag extraction**: Nemotron wraps reasoning in `<think>...</think>`
//!   tags inline in text output. This module parses them out and emits
//!   [`ResponseEvent::ReasoningContentDelta`] events instead.
//! - **`thinking_budget` injection**: Passes `chat_template_kwargs.thinking_budget`
//!   to control the reasoning token budget.
//! - **Schema flattening**: Tool parameter schemas are flattened to remove
//!   `oneOf`/`anyOf` that vLLM's Jinja2 template cannot handle.
//! - **`stream_options.include_usage`**: Requests token counts in the final SSE chunk.

use std::collections::HashMap;
use std::time::Duration;

use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::StatusCode;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::nemotron_types::ChatCompletionChunk;
use crate::nemotron_types::ChatCompletionRequest;
use crate::nemotron_types::ChatMessage;
use crate::nemotron_types::ChatTemplateKwargs;
use crate::nemotron_types::FunctionCall;
use crate::nemotron_types::ModelListResponse;
use crate::nemotron_types::StreamOptions;
use crate::nemotron_types::ToolCallMessage;
use crate::tools::nemotron_tools::create_tools_for_nemotron;
use crate::util::backoff;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;

/// Default max_tokens for Nemotron models.
const DEFAULT_MAX_TOKENS: u32 = 32_768;

/// Minimum thinking budget — Nemotron requires at least 1024 even when
/// thinking is nominally disabled, to avoid format errors.
const MIN_THINKING_BUDGET: u32 = 1024;

/// Default thinking budget when reasoning is enabled.
const DEFAULT_THINKING_BUDGET: u32 = 4096;

/// Default per-request timeout in seconds. If the HTTP POST does not receive
/// first byte within this window the attempt is abandoned and retried (the
/// "nudge" behaviour). Override via `NEMOTRON_REQUEST_TIMEOUT`.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;

/// After a request timeout, the thinking budget is reduced by this factor
/// on the retry to give vLLM a better chance of responding in time.
const THINKING_BUDGET_REDUCTION_FACTOR: u32 = 2;

/// Resolve the model slug for a Nemotron/vLLM endpoint.
///
/// - If `slug` is `"auto"` or `"*"`, queries `/v1/models` and returns the
///   first available model.
/// - Otherwise queries `/v1/models` and checks for an exact match. If none
///   is found, picks the best substring match (the model whose `id` contains
///   `slug` as a case-insensitive substring, preferring the shortest match).
/// - Returns an error only when no model can be matched.
async fn resolve_nemotron_model(slug: &str, provider: &ModelProviderInfo) -> Result<String> {
    let is_auto = slug.eq_ignore_ascii_case("auto") || slug == "*";

    // If the slug looks like a fully-qualified model ID (not auto), try it
    // as-is first without hitting /v1/models. We'll only query the endpoint
    // when auto-detection is needed.
    if !is_auto && slug.contains('/') && slug.chars().filter(|c| *c == '/').count() >= 1 {
        // Heuristic: a slug with slashes and a long final segment is likely
        // a full model ID. Skip the models list call — if it's wrong the
        // chat completions call will return a clear error anyway.
        let last_segment = slug.rsplit('/').next().unwrap_or(slug);
        if last_segment.len() > 20 {
            return Ok(slug.to_string());
        }
    }

    let api_provider = provider.to_api_provider(None)?;
    let url = api_provider.url_for_path("/models");
    let mut header_map = reqwest::header::HeaderMap::new();
    for (name, value) in api_provider.headers.iter() {
        if let Ok(rname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes())
            && let Ok(rvalue) = reqwest::header::HeaderValue::from_bytes(value.as_bytes())
        {
            header_map.insert(rname, rvalue);
        }
    }

    let resp = reqwest::Client::new()
        .get(&url)
        .headers(header_map)
        .send()
        .await
        .map_err(|e| CodexErr::UnsupportedOperation(format!("Failed to query {url}: {e}")))?;

    if !resp.status().is_success() {
        if is_auto {
            return Err(CodexErr::UnsupportedOperation(format!(
                "Cannot auto-detect model: /v1/models returned {}",
                resp.status()
            )));
        }
        // Fall through — use the slug as-is and let the chat call fail with
        // a more specific error.
        debug!(
            "/v1/models returned {}; using slug as-is: {slug}",
            resp.status()
        );
        return Ok(slug.to_string());
    }

    let body: ModelListResponse = resp.json().await.map_err(|e| {
        CodexErr::UnsupportedOperation(format!("Failed to parse /v1/models response: {e}"))
    })?;

    let models: Vec<&str> = body.data.iter().map(|m| m.id.as_str()).collect();

    if models.is_empty() {
        return Err(CodexErr::UnsupportedOperation(
            "No models available on the vLLM endpoint".to_string(),
        ));
    }

    if is_auto {
        let picked = models[0].to_string();
        debug!("Auto-detected Nemotron model: {picked}");
        return Ok(picked);
    }

    if let Some(matched) = pick_best_model(slug, &models) {
        if matched != slug {
            debug!("Resolved '{slug}' → '{matched}' via substring match");
        }
        return Ok(matched.to_string());
    }

    Err(CodexErr::UnsupportedOperation(format!(
        "Model '{slug}' not found. Available models: {}",
        models.join(", ")
    )))
}

/// Given a requested slug and a list of available model IDs, return the best
/// match. Returns `None` when no match is found.
///
/// - Exact match takes priority.
/// - Otherwise, the shortest model ID containing `slug` as a case-insensitive
///   substring is selected.
fn pick_best_model<'a>(slug: &str, models: &[&'a str]) -> Option<&'a str> {
    // Exact match?
    if let Some(&m) = models.iter().find(|&&m| m == slug) {
        return Some(m);
    }

    // Substring match (case-insensitive), prefer shortest model ID.
    let slug_lower = slug.to_ascii_lowercase();
    let mut best: Option<&str> = None;
    for &m in models {
        if m.to_ascii_lowercase().contains(&slug_lower) {
            match best {
                None => best = Some(m),
                Some(prev) if m.len() < prev.len() => best = Some(m),
                _ => {}
            }
        }
    }
    best
}

/// Stream a completion from a Nemotron/vLLM Chat Completions endpoint.
pub async fn stream_nemotron_chat(
    prompt: &Prompt,
    model_info: &ModelInfo,
    provider: &ModelProviderInfo,
) -> Result<ResponseStream> {
    if prompt.output_schema.is_some() {
        return Err(CodexErr::UnsupportedOperation(
            "output_schema is not supported for Nemotron Chat Completions API".to_string(),
        ));
    }

    // ── Resolve model slug (supports "auto", "*", and partial names) ────
    let resolved_model = resolve_nemotron_model(&model_info.slug, provider).await?;
    debug!("Using Nemotron model: {resolved_model}");

    // ── Build messages ──────────────────────────────────────────────────
    let system_text = &prompt.base_instructions.text;
    let input = prompt.get_formatted_input();
    let mut messages: Vec<ChatMessage> = Vec::new();

    // System message.
    messages.push(ChatMessage {
        role: "system".to_string(),
        content: Some(system_text.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

    for item in &input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let chat_role = match role.as_str() {
                    "assistant" => "assistant",
                    "system" => continue, // already added as first message
                    _ => "user",
                };

                let text: String = content
                    .iter()
                    .map(|c| match c {
                        ContentItem::InputText { text: t }
                        | ContentItem::OutputText { text: t } => t.as_str(),
                        ContentItem::InputImage { image_url } => image_url.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("");

                if text.trim().is_empty() {
                    continue;
                }

                messages.push(ChatMessage {
                    role: chat_role.to_string(),
                    content: Some(text),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ToolCallMessage {
                        id: call_id.clone(),
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    }]),
                    tool_call_id: None,
                });
            }

            ResponseItem::FunctionCallOutput { call_id, output } => {
                let content_text = if let Some(items) = output.content_items() {
                    items
                        .iter()
                        .map(|it| match it {
                            FunctionCallOutputContentItem::InputText { text } => text.clone(),
                            FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                                format!("[Image: {image_url}]")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    output.text_content().unwrap_or_default().to_string()
                };
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(content_text),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                });
            }

            ResponseItem::LocalShellCall { id, action, .. } => {
                let args_json = serde_json::to_string(action).unwrap_or_else(|_| "{}".to_string());
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ToolCallMessage {
                        id: id.clone().unwrap_or_default(),
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: "shell".to_string(),
                            arguments: args_json,
                        },
                    }]),
                    tool_call_id: None,
                });
            }

            ResponseItem::CustomToolCall {
                id,
                name,
                input: tool_input,
                ..
            } => {
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![ToolCallMessage {
                        id: id.clone().unwrap_or_default(),
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: tool_input.clone(),
                        },
                    }]),
                    tool_call_id: None,
                });
            }

            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let content_text = output.text_content().unwrap_or_default().to_string();
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(content_text),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                });
            }

            // Omit from conversation history.
            ResponseItem::GhostSnapshot { .. }
            | ResponseItem::Reasoning { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::Other => continue,
        }
    }

    // ── Build tools ─────────────────────────────────────────────────────
    let tools = create_tools_for_nemotron(&prompt.tools)?;
    let tools_param = if tools.is_empty() { None } else { Some(tools) };
    let tool_choice = if tools_param.is_some() {
        Some("auto")
    } else {
        None
    };

    // ── Thinking budget ─────────────────────────────────────────────────
    let initial_thinking_budget = std::env::var("NEMOTRON_THINKING_BUDGET")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_THINKING_BUDGET)
        .max(MIN_THINKING_BUDGET);

    // ── Request timeout ─────────────────────────────────────────────────
    let request_timeout = Duration::from_secs(
        std::env::var("NEMOTRON_REQUEST_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS),
    );

    // ── Resolve URL + headers from provider ─────────────────────────────
    let api_provider = provider.to_api_provider(/*auth_mode*/ None)?;
    let url = api_provider.url_for_path("/chat/completions");
    let mut header_map = reqwest::header::HeaderMap::new();
    for (name, value) in api_provider.headers.iter() {
        if let Ok(rname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes())
            && let Ok(rvalue) = reqwest::header::HeaderValue::from_bytes(value.as_bytes())
        {
            header_map.insert(rname, rvalue);
        }
    }

    debug!("Nemotron POST to {url}");

    // ── Execute with retries + nudge on timeout ─────────────────────────
    let http_client = reqwest::Client::builder()
        .timeout(request_timeout)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut attempt = 0;
    let max_retries = provider.request_max_retries();
    let mut thinking_budget = initial_thinking_budget;
    loop {
        attempt += 1;

        // Rebuild payload each iteration since thinking_budget may shrink.
        let request = ChatCompletionRequest {
            model: &resolved_model,
            messages: messages.clone(),
            temperature: None,
            max_tokens: Some(DEFAULT_MAX_TOKENS),
            tools: tools_param.clone(),
            tool_choice,
            chat_template_kwargs: Some(ChatTemplateKwargs { thinking_budget }),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
        };

        let payload = serde_json::to_value(&request).map_err(|e| {
            CodexErr::UnsupportedOperation(format!("Failed to serialize Nemotron request: {e}"))
        })?;

        debug!("Nemotron attempt {attempt}/{max_retries} (thinking_budget={thinking_budget})");

        let res = http_client
            .post(&url)
            .headers(header_map.clone())
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent>>(1600);
                // Report the resolved model so the TUI/caller sees the actual
                // model name instead of "auto" or a partial slug.
                let _ = tx_event
                    .send(Ok(ResponseEvent::ServerModel(resolved_model.clone())))
                    .await;
                let idle_timeout = provider.stream_idle_timeout();
                tokio::spawn(async move {
                    process_nemotron_response(resp, tx_event, idle_timeout).await;
                });
                return Ok(ResponseStream { rx_event });
            }
            Ok(resp) => {
                let status = resp.status();
                if !(status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()) {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(CodexErr::UnsupportedOperation(format!(
                        "Nemotron API returned {status}: {body}"
                    )));
                }

                if attempt > max_retries {
                    return Err(CodexErr::UnsupportedOperation(format!(
                        "Nemotron API retry limit reached (status {status})"
                    )));
                }

                let retry_after_secs = resp
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());

                let delay = retry_after_secs
                    .map(|s| Duration::from_millis(s * 1_000))
                    .unwrap_or_else(|| backoff(attempt));
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                if attempt > max_retries {
                    return Err(CodexErr::UnsupportedOperation(format!(
                        "Nemotron connection failed after retries: {e}"
                    )));
                }

                // Nudge: if the error looks like a timeout, reduce the
                // thinking budget on the next attempt so vLLM has less
                // work to do and is more likely to respond in time.
                if e.is_timeout() {
                    let reduced = (thinking_budget / THINKING_BUDGET_REDUCTION_FACTOR)
                        .max(MIN_THINKING_BUDGET);
                    tracing::warn!(
                        "Nemotron request timed out after {request_timeout:?}; \
                         nudging with reduced thinking_budget {thinking_budget} → {reduced} \
                         (attempt {attempt}/{max_retries})"
                    );
                    thinking_budget = reduced;
                }

                let delay = backoff(attempt);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

// ── Think-tag state machine ─────────────────────────────────────────────

/// Tracks the state of `<think>...</think>` tag extraction from streamed text.
///
/// Nemotron emits reasoning wrapped in `<think>...</think>` inline within the
/// text content. This parser splits the stream into reasoning deltas and text
/// deltas so they can be emitted as separate `ResponseEvent` types.
#[derive(Default)]
struct ThinkTagParser {
    /// Whether we are currently inside a `<think>` block.
    inside_think: bool,
    /// Partial buffer for incomplete tag sequences across chunk boundaries.
    buffer: String,
}

/// Output from the think-tag parser for a single text delta.
enum ParsedDelta {
    /// Plain text content (outside any `<think>` block).
    Text(String),
    /// Reasoning content (inside a `<think>` block).
    Thinking(String),
}

impl ThinkTagParser {
    /// Feed a text delta and return parsed segments.
    fn feed(&mut self, text: &str) -> Vec<ParsedDelta> {
        self.buffer.push_str(text);
        let mut results = Vec::new();
        self.drain(&mut results);
        results
    }

    /// Flush any remaining buffered content at end of stream.
    fn flush(&mut self) -> Vec<ParsedDelta> {
        let mut results = Vec::new();
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            if self.inside_think {
                results.push(ParsedDelta::Thinking(remaining));
            } else {
                results.push(ParsedDelta::Text(remaining));
            }
        }
        results
    }

    fn drain(&mut self, results: &mut Vec<ParsedDelta>) {
        loop {
            if self.inside_think {
                // Look for </think>
                if let Some(end_pos) = self.buffer.find("</think>") {
                    let thinking = self.buffer[..end_pos].to_string();
                    self.buffer = self.buffer[end_pos + "</think>".len()..].to_string();
                    self.inside_think = false;
                    if !thinking.is_empty() {
                        results.push(ParsedDelta::Thinking(thinking));
                    }
                } else if self.buffer.contains("</") {
                    // Might be a partial closing tag — keep buffering.
                    break;
                } else {
                    // All buffered content is thinking.
                    let thinking = std::mem::take(&mut self.buffer);
                    if !thinking.is_empty() {
                        results.push(ParsedDelta::Thinking(thinking));
                    }
                    break;
                }
            } else {
                // Look for <think>
                if let Some(start_pos) = self.buffer.find("<think>") {
                    let text = self.buffer[..start_pos].to_string();
                    self.buffer = self.buffer[start_pos + "<think>".len()..].to_string();
                    self.inside_think = true;
                    if !text.is_empty() {
                        results.push(ParsedDelta::Text(text));
                    }
                } else if let Some(lt_pos) = self.buffer.rfind('<') {
                    // Might be a partial opening tag — emit everything before
                    // the `<` and keep the rest buffered.
                    let text = self.buffer[..lt_pos].to_string();
                    self.buffer = self.buffer[lt_pos..].to_string();
                    if !text.is_empty() {
                        results.push(ParsedDelta::Text(text));
                    }
                    break;
                } else {
                    // All buffered content is text.
                    let text = std::mem::take(&mut self.buffer);
                    if !text.is_empty() {
                        results.push(ParsedDelta::Text(text));
                    }
                    break;
                }
            }
        }
    }
}

// ── SSE processor ───────────────────────────────────────────────────────

/// SSE processor for the vLLM Chat Completions streaming format.
async fn process_nemotron_response(
    response: reqwest::Response,
    tx_event: mpsc::Sender<Result<ResponseEvent>>,
    idle_timeout: Duration,
) {
    let byte_stream = response.bytes_stream();
    let mut stream = byte_stream.eventsource();

    /// Accumulated state for a tool call being streamed piece by piece.
    #[derive(Default)]
    struct ToolCallState {
        id: String,
        name: String,
        arguments: String,
    }

    let mut response_id = String::new();
    let mut assistant_item: Option<ResponseItem> = None;
    let mut tool_calls: HashMap<usize, ToolCallState> = HashMap::new();
    let mut think_parser = ThinkTagParser::default();
    let mut created_sent = false;

    // Token usage — populated from the final chunk.
    let mut input_tokens: i64 = 0;
    let mut output_tokens: i64 = 0;

    loop {
        let sse = match timeout(idle_timeout, stream.next()).await {
            Ok(Some(Ok(ev))) => ev,
            Ok(Some(Err(e))) => {
                let _ = tx_event
                    .send(Err(CodexErr::Stream(e.to_string(), None)))
                    .await;
                return;
            }
            Ok(None) => {
                // Stream ended — flush any remaining think-tag buffer.
                for delta in think_parser.flush() {
                    match delta {
                        ParsedDelta::Text(t) => {
                            if let Some(ResponseItem::Message { content, .. }) = &mut assistant_item
                            {
                                content.push(ContentItem::OutputText { text: t.clone() });
                            }
                            let _ = tx_event.send(Ok(ResponseEvent::OutputTextDelta(t))).await;
                        }
                        ParsedDelta::Thinking(t) => {
                            let _ = tx_event
                                .send(Ok(ResponseEvent::ReasoningContentDelta {
                                    delta: t,
                                    content_index: 0,
                                }))
                                .await;
                        }
                    }
                }

                // Finalize any pending assistant message.
                if let Some(item) = assistant_item.take() {
                    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }

                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id,
                        token_usage: Some(TokenUsage {
                            input_tokens,
                            cached_input_tokens: 0,
                            output_tokens,
                            reasoning_output_tokens: 0,
                            total_tokens: input_tokens + output_tokens,
                        }),
                    }))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(CodexErr::Stream(
                        "idle timeout waiting for Nemotron SSE".into(),
                        None,
                    )))
                    .await;
                return;
            }
        };

        // vLLM sends `data: [DONE]` as the final SSE event.
        if sse.data.trim() == "[DONE]" {
            // Flush think parser.
            for delta in think_parser.flush() {
                match delta {
                    ParsedDelta::Text(t) => {
                        if let Some(ResponseItem::Message { content, .. }) = &mut assistant_item {
                            content.push(ContentItem::OutputText { text: t.clone() });
                        }
                        let _ = tx_event.send(Ok(ResponseEvent::OutputTextDelta(t))).await;
                    }
                    ParsedDelta::Thinking(t) => {
                        let _ = tx_event
                            .send(Ok(ResponseEvent::ReasoningContentDelta {
                                delta: t,
                                content_index: 0,
                            }))
                            .await;
                    }
                }
            }

            // Emit pending tool calls.
            for (_idx, tc) in tool_calls.drain() {
                if let Some(item) = assistant_item.take() {
                    let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                }
                let item = ResponseItem::FunctionCall {
                    id: None,
                    name: tc.name,
                    namespace: None,
                    arguments: tc.arguments,
                    call_id: tc.id,
                };
                let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            }

            // Finalize assistant message.
            if let Some(item) = assistant_item.take() {
                let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
            }

            let _ = tx_event
                .send(Ok(ResponseEvent::Completed {
                    response_id: response_id.clone(),
                    token_usage: Some(TokenUsage {
                        input_tokens,
                        cached_input_tokens: 0,
                        output_tokens,
                        reasoning_output_tokens: 0,
                        total_tokens: input_tokens + output_tokens,
                    }),
                }))
                .await;
            return;
        }

        let chunk: ChatCompletionChunk = match serde_json::from_str(&sse.data) {
            Ok(c) => c,
            Err(e) => {
                trace!("Skipping unparseable Nemotron SSE: {e}");
                continue;
            }
        };

        trace!("nemotron SSE chunk: {chunk:?}");

        // Capture response ID from first chunk.
        if response_id.is_empty() {
            response_id = chunk.id.clone();
        }

        // Emit Created event once.
        if !created_sent {
            let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
            created_sent = true;
        }

        // Capture usage from the final chunk.
        if let Some(usage) = &chunk.usage {
            input_tokens = usage.prompt_tokens;
            output_tokens = usage.completion_tokens;
        }

        for choice in &chunk.choices {
            // ── Text content (with think-tag extraction) ────────────
            if let Some(content) = &choice.delta.content {
                // Ensure we have an assistant message item.
                if assistant_item.is_none() {
                    let item = ResponseItem::Message {
                        id: None,
                        role: "assistant".to_string(),
                        content: vec![],
                        end_turn: None,
                        phase: None,
                    };
                    assistant_item = Some(item.clone());
                    let _ = tx_event
                        .send(Ok(ResponseEvent::OutputItemAdded(item)))
                        .await;
                }

                for delta in think_parser.feed(content) {
                    match delta {
                        ParsedDelta::Text(t) => {
                            if let Some(ResponseItem::Message {
                                content: msg_content,
                                ..
                            }) = &mut assistant_item
                            {
                                msg_content.push(ContentItem::OutputText { text: t.clone() });
                            }
                            let _ = tx_event.send(Ok(ResponseEvent::OutputTextDelta(t))).await;
                        }
                        ParsedDelta::Thinking(t) => {
                            let _ = tx_event
                                .send(Ok(ResponseEvent::ReasoningContentDelta {
                                    delta: t,
                                    content_index: 0,
                                }))
                                .await;
                        }
                    }
                }
            }

            // ── Tool calls ──────────────────────────────────────────
            if let Some(tcs) = &choice.delta.tool_calls {
                for tc in tcs {
                    let state = tool_calls.entry(tc.index).or_default();
                    if let Some(id) = &tc.id {
                        state.id.clone_from(id);
                    }
                    if let Some(func) = &tc.function {
                        if let Some(name) = &func.name {
                            state.name.clone_from(name);
                        }
                        if let Some(args) = &func.arguments {
                            state.arguments.push_str(args);
                        }
                    }
                }
            }

            // ── Finish reason ───────────────────────────────────────
            if let Some(reason) = &choice.finish_reason {
                match reason.as_str() {
                    "tool_calls" => {
                        // Emit pending tool calls.
                        for (_idx, tc) in tool_calls.drain() {
                            if let Some(item) = assistant_item.take() {
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }
                            let item = ResponseItem::FunctionCall {
                                id: None,
                                name: tc.name,
                                namespace: None,
                                arguments: tc.arguments,
                                call_id: tc.id,
                            };
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }
                    }
                    "stop" | "length" => {
                        // Flush think parser on stop.
                        for delta in think_parser.flush() {
                            match delta {
                                ParsedDelta::Text(t) => {
                                    if let Some(ResponseItem::Message { content, .. }) =
                                        &mut assistant_item
                                    {
                                        content.push(ContentItem::OutputText { text: t.clone() });
                                    }
                                    let _ =
                                        tx_event.send(Ok(ResponseEvent::OutputTextDelta(t))).await;
                                }
                                ParsedDelta::Thinking(t) => {
                                    let _ = tx_event
                                        .send(Ok(ResponseEvent::ReasoningContentDelta {
                                            delta: t,
                                            content_index: 0,
                                        }))
                                        .await;
                                }
                            }
                        }
                        if let Some(item) = assistant_item.take() {
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_think_parser_basic() {
        let mut parser = ThinkTagParser::default();
        let deltas = parser.feed("<think>reasoning here</think>actual output");
        let mut thinking = String::new();
        let mut text = String::new();
        for d in deltas {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(t) => text.push_str(&t),
            }
        }
        for d in parser.flush() {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(t) => text.push_str(&t),
            }
        }
        assert_eq!(thinking, "reasoning here");
        assert_eq!(text, "actual output");
    }

    #[test]
    fn test_think_parser_across_chunks() {
        let mut parser = ThinkTagParser::default();
        let mut thinking = String::new();
        let mut text = String::new();

        // Feed in small chunks that split across tag boundaries.
        for chunk in ["<thi", "nk>reas", "oning</thi", "nk>hello"] {
            for d in parser.feed(chunk) {
                match d {
                    ParsedDelta::Thinking(t) => thinking.push_str(&t),
                    ParsedDelta::Text(t) => text.push_str(&t),
                }
            }
        }
        for d in parser.flush() {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(t) => text.push_str(&t),
            }
        }
        assert_eq!(thinking, "reasoning");
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_think_parser_no_tags() {
        let mut parser = ThinkTagParser::default();
        let deltas = parser.feed("just plain text");
        let mut text = String::new();
        for d in deltas {
            match d {
                ParsedDelta::Text(t) => text.push_str(&t),
                ParsedDelta::Thinking(_) => panic!("unexpected thinking delta"),
            }
        }
        for d in parser.flush() {
            match d {
                ParsedDelta::Text(t) => text.push_str(&t),
                ParsedDelta::Thinking(_) => panic!("unexpected thinking delta"),
            }
        }
        assert_eq!(text, "just plain text");
    }

    #[test]
    fn test_think_parser_multiple_blocks() {
        let mut parser = ThinkTagParser::default();
        let mut thinking = String::new();
        let mut text = String::new();

        for d in parser.feed("<think>first</think>mid<think>second</think>end") {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(t) => text.push_str(&t),
            }
        }
        for d in parser.flush() {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(t) => text.push_str(&t),
            }
        }
        assert_eq!(thinking, "firstsecond");
        assert_eq!(text, "midend");
    }

    #[test]
    fn test_think_parser_flush_inside_think() {
        let mut parser = ThinkTagParser::default();
        let mut thinking = String::new();

        for d in parser.feed("<think>incomplete reasoning") {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(_) => {}
            }
        }
        // Flush without closing tag — remaining content treated as thinking.
        for d in parser.flush() {
            match d {
                ParsedDelta::Thinking(t) => thinking.push_str(&t),
                ParsedDelta::Text(_) => {}
            }
        }
        assert_eq!(thinking, "incomplete reasoning");
    }

    #[test]
    fn test_pick_best_model_exact_match() {
        let models = vec![
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        ];
        assert_eq!(
            pick_best_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", &models),
            Some("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        );
    }

    #[test]
    fn test_pick_best_model_substring() {
        let models = vec![
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "nemotron-2025-Q4",
        ];
        // Partial slug should match all two NVIDIA models; shortest wins.
        assert_eq!(
            pick_best_model("nvidia/NVIDIA-Nemotron-3-Nano", &models),
            Some("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        );
    }

    #[test]
    fn test_pick_best_model_case_insensitive() {
        let models = vec!["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"];
        assert_eq!(
            pick_best_model("nemotron-3-nano", &models),
            Some("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        );
    }

    #[test]
    fn test_pick_best_model_no_match() {
        let models = vec!["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"];
        assert_eq!(pick_best_model("llama-3", &models), None);
    }

    #[test]
    fn test_pick_best_model_prefers_shortest() {
        let models = vec![
            "nemotron-2026-Q1-preview-extra-long-name",
            "nemotron-2026-Q1",
        ];
        assert_eq!(
            pick_best_model("nemotron-2026", &models),
            Some("nemotron-2026-Q1")
        );
    }
}
