//! Anthropic Messages API streaming provider.
//!
//! This module implements the translation layer between Codex's internal
//! [`Prompt`] / [`ResponseEvent`] types and the Anthropic Messages API wire
//! format. It builds a request, posts it, and then spawns a task that parses
//! the SSE stream and forwards [`ResponseEvent`]s through a channel.

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
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

use crate::anthropic_types::ContentBlock;
use crate::anthropic_types::ContentBlockDelta;
use crate::anthropic_types::ContentBlockParam;
use crate::anthropic_types::MessageContent;
use crate::anthropic_types::MessageParam;
use crate::anthropic_types::MessagesRequest;
use crate::anthropic_types::StreamEvent;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::tools::anthropic_tools::create_tools_json_for_anthropic_api;
use crate::util::backoff;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;

/// Default max_tokens for Anthropic models. Claude 3.5/4 Sonnet supports up to
/// 8192, Claude 3 Opus up to 4096. We use 8192 as a safe default; the model
/// will simply stop if it hits its own lower limit.
const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Stream a completion from the Anthropic Messages API.
pub(crate) async fn stream_anthropic_messages(
    prompt: &Prompt,
    model_info: &ModelInfo,
    provider: &ModelProviderInfo,
) -> Result<ResponseStream> {
    if prompt.output_schema.is_some() {
        return Err(CodexErr::UnsupportedOperation(
            "output_schema is not supported for Anthropic Messages API".to_string(),
        ));
    }

    // ── Build messages ──────────────────────────────────────────────────
    let system_text = &prompt.base_instructions.text;
    let input = prompt.get_formatted_input();
    let mut messages: Vec<MessageParam> = Vec::new();

    for item in &input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let anthropic_role = match role.as_str() {
                    "assistant" => "assistant",
                    "system" => {
                        // Anthropic does not allow system role in messages;
                        // system content is passed via the top-level `system`
                        // field. Skip here.
                        continue;
                    }
                    _ => "user",
                };

                let mut text = String::new();
                let mut blocks: Vec<ContentBlockParam> = Vec::new();
                let mut saw_image = false;

                for c in content {
                    match c {
                        ContentItem::InputText { text: t }
                        | ContentItem::OutputText { text: t } => {
                            text.push_str(t);
                            blocks.push(ContentBlockParam::Text { text: t.clone() });
                        }
                        ContentItem::InputImage { image_url, .. } => {
                            saw_image = true;
                            blocks.push(ContentBlockParam::Text {
                                text: format!("[Image: {image_url}]"),
                            });
                        }
                    }
                }

                // Skip messages with empty/whitespace-only text content when
                // there are no images.
                if !saw_image && text.trim().is_empty() {
                    continue;
                }

                let msg_content = if saw_image {
                    MessageContent::Blocks(blocks)
                } else {
                    MessageContent::Text(text)
                };

                messages.push(MessageParam {
                    role: anthropic_role.to_string(),
                    content: msg_content,
                });
            }

            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                // Surface malformed tool arguments rather than silently replacing
                // them with an empty object, which could cause a tool to execute
                // with incorrect inputs.
                let input_value: Value = serde_json::from_str(arguments).map_err(|e| {
                    CodexErr::UnsupportedOperation(format!(
                        "Invalid JSON in tool arguments for `{name}`: {e}"
                    ))
                })?;
                let block = ContentBlockParam::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: input_value,
                };
                messages.push(MessageParam {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![block]),
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
                let block = ContentBlockParam::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: Some(content_text),
                    is_error: None,
                };
                messages.push(MessageParam {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![block]),
                });
            }

            ResponseItem::LocalShellCall { id, action, .. } => {
                let input_value = serde_json::to_value(action).unwrap_or(Value::Null);
                let block = ContentBlockParam::ToolUse {
                    id: id.clone().unwrap_or_default(),
                    name: "shell".to_string(),
                    input: input_value,
                };
                messages.push(MessageParam {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![block]),
                });
            }

            ResponseItem::CustomToolCall {
                id,
                name,
                input: tool_input,
                ..
            } => {
                let input_value: Value =
                    serde_json::from_str(tool_input).unwrap_or(Value::Object(Default::default()));
                let block = ContentBlockParam::ToolUse {
                    id: id.clone().unwrap_or_default(),
                    name: name.clone(),
                    input: input_value,
                };
                messages.push(MessageParam {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![block]),
                });
            }

            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let content_text = output.text_content().unwrap_or_default().to_string();
                let block = ContentBlockParam::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: Some(content_text),
                    is_error: None,
                };
                messages.push(MessageParam {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![block]),
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

    // ── Coalesce adjacent messages with the same role ────────────────────
    messages = coalesce_messages(messages);

    // ── Strip empty text blocks ─────────────────────────────────────────
    strip_empty_text_blocks(&mut messages);

    // ── Build tools ─────────────────────────────────────────────────────
    let tools = create_tools_json_for_anthropic_api(&prompt.tools)?;
    let tools_param = if tools.is_empty() { None } else { Some(tools) };

    // ── Build request ───────────────────────────────────────────────────
    let request = MessagesRequest {
        model: &model_info.slug,
        max_tokens: DEFAULT_MAX_TOKENS,
        messages,
        system: Some(system_text),
        temperature: None,
        tools: tools_param,
        tool_choice: None,
        stream: true,
    };

    let payload = serde_json::to_value(&request).map_err(|e| {
        CodexErr::UnsupportedOperation(format!("Failed to serialize Anthropic request: {e}"))
    })?;

    // ── Resolve URL + headers from provider ─────────────────────────────
    let api_provider = provider.to_api_provider(/*auth_mode*/ None)?;
    let url = api_provider.url_for_path("/messages");
    let mut header_map = reqwest::header::HeaderMap::new();
    for (name, value) in api_provider.headers.iter() {
        if let Ok(rname) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes())
            && let Ok(rvalue) = reqwest::header::HeaderValue::from_bytes(value.as_bytes())
        {
            header_map.insert(rname, rvalue);
        }
    }

    debug!("Anthropic POST to {url}");

    // ── Execute with retries ────────────────────────────────────────────
    let http_client = reqwest::Client::new();
    let mut attempt = 0;
    let max_retries = provider.request_max_retries();
    loop {
        attempt += 1;

        let res = http_client
            .post(&url)
            .headers(header_map.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent>>(1600);
                let idle_timeout = provider.stream_idle_timeout();
                tokio::spawn(async move {
                    process_anthropic_response(resp, tx_event, idle_timeout).await;
                });
                return Ok(ResponseStream { rx_event });
            }
            Ok(resp) => {
                let status = resp.status();
                if !(status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()) {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(CodexErr::UnsupportedOperation(format!(
                        "Anthropic API returned {status}: {body}"
                    )));
                }

                if attempt > max_retries {
                    return Err(CodexErr::UnsupportedOperation(format!(
                        "Anthropic API retry limit reached (status {status})"
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
                        "Anthropic connection failed after retries: {e}"
                    )));
                }
                let delay = backoff(attempt);
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// SSE processor for the Anthropic Messages streaming format.
async fn process_anthropic_response(
    response: reqwest::Response,
    tx_event: mpsc::Sender<Result<ResponseEvent>>,
    idle_timeout: Duration,
) {
    let byte_stream = response.bytes_stream();
    let mut stream = byte_stream.eventsource();

    #[derive(Default)]
    struct BlockState {
        kind: BlockKind,
        text: String,
        tool_name: String,
        tool_id: String,
        tool_args: String,
    }

    #[derive(Default, PartialEq)]
    enum BlockKind {
        #[default]
        Text,
        ToolUse,
        Thinking,
    }

    let mut blocks: HashMap<usize, BlockState> = HashMap::new();
    let mut response_id = String::new();
    let mut assistant_item: Option<ResponseItem> = None;

    // Token usage tracking: accumulated from message_start and message_delta SSE events.
    let mut input_tokens: i64 = 0;
    let mut output_tokens: i64 = 0;
    let mut cached_input_tokens: i64 = 0;

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
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id,
                        token_usage: Some(TokenUsage {
                            input_tokens,
                            cached_input_tokens,
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
                        "idle timeout waiting for Anthropic SSE".into(),
                        None,
                    )))
                    .await;
                return;
            }
        };

        let event: StreamEvent = match serde_json::from_str(&sse.data) {
            Ok(ev) => ev,
            Err(e) => {
                trace!("Skipping unparseable Anthropic SSE: {e}");
                continue;
            }
        };

        trace!("anthropic SSE event: {event:?}");

        match event {
            StreamEvent::MessageStart { message } => {
                response_id = message.id;
                input_tokens = i64::from(message.usage.input_tokens);
                output_tokens = i64::from(message.usage.output_tokens);
                cached_input_tokens = i64::from(message.usage.cache_read_input_tokens.unwrap_or(0));
                let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
            }

            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match &content_block {
                ContentBlock::Text { .. } => {
                    blocks.insert(
                        index,
                        BlockState {
                            kind: BlockKind::Text,
                            ..Default::default()
                        },
                    );
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
                }
                ContentBlock::ToolUse { id, name, .. } => {
                    blocks.insert(
                        index,
                        BlockState {
                            kind: BlockKind::ToolUse,
                            tool_name: name.clone(),
                            tool_id: id.clone(),
                            ..Default::default()
                        },
                    );
                }
                ContentBlock::Thinking { .. } => {
                    blocks.insert(
                        index,
                        BlockState {
                            kind: BlockKind::Thinking,
                            ..Default::default()
                        },
                    );
                }
            },

            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Some(state) = blocks.get_mut(&index) {
                    match delta {
                        ContentBlockDelta::TextDelta { text } => {
                            state.text.push_str(&text);
                            if let Some(ResponseItem::Message { content, .. }) = &mut assistant_item
                            {
                                content.push(ContentItem::OutputText { text: text.clone() });
                            }
                            let _ = tx_event
                                .send(Ok(ResponseEvent::OutputTextDelta(text)))
                                .await;
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            state.tool_args.push_str(&partial_json);
                        }
                        ContentBlockDelta::ThinkingDelta { thinking } => {
                            state.text.push_str(&thinking);
                            let _ = tx_event
                                .send(Ok(ResponseEvent::ReasoningContentDelta {
                                    delta: thinking,
                                    content_index: 0,
                                }))
                                .await;
                        }
                        ContentBlockDelta::SignatureDelta { .. } => {}
                    }
                }
            }

            StreamEvent::ContentBlockStop { index } => {
                if let Some(state) = blocks.remove(&index) {
                    match state.kind {
                        BlockKind::ToolUse => {
                            if let Some(item) = assistant_item.take() {
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }

                            let item = ResponseItem::FunctionCall {
                                id: None,
                                name: state.tool_name,
                                namespace: None,
                                arguments: state.tool_args,
                                call_id: state.tool_id,
                            };
                            // Preserve the Added → Done lifecycle that downstream
                            // consumers expect for every output item.
                            let _ = tx_event
                                .send(Ok(ResponseEvent::OutputItemAdded(item.clone())))
                                .await;
                            let _ = tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                        }
                        BlockKind::Text => {
                            // Text accumulated in assistant_item; finalized at message_stop.
                        }
                        BlockKind::Thinking => {
                            if !state.text.is_empty() {
                                let item = ResponseItem::Reasoning {
                                    id: String::new(),
                                    summary: Vec::new(),
                                    content: Some(vec![
                                        codex_protocol::models::ReasoningItemContent::ReasoningText {
                                            text: state.text,
                                        },
                                    ]),
                                    encrypted_content: None,
                                };
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }
                        }
                    }
                }
            }

            StreamEvent::MessageDelta { delta, usage } => {
                // message_delta carries the final cumulative output token count.
                output_tokens = i64::from(usage.output_tokens);
                if let Some(stop_reason) = delta.stop_reason {
                    use crate::anthropic_types::StopReason;
                    match stop_reason {
                        StopReason::EndTurn
                        | StopReason::StopSequence
                        | StopReason::MaxTokens
                        | StopReason::ToolUse => {
                            if let Some(item) = assistant_item.take() {
                                let _ =
                                    tx_event.send(Ok(ResponseEvent::OutputItemDone(item))).await;
                            }
                        }
                    }
                }
            }

            StreamEvent::MessageStop => {
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id: response_id.clone(),
                        token_usage: Some(TokenUsage {
                            input_tokens,
                            cached_input_tokens,
                            output_tokens,
                            reasoning_output_tokens: 0,
                            total_tokens: input_tokens + output_tokens,
                        }),
                    }))
                    .await;
                return;
            }

            StreamEvent::Ping => {}

            StreamEvent::Error { error } => {
                let _ = tx_event
                    .send(Err(CodexErr::Stream(
                        format!("Anthropic error: {}", error.message),
                        None,
                    )))
                    .await;
                return;
            }
        }
    }
}

/// Coalesce adjacent messages with the same role into a single message.
///
/// Anthropic requires strictly alternating `user` / `assistant` roles in the
/// messages array. When the Codex conversation history produces consecutive
/// messages with the same role (e.g. multiple tool results → multiple "user"
/// messages), we merge them into a single message with a `Blocks` content
/// containing all the content blocks from the original messages.
fn coalesce_messages(messages: Vec<MessageParam>) -> Vec<MessageParam> {
    let mut result: Vec<MessageParam> = Vec::with_capacity(messages.len());
    for msg in messages {
        if let Some(last) = result.last_mut()
            && last.role == msg.role
        {
            let existing_blocks = content_to_blocks(&mut last.content);
            let new_blocks = match msg.content {
                MessageContent::Text(t) => vec![ContentBlockParam::Text { text: t }],
                MessageContent::Blocks(b) => b,
            };
            existing_blocks.extend(new_blocks);
        } else {
            result.push(msg);
        }
    }
    result
}

/// Convert a `MessageContent` to blocks in-place, returning a mutable
/// reference to the blocks vector for appending.
fn content_to_blocks(content: &mut MessageContent) -> &mut Vec<ContentBlockParam> {
    match content {
        MessageContent::Blocks(blocks) => blocks,
        MessageContent::Text(_) => {
            let old = std::mem::replace(content, MessageContent::Blocks(Vec::new()));
            if let MessageContent::Text(t) = old
                && let MessageContent::Blocks(blocks) = content
            {
                blocks.push(ContentBlockParam::Text { text: t });
                return blocks;
            }
            match content {
                MessageContent::Blocks(blocks) => blocks,
                _ => unreachable!(),
            }
        }
    }
}

/// Remove `ContentBlockParam::Text` entries whose text is empty or
/// whitespace-only.  If stripping leaves a `Blocks` list empty, the
/// entire message is dropped.
fn strip_empty_text_blocks(messages: &mut Vec<MessageParam>) {
    for msg in messages.iter_mut() {
        match &mut msg.content {
            MessageContent::Text(t) if t.trim().is_empty() => {
                msg.content = MessageContent::Blocks(Vec::new());
            }
            MessageContent::Blocks(blocks) => {
                blocks.retain(
                    |b| !matches!(b, ContentBlockParam::Text { text } if text.trim().is_empty()),
                );
            }
            _ => {}
        }
    }
    messages.retain(|msg| !matches!(&msg.content, MessageContent::Blocks(b) if b.is_empty()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesce_empty() {
        let result = coalesce_messages(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_coalesce_no_merge_needed() {
        let messages = vec![
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            },
            MessageParam {
                role: "assistant".to_string(),
                content: MessageContent::Text("Hi".to_string()),
            },
        ];
        let result = coalesce_messages(messages);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_coalesce_merges_adjacent_user_messages() {
        let messages = vec![
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            },
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Text("World".to_string()),
            },
            MessageParam {
                role: "assistant".to_string(),
                content: MessageContent::Text("Hi".to_string()),
            },
        ];
        let result = coalesce_messages(messages);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, "user");
        match &result[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
            }
            _ => panic!("Expected Blocks content after coalescing"),
        }
    }

    #[test]
    fn test_coalesce_tool_results() {
        let messages = vec![
            MessageParam {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(vec![ContentBlockParam::ToolUse {
                    id: "tool1".to_string(),
                    name: "shell".to_string(),
                    input: serde_json::Value::Null,
                }]),
            },
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlockParam::ToolResult {
                    tool_use_id: "tool1".to_string(),
                    content: Some("result1".to_string()),
                    is_error: None,
                }]),
            },
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlockParam::ToolResult {
                    tool_use_id: "tool2".to_string(),
                    content: Some("result2".to_string()),
                    is_error: None,
                }]),
            },
        ];
        let result = coalesce_messages(messages);
        assert_eq!(result.len(), 2);
        match &result[1].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn test_strip_empty_text_blocks_removes_empty_text() {
        let mut messages = vec![MessageParam {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(vec![
                ContentBlockParam::Text {
                    text: "".to_string(),
                },
                ContentBlockParam::ToolUse {
                    id: "t1".to_string(),
                    name: "shell".to_string(),
                    input: serde_json::Value::Null,
                },
            ]),
        }];
        strip_empty_text_blocks(&mut messages);
        assert_eq!(messages.len(), 1);
        match &messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(&blocks[0], ContentBlockParam::ToolUse { .. }));
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn test_strip_empty_text_blocks_drops_empty_message() {
        let mut messages = vec![
            MessageParam {
                role: "assistant".to_string(),
                content: MessageContent::Text("  ".to_string()),
            },
            MessageParam {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            },
        ];
        strip_empty_text_blocks(&mut messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }
}
