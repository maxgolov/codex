//! Integration tests for the Nemotron/vLLM adapter.
//!
//! These tests require a running vLLM instance at `localhost:8002` (or the
//! URL in `NVIDIA_BASE_URL`). They are skipped automatically when the
//! endpoint is not reachable.

use std::time::Duration;

use codex_core::ResponseEvent;
use codex_model_provider_info::create_nemotron_provider;
use codex_protocol::openai_models::ModelInfo;
use tokio::time::timeout;

/// Returns the base URL for the Nemotron endpoint.
fn nemotron_base_url() -> String {
    std::env::var("NVIDIA_BASE_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "http://localhost:8002/v1".to_string())
}

/// Quick health check: can we reach `/v1/models`?
async fn vllm_is_available() -> bool {
    let url = format!("{}/models", nemotron_base_url());
    let Ok(resp) = timeout(Duration::from_secs(5), reqwest::get(&url)).await else {
        return false;
    };
    resp.is_ok_and(|r| r.status().is_success())
}

/// Build a minimal `ModelInfo` suitable for calling the adapter.
fn test_model_info(slug: &str) -> ModelInfo {
    use codex_models_manager::model_info::model_info_from_slug;
    let mut info = model_info_from_slug(slug);
    info.slug = slug.to_string();
    info.context_window = Some(128_000);
    info
}

/// Helper: run a single streaming turn and collect all events.
async fn run_nemotron_turn(model_slug: &str, user_message: &str) -> Vec<ResponseEvent> {
    use codex_core::Prompt;
    use codex_protocol::models::{BaseInstructions, ContentItem, ResponseItem};
    use futures::StreamExt;

    let provider = create_nemotron_provider();
    let model_info = test_model_info(model_slug);

    let mut prompt = Prompt::default();
    prompt.base_instructions = BaseInstructions {
        text: "You are a helpful assistant.".to_string(),
    };
    prompt.input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: user_message.to_string(),
        }],
        end_turn: None,
        phase: None,
    }];

    let mut stream = codex_core::nemotron::stream_nemotron_chat(&prompt, &model_info, &provider)
        .await
        .expect("stream_nemotron_chat failed");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            Ok(ev) => events.push(ev),
            Err(e) => panic!("Unexpected error event: {e:?}"),
        }
    }
    events
}

#[tokio::test]
async fn nemotron_simple_chat_returns_text() {
    if !vllm_is_available().await {
        eprintln!("SKIP: vLLM not available at {}", nemotron_base_url());
        return;
    }

    let events = run_nemotron_turn("auto", "What is 2+2? Reply with just the number.").await;

    let has_server_model = events
        .iter()
        .any(|e| matches!(e, ResponseEvent::ServerModel(_)));
    let has_created = events.iter().any(|e| matches!(e, ResponseEvent::Created));
    let has_completed = events
        .iter()
        .any(|e| matches!(e, ResponseEvent::Completed { .. }));

    let text: String = events
        .iter()
        .filter_map(|e| {
            if let ResponseEvent::OutputTextDelta(t) = e {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect();

    assert!(has_server_model, "expected ServerModel event");
    assert!(has_created, "expected Created event");
    assert!(has_completed, "expected Completed event");
    assert!(
        text.contains('4'),
        "expected '4' in response text, got: {text}"
    );

    // Check token usage.
    let usage = events.iter().find_map(|e| {
        if let ResponseEvent::Completed { token_usage, .. } = e {
            token_usage.as_ref()
        } else {
            None
        }
    });
    assert!(usage.is_some(), "expected token usage in Completed event");
    let usage = usage.expect("checked above");
    assert!(usage.input_tokens > 0, "expected input_tokens > 0");
    assert!(usage.output_tokens > 0, "expected output_tokens > 0");
}

#[tokio::test]
async fn nemotron_model_auto_detect() {
    if !vllm_is_available().await {
        eprintln!("SKIP: vLLM not available at {}", nemotron_base_url());
        return;
    }

    let events = run_nemotron_turn("auto", "Say hello").await;

    let server_model: Option<String> = events.iter().find_map(|e| {
        if let ResponseEvent::ServerModel(m) = e {
            Some(m.clone())
        } else {
            None
        }
    });

    assert!(
        server_model.is_some(),
        "expected ServerModel event with resolved model name"
    );
    let model = server_model.expect("checked above");
    assert_ne!(model, "auto", "model should have been resolved from 'auto'");
    assert!(!model.is_empty(), "resolved model name should not be empty");
}

#[tokio::test]
async fn nemotron_partial_model_name() {
    if !vllm_is_available().await {
        eprintln!("SKIP: vLLM not available at {}", nemotron_base_url());
        return;
    }

    // Use a partial name that should substring-match to the full model.
    let events = run_nemotron_turn("Nemotron-3-Nano", "Say hi").await;

    let server_model: Option<String> = events.iter().find_map(|e| {
        if let ResponseEvent::ServerModel(m) = e {
            Some(m.clone())
        } else {
            None
        }
    });

    assert!(
        server_model.is_some(),
        "expected ServerModel event from partial name"
    );
    let model = server_model.expect("checked above");
    assert!(
        model.to_ascii_lowercase().contains("nemotron"),
        "resolved model should contain 'nemotron', got: {model}"
    );
}
