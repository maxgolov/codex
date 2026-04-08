//! Quick smoke-test for the Nemotron/vLLM wire adapter.
//!
//! Usage:
//!     cargo run -p codex-core --example nemotron_hello
//!
//! Environment variables:
//!     NVIDIA_BASE_URL   Override vLLM endpoint (default: http://localhost:8002/v1)
//!     NEMOTRON_MODEL    Override model slug
//!     RUST_LOG          Set log level (default: info)

use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_core::nemotron::stream_nemotron_chat;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use std::env;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model = env::var("NEMOTRON_MODEL")
        .ok()
        .unwrap_or_else(|| "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16".to_string());

    let provider = codex_model_provider_info::create_nemotron_provider();
    let model_info = codex_models_manager::model_info::model_info_from_slug(&model);

    println!("=== Nemotron hello-world smoke test ===");
    println!(
        "  base_url: {}",
        provider.base_url.as_deref().unwrap_or("(none)")
    );
    println!("  model:    {model}");
    println!();

    let mut prompt = Prompt::default();
    prompt.base_instructions = BaseInstructions {
        text: "You are a helpful assistant. Keep answers brief.".to_string(),
    };
    prompt.input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "What is 2 + 2? Reply in one sentence.".to_string(),
        }],
        end_turn: None,
        phase: None,
    }];

    let mut stream = match stream_nemotron_chat(&prompt, &model_info, &provider).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to start stream: {e}");
            std::process::exit(1);
        }
    };

    let mut full_text = String::new();
    let mut got_thinking = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(ResponseEvent::Created) => {
                println!("[Created]");
            }
            Ok(ResponseEvent::OutputItemAdded(item)) => {
                if let ResponseItem::Message { ref role, .. } = item {
                    println!("[OutputItemAdded] role={role}");
                }
            }
            Ok(ResponseEvent::OutputTextDelta(delta)) => {
                print!("{delta}");
                full_text.push_str(&delta);
            }
            Ok(ResponseEvent::ReasoningContentDelta { .. }) => {
                if !got_thinking {
                    println!("[Thinking] ...");
                    got_thinking = true;
                }
            }
            Ok(ResponseEvent::OutputItemDone(ref item)) => match item {
                ResponseItem::Message { .. } => {
                    println!();
                    println!("[OutputItemDone] Message");
                }
                ResponseItem::FunctionCall { name, .. } => {
                    println!("[OutputItemDone] FunctionCall: {name}");
                }
                ResponseItem::Reasoning { .. } => {
                    println!("[OutputItemDone] Reasoning block");
                }
                _ => {
                    println!("[OutputItemDone] other");
                }
            },
            Ok(ResponseEvent::Completed {
                response_id,
                token_usage,
            }) => {
                println!("[Completed] response_id={response_id}");
                if let Some(usage) = token_usage {
                    println!(
                        "  tokens: input={} output={} total={}",
                        usage.input_tokens, usage.output_tokens, usage.total_tokens
                    );
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("ERROR in stream: {e}");
                break;
            }
        }
    }

    println!();
    if full_text.is_empty() {
        eprintln!("WARNING: No text output received!");
        std::process::exit(1);
    } else {
        println!("=== Success! Got {} chars of output ===", full_text.len());
    }
}
