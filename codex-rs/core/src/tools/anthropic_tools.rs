//! Anthropic-specific tool conversion.
//!
//! Converts Codex [`ToolSpec`] definitions into the Anthropic tool format
//! expected by the Messages API.

use crate::anthropic_types::AnthropicTool;
use codex_protocol::error::Result;
use codex_tools::ToolSpec;

/// Returns Anthropic-compatible tool definitions.
///
/// Only `ToolSpec::Function` variants are translated; `LocalShell`, `WebSearch`,
/// `ImageGeneration`, `ToolSearch`, and `Freeform` are filtered out because
/// the Anthropic Messages API does not support these OpenAI-specific tool
/// types.
pub(crate) fn create_tools_json_for_anthropic_api(
    tools: &[ToolSpec],
) -> Result<Vec<AnthropicTool>> {
    let mut result = Vec::new();

    for tool in tools {
        if let ToolSpec::Function(func) = tool {
            let input_schema = serde_json::to_value(&func.parameters)?;
            result.push(AnthropicTool {
                name: func.name.clone(),
                description: func.description.clone(),
                input_schema,
            });
        }
    }
    Ok(result)
}
