//! Nemotron/vLLM-specific tool conversion.
//!
//! Converts Codex [`ToolSpec`] definitions into the OpenAI Chat Completions
//! tool format, with schema flattening to work around vLLM's Jinja2 chat
//! template limitations with `oneOf`/`anyOf` union types.

use crate::nemotron_types::ChatTool;
use crate::nemotron_types::ChatToolFunction;
use codex_protocol::error::Result;
use codex_tools::ToolSpec;
use serde_json::Value;

/// Convert Codex tool specs to Chat Completions tool definitions.
///
/// Only `ToolSpec::Function` variants are translated. Tool parameter schemas
/// are flattened to remove `oneOf`/`anyOf` constructs that vLLM's Jinja2
/// chat template cannot handle.
pub(crate) fn create_tools_for_nemotron(tools: &[ToolSpec]) -> Result<Vec<ChatTool>> {
    let mut result = Vec::new();

    for tool in tools {
        if let ToolSpec::Function(func) = tool {
            let mut parameters = serde_json::to_value(&func.parameters)?;
            flatten_schema(&mut parameters);
            result.push(ChatTool {
                r#type: "function".to_string(),
                function: ChatToolFunction {
                    name: func.name.clone(),
                    description: func.description.clone(),
                    parameters,
                },
            });
        }
    }
    Ok(result)
}

/// Recursively flatten `oneOf`/`anyOf` union types in a JSON schema.
///
/// vLLM's Jinja2 chat template cannot handle these constructs, so we replace
/// them with the first variant in the union, preserving any outer description.
fn flatten_schema(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };

    // If this node itself is a oneOf/anyOf, replace it with the first variant.
    // Inspect before removing: an empty or non-array value would otherwise be
    // silently dropped, leaving the schema subtly corrupted.
    for key in &["oneOf", "anyOf"] {
        let Some(first) = obj
            .get(*key)
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .cloned()
        else {
            continue;
        };
        // Safe to remove now that we have a valid first variant in hand.
        obj.remove(*key);
        let desc = obj.remove("description");
        let mut replacement = first;
        // Preserve the outer description if the first variant lacks one.
        if let Some(desc) = desc
            && let Some(rep_obj) = replacement.as_object_mut()
        {
            rep_obj.entry("description").or_insert(desc);
        }
        *schema = replacement;
        flatten_schema(schema);
        return;
    }

    // Recurse into `properties`.
    if let Some(props) = obj.get_mut("properties")
        && let Some(props_obj) = props.as_object_mut()
    {
        for value in props_obj.values_mut() {
            flatten_schema(value);
        }
    }

    // Recurse into `items` (array schemas).
    if let Some(items) = obj.get_mut("items") {
        flatten_schema(items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_flatten_one_of() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "description": "A flexible value",
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            }
        });
        flatten_schema(&mut schema);
        let props = schema["properties"]["value"].as_object().unwrap();
        assert_eq!(props["type"], "string");
        assert_eq!(props["description"], "A flexible value");
        assert!(!props.contains_key("oneOf"));
    }

    #[test]
    fn test_flatten_any_of() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [
                        {"type": "number", "description": "inner desc"},
                        {"type": "null"}
                    ]
                }
            }
        });
        flatten_schema(&mut schema);
        let props = schema["properties"]["data"].as_object().unwrap();
        assert_eq!(props["type"], "number");
        assert_eq!(props["description"], "inner desc");
        assert!(!props.contains_key("anyOf"));
    }

    #[test]
    fn test_flatten_nested_items() {
        let mut schema = json!({
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "boolean"}
                ]
            }
        });
        flatten_schema(&mut schema);
        assert_eq!(schema["items"]["type"], "string");
    }

    #[test]
    fn test_flatten_no_op_on_simple_schema() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        let original = schema.clone();
        flatten_schema(&mut schema);
        assert_eq!(schema, original);
    }
}
