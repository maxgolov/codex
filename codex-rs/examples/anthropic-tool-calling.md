# Running Codex with Anthropic + Tool Calling

This guide shows how to build and run Codex with the Anthropic Messages API
provider, including shell tool calling and sandbox configuration.

## Prerequisites

1. An Anthropic API key from <https://console.anthropic.com/settings/keys>
2. Rust toolchain (see `codex-rs/rust-toolchain.toml`)

## 1. Set your API key

Create a `.env.secret` file in the repo root (git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Or export it directly:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

## 2. Configure `config.toml`

Create or edit `~/.codex/config.toml`:

```toml
model = "claude-sonnet-4-20250514"
model_provider = "anthropic"
```

The built-in `anthropic` provider is pre-registered with:
- Base URL: `https://api.anthropic.com/v1` (override with `ANTHROPIC_BASE_URL`)
- Auth: `x-api-key` header populated from `ANTHROPIC_API_KEY` env var
- Wire API: Anthropic Messages (`/messages` endpoint with SSE streaming)

### Custom endpoint (e.g. proxy or gateway)

```toml
model = "claude-sonnet-4-20250514"
model_provider = "my-anthropic"

[model_providers.my-anthropic]
name = "My Anthropic Proxy"
base_url = "https://my-proxy.example.com/v1"
wire_api = "anthropic"
env_http_headers = { "x-api-key" = "ANTHROPIC_API_KEY" }
http_headers = { "anthropic-version" = "2023-06-01", "content-type" = "application/json" }
```

## 3. Build

```powershell
cd codex-rs
cargo build -p codex-cli --release
```

## 4. Run (interactive TUI)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
cargo run -p codex-cli --release
```

## 5. Run (non-interactive / scripted)

```powershell
cargo run -p codex-exec --release -- "List the files in the current directory"
```

Or with explicit model selection (overrides config.toml):

```powershell
cargo run -p codex-exec --release -- --model claude-sonnet-4-20250514 --provider anthropic "What Rust version is this project using?"
```

## How tool calling works

When Claude returns a tool_use content block, Codex:

1. **Parses** the SSE `content_block_start` / `content_block_delta` / `content_block_stop`
   events and assembles a `ResponseItem::FunctionCall` with `name`, `arguments`, and `call_id`.

2. **Dispatches** the tool call through the standard Codex tool runtime:
   - **Shell commands** → spawned via PTY with sandbox enforcement
   - **MCP tools** → forwarded to the configured MCP server
   - **Custom tools** → handled by registered tool handlers

3. **Returns** the result as a `tool_result` content block in the next Anthropic
   request, with the matching `tool_use_id`.

4. Claude sees the tool result and can continue generating text or invoke
   additional tools.

### Supported tool types over Anthropic

| Codex Tool              | Anthropic Mapping | Notes |
|------------------------|-------------------|-------|
| `LocalShell`           | `tool_use` → shell execution → `tool_result` | Full sandbox support |
| `Function` (MCP tools) | `tool_use` → MCP call → `tool_result` | Standard MCP flow |
| `Freeform` (custom)    | `tool_use` → custom handler → `tool_result` | Custom tool handlers |
| `WebSearch`            | Not sent to Anthropic | OpenAI-specific; skipped |
| `ImageGeneration`      | Not sent to Anthropic | OpenAI-specific; skipped |
| `ToolSearch`           | Not sent to Anthropic | OpenAI-specific; skipped |

## Sandbox & code execution

Codex applies the **same sandbox policies** regardless of provider. The sandbox
mode is configured independently of the model provider:

### Sandbox modes

```toml
# In config.toml:
sandbox_mode = "read-only"   # Default: model can read but not write
# sandbox_mode = "full"      # Model can read and write (use carefully)
```

### Windows-specific sandbox

On Windows, Codex can run shell commands in an isolated sandbox:

```toml
[windows]
sandbox = "Unelevated"       # Reduced token without UAC prompt
# sandbox = "Elevated"       # Full process isolation (requires UAC)
# sandbox = "Disabled"       # No sandboxing
```

### What the sandbox enforces

- **Network isolation**: `CODEX_SANDBOX_NETWORK_DISABLED=1` is set on child
  processes, preventing outbound connections from tool executions.
- **Filesystem restrictions**: Based on `sandbox_mode`, writes may be blocked
  or restricted to the working directory.
- **Process isolation**: On Windows with `Elevated` mode, a restricted token
  limits the process capabilities.
- **Output limits**: Tool output is capped at ~1 MiB per process to prevent
  context overflow.

### Approval flow

When a tool call requires approval (e.g. a shell command that writes files in
`read-only` mode), Codex prompts the user before execution. In non-interactive
mode (`codex exec`), unapproved commands are denied by default.

## Extended thinking / reasoning

Claude models that support extended thinking will emit `thinking` content blocks.
These are mapped to `ResponseItem::Reasoning` and displayed as reasoning traces
in the TUI. The `ReasoningContentDelta` events stream thinking tokens in real time.

## Retry behavior

The Anthropic provider respects the standard retry configuration:

```toml
[model_providers.anthropic]
request_max_retries = 3          # Default retry count
stream_idle_timeout_ms = 30000   # SSE idle timeout
```

On 429 (rate limit) or 5xx errors, requests are retried with exponential
backoff. The `Retry-After` header is honored when present.
