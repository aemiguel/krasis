//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Python is called only for GPU prefill (unavoidable — PyTorch/CUDA).
//! The decode loop runs GIL-free with zero Python involvement.
//!
//! Single-request at a time (matches our hardware constraint).

use crate::gpu_decode::GpuDecodeStore;

/// Streaming detokenizer that buffers incomplete UTF-8 sequences.
///
/// Some characters (emojis, CJK, etc.) span multiple tokens in byte-level BPE.
/// Decoding each token individually produces incomplete UTF-8 bytes → U+FFFD.
/// This struct buffers tokens until the decoded text contains no trailing FFFD,
/// then emits the complete text.
pub struct StreamDetokenizer<'a> {
    tokenizer: &'a tokenizers::Tokenizer,
    pending: Vec<u32>,
}

impl<'a> StreamDetokenizer<'a> {
    pub fn new(tokenizer: &'a tokenizers::Tokenizer) -> Self {
        Self { tokenizer, pending: Vec::new() }
    }

    /// Add a token. Returns the decoded text if the sequence is complete UTF-8,
    /// or an empty string if we're still buffering incomplete bytes.
    pub fn add(&mut self, token_id: u32) -> String {
        self.pending.push(token_id);
        let decoded = self.tokenizer.decode(&self.pending, true).unwrap_or_default();
        if decoded.is_empty() {
            return String::new();
        }
        // If the decoded text ends with U+FFFD, we likely have incomplete bytes.
        // Buffer up to 8 tokens before giving up and emitting anyway.
        if decoded.ends_with('\u{FFFD}') && self.pending.len() < 8 {
            return String::new();
        }
        self.pending.clear();
        decoded
    }

    /// Flush any remaining buffered tokens (end of stream).
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let decoded = self.tokenizer.decode(&self.pending, true).unwrap_or_default();
        self.pending.clear();
        decoded
    }
}
use pyo3::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::sync::mpsc;

/// Global pointer to the server's `running` flag so the raw signal handler
/// can set it to `false` without going through Python's signal mechanism.
/// This is only written once (before the accept loop) and read from the
/// signal handler, so the raw pointer is safe in practice.
static SIGINT_RUNNING: AtomicBool = AtomicBool::new(false);
static SIGNAL_FLAG_PTR: std::sync::atomic::AtomicPtr<AtomicBool> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Raw signal handler for SIGINT and SIGTERM.  Sets the server's `running`
/// flag to false so the accept loop exits cleanly, even when the GIL is
/// released (Python signal handlers can't run during allow_threads).
extern "C" fn shutdown_signal_handler(_sig: libc::c_int) {
    let ptr = SIGNAL_FLAG_PTR.load(Ordering::Acquire);
    if !ptr.is_null() {
        // Safety: ptr points to the Arc<AtomicBool>'s inner value,
        // which outlives this handler (server.run() is still on the stack).
        unsafe { &*ptr }.store(false, Ordering::Release);
    }
    // Also set our own flag so we can detect it was us
    SIGINT_RUNNING.store(true, Ordering::Release);
}

/// Server state shared across request handling.
struct ServerState {
    py_model: Py<PyAny>,
    model_name: String,
    tokenizer: tokenizers::Tokenizer,
    chat_template: crate::chat_template::ChatTemplateEngine,
    max_context_tokens: usize,
    default_enable_thinking: bool,
    /// Token ID for `</think>` — when set, thinking tokens are exempt from max_tokens.
    thinking_end_token: Option<usize>,
    /// Raw pointer to a GpuDecodeStore instance (set from Python during server init).
    /// Safety: single-request guarantee means no concurrent access.
    gpu_store_addr: usize,
    /// When true, log wall-clock time for each Python GIL acquisition.
    /// Enabled by KRASIS_GIL_TIMING=1. Zero cost when off (branch only).
    gil_timing: bool,
    /// When set, write full request JSON to this directory for debugging IDE clients.
    /// Enabled by KRASIS_LOG_REQUESTS=1 (writes to logs/requests/).
    log_requests_dir: Option<String>,
    /// Multi-GPU: auxiliary store addresses (empty = single GPU mode).
    aux_gpu_store_addrs: Vec<usize>,
    /// Multi-GPU: layer indices where each segment boundary falls.
    multi_gpu_split_layers: Vec<usize>,
    /// Multi-GPU: number of GQA layers before each split point (for KV cache indexing).
    multi_gpu_gqa_offsets: Vec<usize>,
}

/// Parsed HTTP request.
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

/// Parse an HTTP request from a TCP stream.
fn parse_request(stream: &mut BufReader<TcpStream>) -> std::io::Result<HttpRequest> {
    // Request line
    let mut request_line = String::new();
    stream.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid request line",
        ));
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Headers
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        stream.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            content_length = val.trim().parse().unwrap_or(0);
        } else if let Some(val) = trimmed.strip_prefix("content-length:") {
            content_length = val.trim().parse().unwrap_or(0);
        }
    }

    // Body
    let mut body = String::new();
    if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        stream.read_exact(&mut buf)?;
        body = String::from_utf8_lossy(&buf).to_string();
    }

    Ok(HttpRequest { method, path, body })
}

/// Send a JSON response.
fn send_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        status_text,
        body.len(),
        body
    )?;
    stream.flush()
}

/// Begin an SSE stream (send headers, return stream for data).
fn begin_sse(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: keep-alive\r\n\r\n"
    )?;
    stream.flush()
}

/// Send one SSE data chunk.
fn send_sse_chunk(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(stream, "data: {}\n\n", data)?;
    stream.flush()
}

/// Format an SSE chunk as OpenAI chat.completion.chunk JSON.
fn format_sse_token(
    request_id: &str,
    model_name: &str,
    text: &str,
    finish_reason: Option<&str>,
    created: u64,
) -> String {
    let delta = if text.is_empty() {
        "{}".to_string()
    } else {
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"")
            .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
        format!(r#"{{"content":"{}"}}"#, escaped)
    };
    let fr = match finish_reason {
        Some(r) => format!(r#""{}""#, r),
        None => "null".to_string(),
    };
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{},"finish_reason":{}}}]}}"#,
        request_id, created, model_name, delta, fr
    )
}

/// Format a complete (non-streaming) chat completion response.
fn format_completion(
    request_id: &str,
    model_name: &str,
    text: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
    created: u64,
) -> String {
    let escaped = text.replace('\\', "\\\\").replace('"', "\\\"")
        .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
    format!(
        r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"{}"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id, created, model_name, escaped, finish_reason,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    )
}

// ── Tool use support ──────────────────────────────────────────────

/// A parsed tool call extracted from model output.
struct ParsedToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

/// Parse tool calls from model-generated text (Qwen XML format).
/// Returns (content_text, tool_calls).
/// Content is everything outside `<tool_call>...</tool_call>` blocks.
fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut remaining = text;
    let mut call_idx = 0u64;

    while let Some(start) = remaining.find("<tool_call>") {
        content.push_str(&remaining[..start]);
        remaining = &remaining[start + "<tool_call>".len()..];

        if let Some(end) = remaining.find("</tool_call>") {
            let block = remaining[..end].trim();
            remaining = &remaining[end + "</tool_call>".len()..];

            // Parse <function=name>
            if let Some(fn_start) = block.find("<function=") {
                let after = &block[fn_start + "<function=".len()..];
                if let Some(fn_end) = after.find('>') {
                    let name = after[..fn_end].to_string();
                    let inner = &after[fn_end + 1..];

                    // Find </function> boundary
                    let params_text = if let Some(fe) = inner.find("</function>") {
                        &inner[..fe]
                    } else {
                        inner
                    };

                    // Parse <parameter=name>value</parameter> pairs
                    let mut args = serde_json::Map::new();
                    let mut param_rem = params_text;
                    while let Some(p_start) = param_rem.find("<parameter=") {
                        let after_p = &param_rem[p_start + "<parameter=".len()..];
                        if let Some(p_name_end) = after_p.find('>') {
                            let param_name = after_p[..p_name_end].to_string();
                            let value_text = &after_p[p_name_end + 1..];
                            if let Some(p_end) = value_text.find("</parameter>") {
                                let value = value_text[..p_end]
                                    .trim_start_matches('\n')
                                    .trim_end_matches('\n');
                                // Try JSON parse (objects, arrays, numbers, bools)
                                let json_value = serde_json::from_str(value)
                                    .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
                                args.insert(param_name, json_value);
                                param_rem = &value_text[p_end + "</parameter>".len()..];
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }

                    // Generate unique call ID
                    let id = format!("call_{:016x}", {
                        let mut s = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64;
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        s ^= call_idx;
                        s
                    });
                    call_idx += 1;

                    tool_calls.push(ParsedToolCall {
                        id,
                        name,
                        arguments_json: serde_json::Value::Object(args).to_string(),
                    });
                }
            }
        } else {
            // No closing tag — treat as content
            content.push_str("<tool_call>");
            content.push_str(remaining);
            remaining = "";
        }
    }

    content.push_str(remaining);
    (content.trim().to_string(), tool_calls)
}

/// Escape a string for embedding inside a JSON string value.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Format SSE chunk: tool call start (name + empty args).
fn format_sse_tool_call_start(
    request_id: &str,
    model_name: &str,
    call_index: usize,
    call_id: &str,
    function_name: &str,
    created: u64,
) -> String {
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"id":"{}","type":"function","function":{{"name":"{}","arguments":""}}}}]}},"finish_reason":null}}]}}"#,
        request_id, created, model_name, call_index, call_id, function_name
    )
}

/// Format SSE chunk: tool call arguments fragment.
fn format_sse_tool_call_args(
    request_id: &str,
    model_name: &str,
    call_index: usize,
    arguments_json: &str,
    created: u64,
) -> String {
    let escaped = json_escape(arguments_json);
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":{},"function":{{"arguments":"{}"}}}}]}},"finish_reason":null}}]}}"#,
        request_id, created, model_name, call_index, escaped
    )
}

/// Format non-streaming response with tool calls.
fn format_completion_with_tool_calls(
    request_id: &str,
    model_name: &str,
    content: &str,
    tool_calls: &[ParsedToolCall],
    prompt_tokens: usize,
    completion_tokens: usize,
    created: u64,
) -> String {
    let mut tc_parts = Vec::new();
    for tc in tool_calls {
        let escaped_args = json_escape(&tc.arguments_json);
        tc_parts.push(format!(
            r#"{{"id":"{}","type":"function","function":{{"name":"{}","arguments":"{}"}}}}"#,
            tc.id, tc.name, escaped_args
        ));
    }
    let content_field = if content.is_empty() {
        "null".to_string()
    } else {
        format!(r#""{}""#, json_escape(content))
    };
    format!(
        r#"{{"id":"{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":{},"tool_calls":[{}]}},"finish_reason":"tool_calls"}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
        request_id, created, model_name, content_field, tc_parts.join(","),
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    )
}

/// Handle a single HTTP request.
fn handle_request(
    mut tcp_stream: TcpStream,
    state: &ServerState,
) {
    let cloned = match tcp_stream.try_clone() {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to clone TCP stream: {}", e);
            return;
        }
    };
    let mut reader = BufReader::new(cloned);

    let request = match parse_request(&mut reader) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Failed to parse request: {}", e);
            let _ = send_json(&mut tcp_stream, 400, r#"{"error":"Bad request"}"#);
            return;
        }
    };

    // Handle CORS preflight
    if request.method == "OPTIONS" {
        let _ = write!(
            tcp_stream,
            "HTTP/1.1 204 No Content\r\n\
             Access-Control-Allow-Origin: *\r\n\
             Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
             Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
             Connection: close\r\n\r\n"
        );
        return;
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let body = format!(
                r#"{{"status":"ok","max_context_tokens":{}}}"#,
                state.max_context_tokens
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("GET", "/v1/models") => {
            let body = format!(
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","owned_by":"krasis","max_context_tokens":{}}}]}}"#,
                state.model_name, state.max_context_tokens
            );
            let _ = send_json(&mut tcp_stream, 200, &body);
        }

        ("POST", "/v1/chat/completions") => {
            handle_chat_completion(&mut tcp_stream, &request.body, state);
        }

        _ => {
            let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Not found"}"#);
        }
    }
}

/// Overhead timings collected during request setup (before decode).
struct RequestOverhead {
    parse_ms: f64,       // HTTP parse + JSON parse + tokenization
    evict_ms: f64,       // HCS soft-tier eviction
    prefill_ms: f64,     // GIL acquire + Python prefill
    reload_ms: f64,      // HCS soft-tier reload
}

/// Handle /v1/chat/completions request.
fn handle_chat_completion(
    stream: &mut TcpStream,
    body: &str,
    state: &ServerState,
) {
    let t_request = Instant::now();

    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(
                stream,
                400,
                &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e),
            );
            return;
        }
    };

    // Log full request body if request logging is enabled (for IDE debugging)
    if let Some(ref dir) = state.log_requests_dir {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let filename = format!("{}/{}.json", dir, ts.as_millis());
        if let Ok(pretty) = serde_json::to_string_pretty(&req) {
            std::fs::write(&filename, &pretty).ok();
        } else {
            std::fs::write(&filename, body).ok();
        }
    }

    let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(8192) as usize;
    let temperature = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.6) as f32;
    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32;
    let presence_penalty = req.get("presence_penalty").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
    let enable_thinking = req.get("enable_thinking").and_then(|v| v.as_bool()).unwrap_or(state.default_enable_thinking);

    let request_id = format!("chatcmpl-{:016x}", {
        let mut s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        s
    });
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Extract messages JSON for Python
    let messages_json = match req.get("messages") {
        Some(m) => m.to_string(),
        None => {
            let _ = send_json(stream, 400, r#"{"error":"Missing messages"}"#);
            return;
        }
    };

    // Custom stop tokens
    let stop_tokens: Vec<String> = match req.get("stop") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => {
            arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        }
        _ => vec![],
    };

    // Tool use: extract tools array and tool_choice
    // tool_choice can be a string ("auto", "none", "required") or an object
    // {"type": "function", "function": {"name": "..."}} — we pass tools through
    // unless tool_choice is explicitly "none".
    let tools_json = match req.get("tools") {
        Some(t) if t.is_array() => {
            let is_none = match req.get("tool_choice") {
                Some(serde_json::Value::String(s)) => s == "none",
                _ => false,  // object form or missing = allow tools
            };
            if is_none {
                String::new()
            } else {
                t.to_string()
            }
        }
        _ => String::new(),
    };
    let has_tools = !tools_json.is_empty();

    // ── Estimate prompt tokens for soft-tier HCS eviction ──
    let estimated_tokens = {
        // Apply actual chat template (with tools) to get full rendered text, then tokenize
        let rendered = match state.chat_template.apply_with_tools(&messages_json, &tools_json, true) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Chat template failed: {}", e);
                let _ = send_json(
                    stream,
                    500,
                    &format!(r#"{{"error":"Chat template failed: {}. This indicates a broken model setup."}}"#, e),
                );
                return;
            }
        };
        let token_count = match state.tokenizer.encode(rendered.as_str(), false) {
            Ok(e) => e.len(),
            Err(e) => {
                log::error!("Tokenizer failed to encode prompt: {}", e);
                let _ = send_json(
                    stream,
                    500,
                    &format!(r#"{{"error":"Tokenizer failed: {}. This indicates a broken model setup."}}"#, e),
                );
                return;
            }
        };
        log::info!("Soft HCS: estimated {} tokens (rendered_len={})", token_count, rendered.len());
        token_count
    };
    let parse_ms = t_request.elapsed().as_secs_f64() * 1000.0;

    // ── Evict soft HCS before prefill to free VRAM ──
    crate::vram_monitor::report_event("evict_start");
    let t_evict = Instant::now();
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (evicted, _freed_mb) = store_for_evict.hcs_evict_for_prefill(estimated_tokens);
    // NOTE: aux GPU never does prefill, so no eviction needed there
    let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("evict_end");

    // ── Snapshot VRAM before prefill ──
    log::info!("VRAM before prefill: {} MB free", store_for_evict.query_vram_free_mb());

    // ── Call Python for prefill (GIL required) ──
    crate::vram_monitor::report_event("prefill_start");
    let t_prefill_gil = Instant::now();
    let prefill_result = Python::with_gil(|py| -> PyResult<(usize, usize, Vec<usize>, bool)> {
        let result = state.py_model.call_method(
            py,
            "server_prefill",
            (
                &messages_json,
                max_tokens,
                temperature,
                top_k,
                top_p,
                presence_penalty,
                enable_thinking,
                stop_tokens.clone(),
                "gpu",
                &tools_json,
            ),
            None,
        )?;
        let first_token: usize = result.getattr(py, "first_token")?.extract(py)?;
        let prompt_len: usize = result.getattr(py, "prompt_len")?.extract(py)?;
        let stop_ids: Vec<usize> = result.getattr(py, "stop_ids")?.extract(py)?;
        let kv_overflow: bool = result.getattr(py, "kv_overflow")
            .and_then(|v| v.extract(py))
            .unwrap_or(false);
        Ok((first_token, prompt_len, stop_ids, kv_overflow))
    });
    let prefill_gil_ms = t_prefill_gil.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("prefill_end");

    let (first_token, prompt_len, stop_ids, kv_overflow) = match prefill_result {
        Ok(v) => v,
        Err(e) => {
            let err_str = e.to_string();
            log::error!("Prefill failed: {}", err_str);
            // Return 413 with structured error for KV cache exhaustion
            let (status, body) = if err_str.contains("KV cache exhausted") {
                (413, format!(
                    r#"{{"error":{{"message":"Context length exceeds KV cache capacity ({} tokens max). Reduce context or start a new conversation.","type":"invalid_request_error","code":"context_length_exceeded","max_context_tokens":{}}}}}"#,
                    state.max_context_tokens, state.max_context_tokens
                ))
            } else {
                (500, format!(r#"{{"error":{{"message":"Prefill failed: {}","type":"server_error"}}}}"#, err_str))
            };
            let _ = send_json(stream, status, &body);
            // Cleanup on error
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

    // If prompt exceeded Rust KV cache, return error (not a silent 200 with truncated output)
    if kv_overflow {
        log::error!("Request {}: prompt {} tokens exceeds Rust KV cache capacity",
            request_id, prompt_len);
        let _ = send_json(
            stream,
            507,
            &format!(
                r#"{{"error":{{"message":"Prompt ({} tokens) exceeds KV cache capacity. Increase CFG_KV_CACHE_MB or reduce prompt length.","type":"insufficient_storage","code":"kv_cache_overflow","prompt_tokens":{}}}}}"#,
                prompt_len, prompt_len,
            ),
        );
        Python::with_gil(|py| {
            let _ = state.py_model.call_method0(py, "server_cleanup");
        });
        return;
    }

    // Check context length
    if prompt_len >= state.max_context_tokens {
        let _ = send_json(
            stream,
            413,
            &format!(
                r#"{{"error":{{"message":"Prompt too long: {} tokens exceeds KV cache capacity of {} tokens","type":"invalid_request_error","code":"context_length_exceeded","prompt_tokens":{},"max_context_tokens":{}}}}}"#,
                prompt_len, state.max_context_tokens, prompt_len, state.max_context_tokens
            ),
        );
        Python::with_gil(|py| {
            let _ = state.py_model.call_method0(py, "server_cleanup");
        });
        return;
    }

    log::info!(
        "Request {}: {} prompt tokens, max_new={}, stream={}, decode=gpu",
        request_id, prompt_len, max_tokens, is_stream
    );

    let tokenizer = &state.tokenizer;

    // ── Reload soft HCS after prefill (async — DMA overlaps with decode) ──
    // Always attempt reload — soft pool may have been cancelled by a prior operation
    // even if we didn't evict anything this time.
    crate::vram_monitor::report_event("reload_start");
    let t_reload = Instant::now();
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    {
        let (queued, _alloc_mb) = store.hcs_reload_after_prefill_async(prompt_len);
        if queued > 0 {
            log::info!("Request {}: HCS soft async reload queued {} experts ({} tokens)",
                request_id, queued, prompt_len);
        }
    }
    // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
    // ── Multi-GPU: copy KV cache from primary to all aux GPUs after prefill ──
    if !state.aux_gpu_store_addrs.is_empty() {
        let t_kvcopy = Instant::now();
        let num_aux = state.aux_gpu_store_addrs.len();
        let num_layers = store.num_layers();
        for i in 0..num_aux {
            let aux_store = unsafe { &mut *(state.aux_gpu_store_addrs[i] as *mut GpuDecodeStore) };
            let layer_start = state.multi_gpu_split_layers[i];
            let layer_end = if i + 1 < num_aux { state.multi_gpu_split_layers[i + 1] } else { num_layers };
            if let Err(e) = store.copy_kv_to_aux(aux_store, layer_start, layer_end, state.multi_gpu_gqa_offsets[i], prompt_len) {
                log::error!("Request {}: KV cache copy to aux GPU{} failed: {}", request_id, i + 1, e);
            }
        }
        let kvcopy_ms = t_kvcopy.elapsed().as_secs_f64() * 1000.0;
        log::info!("Request {}: KV cache copied to {} aux GPUs in {:.1}ms", request_id, num_aux, kvcopy_ms);
    }
    let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;

    let overhead = RequestOverhead {
        parse_ms,
        evict_ms,
        prefill_ms: prefill_gil_ms,
        reload_ms,
    };

    // ── Thinking suppression: prevent EOS before </think> ──
    // When thinking is enabled, the model must generate </think> before it can
    // terminate with <|im_end|>. Without this, the model puts its answer inside
    // the thinking block and bails to EOS, resulting in 0 visible answer tokens.
    if enable_thinking {
        if let Some(te_id) = state.thinking_end_token {
            // Budget = max 4096 thinking tokens. If the model hasn't produced </think>
            // by then, it's stuck in a loop. 4096 is generous for real reasoning.
            let think_budget = 4096;
            store.set_think_end_suppress(Some(te_id), think_budget);
            store.set_min_new_tokens_ext(0, stop_ids.to_vec());
        } else {
            store.set_think_end_suppress(None, 0);
            store.set_min_new_tokens_ext(0, vec![]);
        }
    } else {
        store.set_think_end_suppress(None, 0);
        store.set_min_new_tokens_ext(0, vec![]);
    }

    // ── GPU decode: GIL-free Rust decode via GpuDecodeStore ──
    crate::vram_monitor::report_event("decode_start");
    handle_gpu_decode(
        stream, is_stream, state, store, tokenizer,
        first_token, prompt_len, max_tokens, temperature,
        top_k, top_p, presence_penalty, &stop_ids,
        &request_id, &state.model_name, created,
        &overhead, has_tools, enable_thinking,
    );
    crate::vram_monitor::report_event("decode_end");

    // ── Cleanup (GIL required) ──
    let t_cleanup_gil = Instant::now();
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    let cleanup_gil_ms = t_cleanup_gil.elapsed().as_secs_f64() * 1000.0;
    crate::vram_monitor::report_event("cleanup_end");

    let total_ms = t_request.elapsed().as_secs_f64() * 1000.0;
    log::info!(
        "Request {} complete: total={:.0}ms | parse={:.1}ms evict={:.1}ms prefill={:.0}ms reload={:.0}ms cleanup={:.1}ms",
        request_id, total_ms, parse_ms, evict_ms, prefill_gil_ms, reload_ms, cleanup_gil_ms
    );
}

/// GPU decode: GIL-free Rust decode loop via GpuDecodeStore.
/// Pure Rust, zero Python per token.
#[allow(clippy::too_many_arguments)]
fn handle_gpu_decode(
    stream: &mut TcpStream,
    is_stream: bool,
    state: &ServerState,
    store: &mut GpuDecodeStore,
    tokenizer: &tokenizers::Tokenizer,
    first_token: usize,
    prompt_len: usize,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    presence_penalty: f32,
    stop_ids: &[usize],
    request_id: &str,
    model_name: &str,
    created: u64,
    overhead: &RequestOverhead,
    has_tools: bool,
    enable_thinking: bool,
) {
    // Resolve thinking end token early — used by both streaming and non-streaming paths
    let think_end_id = if enable_thinking { state.thinking_end_token } else { None };

    if is_stream {
        if let Err(e) = begin_sse(stream) {
            log::error!("Failed to send SSE headers: {}", e);
            return;
        }

        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();

        // When thinking is enabled, inject <think> at start of stream.
        // The prompt already includes <think>, but the client needs it in the
        // output to know this is a thinking block (for display suppression).
        if think_end_id.is_some() {
            let think_chunk = format_sse_token(request_id, model_name, "<think>", None, created);
            let _ = send_sse_chunk(stream, &think_chunk);
        }

        // When tool use is active, buffer first token (might need tool call parsing).
        // Otherwise send immediately for lowest latency.
        if !has_tools {
            let chunk = format_sse_token(request_id, model_name, &first_text, None, created);
            let _ = send_sse_chunk(stream, &chunk);
        }

        let (tx, rx) = mpsc::channel::<String>();
        let writer_disconnected = Arc::new(AtomicBool::new(false));
        let writer_disc_clone = writer_disconnected.clone();

        let mut writer_stream = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to clone stream for writer: {}", e);
                return;
            }
        };

        let writer_handle = std::thread::spawn(move || {
            let flush_interval = std::time::Duration::from_millis(100);
            let mut buf = String::new();
            let mut last_flush = Instant::now();
            let mut is_first = true;
            loop {
                match rx.recv_timeout(flush_interval) {
                    Ok(chunk) => {
                        buf.push_str(&chunk);
                        if is_first || last_flush.elapsed() >= flush_interval || buf.len() > 8192 {
                            if writer_stream.write_all(buf.as_bytes()).is_err()
                                || writer_stream.flush().is_err()
                            {
                                writer_disc_clone.store(true, Ordering::Release);
                                return;
                            }
                            buf.clear();
                            last_flush = Instant::now();
                            is_first = false;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if !buf.is_empty() {
                            if writer_stream.write_all(buf.as_bytes()).is_err()
                                || writer_stream.flush().is_err()
                            {
                                writer_disc_clone.store(true, Ordering::Release);
                                return;
                            }
                            buf.clear();
                            last_flush = Instant::now();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        if !buf.is_empty() {
                            let _ = writer_stream.write_all(buf.as_bytes());
                            let _ = writer_stream.flush();
                        }
                        return;
                    }
                }
            }
        });

        let decode_start = Instant::now();
        let mut decode_token_count = 0usize;

        // ── Thinking budget tracking ──
        // When thinking is enabled, tokens inside <think>...</think> are exempt
        // from max_tokens. We track the state and only count answer tokens.
        let mut in_thinking = think_end_id.is_some(); // start in thinking if enabled
        let mut answer_token_count = 0usize;
        let mut thinking_token_count = 0usize;
        // Also check first_token — it could be </think> for trivial thinking
        if in_thinking && Some(first_token) == think_end_id {
            in_thinking = false;
        } else if in_thinking {
            thinking_token_count += 1;
        }

        // ── Tool call detection state ──
        // When tools are present: stream content normally, detect <tool_call>,
        // buffer everything from that point, then send structured tool_calls
        // at the end.  Content before tool calls streams with full latency.
        let mut tc_all_text = String::new();
        let mut tc_in_tool_call = false;
        let mut tc_found = false;
        let mut tc_finish = String::new();

        if has_tools {
            tc_all_text.push_str(&first_text);
            // Send first token if it's safe (doesn't contain tool call marker)
            if first_text.contains("<tool_call>") {
                tc_in_tool_call = true;
                tc_found = true;
                // Send content before the marker
                if let Some(idx) = first_text.find("<tool_call>") {
                    let before = &first_text[..idx];
                    if !before.is_empty() {
                        let chunk = format_sse_token(request_id, model_name, before, None, created);
                        let _ = tx.send(format!("data: {}\n\n", chunk));
                    }
                }
            } else if !first_text.is_empty() {
                let chunk = format_sse_token(request_id, model_name, &first_text, None, created);
                let _ = tx.send(format!("data: {}\n\n", chunk));
            }
        }

        // Shared callback for both single-GPU and multi-GPU decode
        let mut on_token = |token_id: usize, text: &str, finish_reason: Option<&str>| -> bool {
            decode_token_count += 1;

            // ── Track thinking state ──
            // Tokens before </think> are "thinking" and don't count against max_tokens.
            if think_end_id.is_some() {
                if in_thinking {
                    thinking_token_count += 1;
                    if Some(token_id) == think_end_id {
                        in_thinking = false;
                        log::info!("Thinking complete: {} tokens", thinking_token_count);
                    }
                } else {
                    answer_token_count += 1;
                }
            }

            // Override finish_reason if answer token limit reached
            let effective_finish = if finish_reason.is_some() {
                finish_reason
            } else if think_end_id.is_some() && !in_thinking && answer_token_count >= max_tokens {
                Some("length")
            } else {
                None
            };

            if has_tools {
                tc_all_text.push_str(text);
                if let Some(fr) = effective_finish {
                    tc_finish = fr.to_string();
                }

                if tc_in_tool_call {
                    // Inside a tool call block — buffer silently
                } else if text.contains("<tool_call>") {
                    // Entering tool call territory
                    tc_in_tool_call = true;
                    tc_found = true;
                    // Send any content before the marker in this text
                    if let Some(idx) = text.find("<tool_call>") {
                        let before = &text[..idx];
                        if !before.is_empty() {
                            let chunk = format_sse_token(
                                request_id, model_name, before, None, created,
                            );
                            let _ = tx.send(format!("data: {}\n\n", chunk));
                        }
                    }
                } else {
                    // Normal content — stream it (no finish_reason; handled post-generation)
                    if !text.is_empty() {
                        let chunk = format_sse_token(
                            request_id, model_name, text, None, created,
                        );
                        let _ = tx.send(format!("data: {}\n\n", chunk));
                    }
                }

                if writer_disconnected.load(Ordering::Acquire) {
                    return false;
                }
                if effective_finish.is_some() {
                    return false;
                }
                true
            } else {
                // Original non-tool path
                let chunk = format_sse_token(
                    request_id, model_name, text, effective_finish, created,
                );
                let formatted = format!("data: {}\n\n", chunk);
                if tx.send(formatted).is_err()
                    || writer_disconnected.load(Ordering::Acquire)
                {
                    return false;
                }
                // Stop if answer limit reached
                if effective_finish.is_some() {
                    return false;
                }
                true
            }
        };

        // When thinking is enabled, give the decode loop extra budget for thinking tokens.
        // The on_token callback enforces the real max_tokens on answer tokens only.
        let decode_budget = if think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };

        if !state.aux_gpu_store_addrs.is_empty() {
            // Multi-GPU decode: pipeline across N GPUs
            store.gpu_generate_stream_multi(
                &state.aux_gpu_store_addrs,
                &state.multi_gpu_split_layers,
                &state.multi_gpu_gqa_offsets,
                first_token,
                prompt_len,
                decode_budget,
                temperature,
                top_k,
                top_p,
                stop_ids,
                tokenizer,
                presence_penalty,
                &mut on_token,
            );
        } else {
            // Single-GPU decode
            store.gpu_generate_stream(
                first_token,
                prompt_len,
                decode_budget,
                temperature,
                top_k,
                top_p,
                stop_ids,
                tokenizer,
                presence_penalty,
                on_token,
            );
        }

        // Capture decode timing BEFORE post-generation processing (tool call parsing etc.)
        let decode_elapsed = decode_start.elapsed().as_secs_f64();

        // ── Post-generation: emit tool calls or finish ──
        if has_tools {
            let (_content, tool_calls) = parse_tool_calls(&tc_all_text);
            if !tool_calls.is_empty() {
                // Content before tool calls was already streamed in the callback.
                // Now send the structured tool_call chunks.
                for (i, tc) in tool_calls.iter().enumerate() {
                    let start_chunk = format_sse_tool_call_start(
                        request_id, model_name, i, &tc.id, &tc.name, created,
                    );
                    let _ = tx.send(format!("data: {}\n\n", start_chunk));
                    let args_chunk = format_sse_tool_call_args(
                        request_id, model_name, i, &tc.arguments_json, created,
                    );
                    let _ = tx.send(format!("data: {}\n\n", args_chunk));
                }
                let finish_chunk = format_sse_token(
                    request_id, model_name, "", Some("tool_calls"), created,
                );
                let _ = tx.send(format!("data: {}\n\n", finish_chunk));
                log::info!(
                    "Request {}: {} tool call(s) detected",
                    request_id, tool_calls.len()
                );
            } else {
                // No tool calls — send finish with original reason
                let fr = if tc_finish.is_empty() { "stop" } else { &tc_finish };
                let finish_chunk = format_sse_token(
                    request_id, model_name, "", Some(fr), created,
                );
                let _ = tx.send(format!("data: {}\n\n", finish_chunk));
            }
        }

        let elapsed = decode_elapsed;
        let total_gen = decode_token_count + 1;
        let decode_tok_s = if elapsed > 0.0 && decode_token_count > 0 {
            decode_token_count as f64 / elapsed
        } else {
            0.0
        };
        let decode_ms = elapsed * 1000.0;
        let prefill_tok_s = if overhead.prefill_ms > 0.0 && prompt_len > 0 {
            prompt_len as f64 / (overhead.prefill_ms / 1000.0)
        } else {
            0.0
        };
        let overhead_total_ms = overhead.parse_ms + overhead.evict_ms + overhead.prefill_ms + overhead.reload_ms;
        let timing_chunk = format!(
            r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[],"krasis_timing":{{"decode_tokens":{},"decode_time_ms":{:.1},"decode_tok_s":{:.2},"thinking_tokens":{},"answer_tokens":{},"total_generated":{},"prompt_tokens":{},"prefill_tok_s":{:.1},"overhead_ms":{:.1},"overhead":{{"parse_ms":{:.1},"evict_ms":{:.1},"prefill_ms":{:.1},"reload_ms":{:.1}}}}}}}"#,
            request_id, created, model_name,
            decode_token_count, decode_ms, decode_tok_s,
            thinking_token_count, answer_token_count,
            total_gen, prompt_len, prefill_tok_s,
            overhead_total_ms,
            overhead.parse_ms, overhead.evict_ms, overhead.prefill_ms, overhead.reload_ms
        );
        let _ = tx.send(format!("data: {}\n\n", timing_chunk));
        let _ = tx.send("data: [DONE]\n\n".to_string());
        drop(tx);
        let _ = writer_handle.join();

        log::info!(
            "Request {} complete: decode={:.2}s ({} tok, {:.1} tok/s) | overhead={:.0}ms (parse={:.1} evict={:.1} prefill={:.0} reload={:.0})",
            request_id, elapsed, total_gen, decode_tok_s,
            overhead_total_ms, overhead.parse_ms, overhead.evict_ms, overhead.prefill_ms, overhead.reload_ms
        );
    } else {
        // ── Non-streaming path ──
        let mut all_text = String::new();
        // Inject <think> prefix so clients can identify thinking blocks
        if enable_thinking && state.thinking_end_token.is_some() {
            all_text.push_str("<think>");
        }
        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        all_text.push_str(&first_text);
        let mut total_tokens = 1usize;
        let mut finish = "length".to_string();

        // Thinking budget for non-streaming
        let ns_think_end_id = if enable_thinking { state.thinking_end_token } else { None };
        let mut ns_in_thinking = ns_think_end_id.is_some();
        let mut ns_answer_tokens = 0usize;
        if ns_in_thinking && Some(first_token) == ns_think_end_id {
            ns_in_thinking = false;
        }

        let ns_decode_budget = if ns_think_end_id.is_some() {
            max_tokens.saturating_add(32768).saturating_sub(1)
        } else {
            max_tokens.saturating_sub(1)
        };

        {
            let mut on_token = |token_id: usize, text: &str, finish_reason: Option<&str>| -> bool {
                all_text.push_str(text);
                total_tokens += 1;

                // Track thinking state
                if ns_think_end_id.is_some() {
                    if ns_in_thinking {
                        if Some(token_id) == ns_think_end_id {
                            ns_in_thinking = false;
                        }
                    } else {
                        ns_answer_tokens += 1;
                    }
                }

                if let Some(fr) = finish_reason {
                    finish = fr.to_string();
                }

                // Stop if answer limit reached
                if ns_think_end_id.is_some() && !ns_in_thinking && ns_answer_tokens >= max_tokens {
                    finish = "length".to_string();
                    return false;
                }

                true
            };
            if !state.aux_gpu_store_addrs.is_empty() {
                store.gpu_generate_stream_multi(
                    &state.aux_gpu_store_addrs,
                    &state.multi_gpu_split_layers,
                    &state.multi_gpu_gqa_offsets,
                    first_token,
                    prompt_len,
                    ns_decode_budget,
                    temperature,
                    top_k,
                    top_p,
                    stop_ids,
                    tokenizer,
                    presence_penalty,
                    &mut on_token,
                );
            } else {
                store.gpu_generate_stream(
                    first_token,
                    prompt_len,
                    ns_decode_budget,
                    temperature,
                    top_k,
                    top_p,
                    stop_ids,
                    tokenizer,
                    presence_penalty,
                    on_token,
                );
            }
        }

        if has_tools {
            let (content, tool_calls) = parse_tool_calls(&all_text);
            if !tool_calls.is_empty() {
                let response = format_completion_with_tool_calls(
                    request_id, model_name, &content, &tool_calls,
                    prompt_len, total_tokens, created,
                );
                let _ = send_json(stream, 200, &response);
                log::info!(
                    "Request {}: {} tool call(s) (non-streaming)",
                    request_id, tool_calls.len()
                );
            } else {
                let response = format_completion(
                    request_id, model_name, &all_text, prompt_len,
                    total_tokens, &finish, created,
                );
                let _ = send_json(stream, 200, &response);
            }
        } else {
            let response = format_completion(
                request_id, model_name, &all_text, prompt_len,
                total_tokens, &finish, created,
            );
            let _ = send_json(stream, 200, &response);
        }
    }
}

/// The Rust HTTP server, exposed to Python via PyO3.
#[pyclass]
pub struct RustServer {
    host: String,
    port: u16,
    model_name: String,
    tokenizer_path: String,
    max_context_tokens: usize,
    default_enable_thinking: bool,
    /// Token ID for `</think>` passed from Python (0 = not available).
    thinking_end_token_id: usize,
    gpu_store_addr: usize,
    py_model: Py<PyAny>,
    running: Arc<AtomicBool>,
    aux_gpu_store_addrs: Vec<usize>,
    multi_gpu_split_layers: Vec<usize>,
    multi_gpu_gqa_offsets: Vec<usize>,
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens, enable_thinking=true, thinking_end_token_id=0, gpu_store_addr=0, aux_gpu_store_addrs=Vec::new(), multi_gpu_split_layers=Vec::new(), multi_gpu_gqa_offsets=Vec::new()))]
    fn new(
        py_model: PyObject,
        host: String,
        port: u16,
        model_name: String,
        tokenizer_path: String,
        max_context_tokens: usize,
        enable_thinking: bool,
        thinking_end_token_id: usize,
        gpu_store_addr: usize,
        aux_gpu_store_addrs: Vec<usize>,
        multi_gpu_split_layers: Vec<usize>,
        multi_gpu_gqa_offsets: Vec<usize>,
    ) -> Self {
        Self {
            host,
            port,
            model_name,
            tokenizer_path,
            max_context_tokens,
            default_enable_thinking: enable_thinking,
            thinking_end_token_id,
            gpu_store_addr,
            py_model: py_model.into(),
            running: Arc::new(AtomicBool::new(false)),
            aux_gpu_store_addrs,
            multi_gpu_split_layers,
            multi_gpu_gqa_offsets,
        }
    }

    /// Start the HTTP server. Blocks until stop() is called.
    /// Releases the GIL so Python remains responsive for prefill calls.
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        self.running.store(true, Ordering::Release);

        let addr = format!("{}:{}", self.host, self.port);
        let py_model = self.py_model.clone_ref(py);
        let model_name = self.model_name.clone();
        let tokenizer_path = self.tokenizer_path.clone();
        let max_context_tokens = self.max_context_tokens;
        let default_enable_thinking = self.default_enable_thinking;
        let thinking_end_token_id = self.thinking_end_token_id;
        let gpu_store_addr = self.gpu_store_addr;
        let aux_gpu_store_addrs = self.aux_gpu_store_addrs.clone();
        let multi_gpu_split_layers = self.multi_gpu_split_layers.clone();
        let multi_gpu_gqa_offsets = self.multi_gpu_gqa_offsets.clone();
        let running = self.running.clone();

        // Install raw SIGINT + SIGTERM handlers BEFORE releasing the GIL.
        // Python's signal.signal handlers only dispatch between bytecodes,
        // but run() enters allow_threads (native Rust) so Python never gets
        // a chance to run the handler.  The raw handler sets `running` to
        // false directly, and the accept loop exits on the next 10ms poll.
        // SIGTERM is needed because the release test (and systemd) send
        // SIGTERM for clean shutdown; without a raw handler, the server
        // never stops and gets SIGKILL'd, skipping VRAM report CSV write.
        let running_ptr = Arc::as_ptr(&self.running) as *mut AtomicBool;
        SIGNAL_FLAG_PTR.store(running_ptr, Ordering::Release);

        // Save previous handlers so we can restore them
        let prev_sigint;
        let prev_sigterm;
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = shutdown_signal_handler as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = libc::SA_RESTART;

            let mut old_int: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGINT, &sa, &mut old_int);
            prev_sigint = old_int;

            let mut old_term: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGTERM, &sa, &mut old_term);
            prev_sigterm = old_term;
        }

        // Release GIL — server loop runs without it.
        // GIL is reacquired inside handle_request only for Python prefill/cleanup.
        py.allow_threads(move || {
            // Load tokenizer once at startup (not per-request)
            let tokenizer = match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Failed to load tokenizer: {}", e);
                    return;
                }
            };

            // Load chat template from tokenizer_config.json (same directory as tokenizer.json)
            let tokenizer_config_path = {
                let p = std::path::Path::new(&tokenizer_path);
                p.parent().unwrap_or(p).join("tokenizer_config.json")
            };
            let chat_template = match crate::chat_template::ChatTemplateEngine::from_config(
                tokenizer_config_path.to_str().unwrap_or("")
            ) {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Failed to load chat template: {}", e);
                    return;
                }
            };

            let listener = match TcpListener::bind(&addr) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("Failed to bind {}: {}", addr, e);
                    return;
                }
            };

            // Set non-blocking so we can check the running flag
            listener
                .set_nonblocking(true)
                .expect("Cannot set non-blocking");

            log::info!("Rust HTTP server listening on {}", addr);

            let gil_timing = std::env::var("KRASIS_GIL_TIMING")
                .map(|v| v == "1")
                .unwrap_or(false);
            if gil_timing {
                log::info!("GIL timing enabled (KRASIS_GIL_TIMING=1)");
            }

            let log_requests_dir = if std::env::var("KRASIS_LOG_REQUESTS").map(|v| v == "1").unwrap_or(false) {
                let dir = "logs/requests".to_string();
                std::fs::create_dir_all(&dir).ok();
                log::info!("Request logging enabled → {}/", dir);
                Some(dir)
            } else {
                None
            };

            // </think> token ID passed from Python (0 = not available)
            let thinking_end_token = if thinking_end_token_id > 0 {
                log::info!("Thinking end token: </think> = {}", thinking_end_token_id);
                Some(thinking_end_token_id)
            } else {
                None
            };

            let state = ServerState {
                py_model,
                model_name,
                tokenizer,
                chat_template,
                max_context_tokens,
                default_enable_thinking,
                thinking_end_token,
                gpu_store_addr,
                gil_timing,
                log_requests_dir,
                aux_gpu_store_addrs,
                multi_gpu_split_layers,
                multi_gpu_gqa_offsets,
            };

            while running.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _addr)) => {
                        // Set blocking for the actual request handling
                        stream.set_nonblocking(false).ok();
                        // Disable Nagle's algorithm for immediate SSE chunk delivery
                        stream.set_nodelay(true).ok();
                        // Set read timeout to prevent hanging on malformed requests
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                            .ok();
                        handle_request(stream, &state);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No connection ready, sleep briefly and retry
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(e) => {
                        log::error!("Accept error: {}", e);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                }
            }

            log::info!("Rust HTTP server stopped");
        });

        // Restore previous signal handlers and clear global pointer
        SIGNAL_FLAG_PTR.store(std::ptr::null_mut(), Ordering::Release);
        unsafe {
            libc::sigaction(libc::SIGINT, &prev_sigint, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &prev_sigterm, std::ptr::null_mut());
        }

        Ok(())
    }

    /// Run a single benchmark request through the engine (no HTTP/SSE).
    /// Same operations as handle_chat_completion but without network I/O.
    /// Returns JSON string with engine-internal timing breakdown.
    ///
    /// Safety: assumes no concurrent HTTP requests during benchmark.
    #[pyo3(signature = (messages_json, max_new_tokens, temperature=0.6, enable_thinking=false))]
    fn benchmark_request(
        &self,
        py: Python<'_>,
        messages_json: String,
        max_new_tokens: usize,
        temperature: f32,
        enable_thinking: bool,
    ) -> PyResult<String> {
        // Load tokenizer and chat template (same as server path)
        let tokenizer = tokenizers::Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load tokenizer: {}", e)))?;
        let tokenizer_config_path = {
            let p = std::path::Path::new(&self.tokenizer_path);
            p.parent().unwrap_or(p).join("tokenizer_config.json")
        };
        let chat_template = crate::chat_template::ChatTemplateEngine::from_config(
            tokenizer_config_path.to_str().unwrap_or("")
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Failed to load chat template: {}", e)))?;

        // Estimate tokens by applying actual chat template and tokenizing
        let estimated_tokens = {
            let rendered = chat_template.apply(&messages_json, true)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Chat template failed: {}", e)))?;
            tokenizer.encode(rendered.as_str(), false)
                .map(|e| e.len())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Tokenizer failed: {}", e)))?
        };

        // Evict soft HCS before prefill (both stores in multi-GPU)
        let store = unsafe { &mut *(self.gpu_store_addr as *mut GpuDecodeStore) };
        let t_evict = Instant::now();
        let (evicted, _) = store.hcs_evict_for_prefill(estimated_tokens);
        // NOTE: aux GPU never does prefill, so no eviction needed there
        let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;

        // Prefill (Python, GIL held)
        crate::vram_monitor::report_event("prefill_start");
        let t_prefill = Instant::now();
        let stop_tokens: Vec<String> = Vec::new();
        let result = self.py_model.call_method(
            py,
            "server_prefill",
            (
                &messages_json,
                max_new_tokens,
                temperature,
                50usize,        // top_k
                0.95f32,        // top_p
                0.0f32,         // presence_penalty
                enable_thinking,
                stop_tokens,
                "gpu",
            ),
            None,
        )?;
        let first_token: usize = result.getattr(py, "first_token")?.extract(py)?;
        let prompt_len: usize = result.getattr(py, "prompt_len")?.extract(py)?;
        let stop_ids: Vec<usize> = result.getattr(py, "stop_ids")?.extract(py)?;
        let kv_overflow: bool = result.getattr(py, "kv_overflow")
            .and_then(|v| v.extract(py))
            .unwrap_or(false);
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("prefill_end");

        // Reload soft HCS after prefill — ASYNC, matching production behavior.
        // Production starts async reload BEFORE KV copy so DMA overlaps with
        // KV copy time, giving soft experts a head start. We match that order.
        crate::vram_monitor::report_event("hcs_soft_load_start");
        let t_reload = Instant::now();
        let (queued, _alloc_mb) = store.hcs_reload_after_prefill_async(prompt_len);
        if queued > 0 {
            log::info!("Benchmark: HCS soft async reload queued {} experts ({} tokens)",
                queued, prompt_len);
        }
        // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
        let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("hcs_soft_load_end");

        // Copy KV cache to aux stores (multi-GPU) — after async reload starts
        if !self.aux_gpu_store_addrs.is_empty() {
            let num_aux = self.aux_gpu_store_addrs.len();
            let num_layers = store.num_layers();
            for i in 0..num_aux {
                let aux_store = unsafe { &mut *(self.aux_gpu_store_addrs[i] as *mut GpuDecodeStore) };
                let layer_start = self.multi_gpu_split_layers[i];
                let layer_end = if i + 1 < num_aux { self.multi_gpu_split_layers[i + 1] } else { num_layers };
                if let Err(e) = store.copy_kv_to_aux(aux_store, layer_start, layer_end, self.multi_gpu_gqa_offsets[i], prompt_len) {
                    log::error!("benchmark_request: KV copy to aux GPU{} failed: {}", i + 1, e);
                }
            }
        }

        // Decode (pure Rust, GIL held but unused by decode loop)
        crate::vram_monitor::report_event("decode_start");
        let (decode_ms, decode_tokens, decode_tok_s) = if kv_overflow || max_new_tokens <= 1 {
            (0.0f64, 1usize, 0.0f64)
        } else {
            let decode_start = Instant::now();
            let mut count = 0usize;
            if !self.aux_gpu_store_addrs.is_empty() {
                store.gpu_generate_stream_multi(
                    &self.aux_gpu_store_addrs,
                    &self.multi_gpu_split_layers,
                    &self.multi_gpu_gqa_offsets,
                    first_token,
                    prompt_len,
                    max_new_tokens.saturating_sub(1),
                    temperature,
                    50,     // top_k
                    0.95,   // top_p
                    &stop_ids,
                    &tokenizer,
                    0.0,    // presence_penalty
                    |_token_id: usize, _text: &str, _finish_reason: Option<&str>| {
                        count += 1;
                        true
                    },
                );
            } else {
                store.gpu_generate_stream(
                    first_token,
                    prompt_len,
                    max_new_tokens.saturating_sub(1),
                    temperature,
                    50,     // top_k
                    0.95,   // top_p
                    &stop_ids,
                    &tokenizer,
                    0.0,    // presence_penalty
                    |_token_id, _text, _finish_reason| {
                        count += 1;
                        true
                    },
                );
            }
            let elapsed = decode_start.elapsed().as_secs_f64();
            let total = count + 1; // includes first_token from prefill
            let tok_s = if elapsed > 0.0 && count > 0 {
                count as f64 / elapsed
            } else {
                0.0
            };
            (elapsed * 1000.0, total, tok_s)
        };

        crate::vram_monitor::report_event("decode_end");

        // Cleanup
        self.py_model.call_method0(py, "server_cleanup")?;

        let prefill_tok_s = if prefill_ms > 0.0 {
            prompt_len as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };

        // Collect HCS stats from primary store
        let (min_free_vram_mb, mut hcs_loaded, mut hcs_total, _) = store.benchmark_stats();
        let safety_margin_mb = store.hcs_safety_margin_mb();

        // Aggregate HCS stats from all aux stores (multi-GPU)
        // Also log per-GPU VRAM stats
        if !self.aux_gpu_store_addrs.is_empty() {
            log::info!("  GPU0: min_free={} MB, HCS {} loaded", min_free_vram_mb, hcs_loaded);
        }
        for (i, &aux_addr) in self.aux_gpu_store_addrs.iter().enumerate() {
            let aux_store = unsafe { &*(aux_addr as *const GpuDecodeStore) };
            let (aux_min_free, aux_loaded, aux_total, aux_pct) = aux_store.benchmark_stats();
            hcs_loaded += aux_loaded;
            hcs_total += aux_total;
            if !self.aux_gpu_store_addrs.is_empty() {
                log::info!("  GPU{}: min_free={} MB, HCS {}/{} ({:.1}%)",
                    i + 1, aux_min_free, aux_loaded, aux_total, aux_pct);
            }
        }
        let hcs_pct = if hcs_total > 0 { hcs_loaded as f64 / hcs_total as f64 * 100.0 } else { 0.0 };

        Ok(format!(
            r#"{{"prefill_ms":{:.1},"prefill_tok_s":{:.1},"prompt_tokens":{},"decode_ms":{:.1},"decode_tok_s":{:.2},"decode_tokens":{},"evict_ms":{:.1},"reload_ms":{:.1},"min_free_vram_mb":{},"hcs_loaded":{},"hcs_total":{},"hcs_pct":{:.1},"safety_margin_mb":{}}}"#,
            prefill_ms, prefill_tok_s, prompt_len,
            decode_ms, decode_tok_s, decode_tokens,
            evict_ms, reload_ms,
            min_free_vram_mb, hcs_loaded, hcs_total, hcs_pct,
            safety_margin_mb
        ))
    }

    /// Signal the server to stop.
    fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Check if server is running.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}
