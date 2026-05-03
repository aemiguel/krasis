//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Prefill and decode run in Rust on the production request path.
//! Python remains for startup/orchestration and model ownership.
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
    /// Shared Rust prefill engine — Arc+Mutex shared with benchmark path.
    /// When engine is available inside the Mutex, prefill runs entirely in Rust.
    rust_prefill: Arc<std::sync::Mutex<Option<crate::gpu_prefill::PrefillEngine>>>,
    /// When true, enable test-only endpoints (e.g. /v1/internal/prefill_logits).
    test_endpoints: bool,
    /// Model's EOS token IDs (from generation_config.json).
    /// These are always included in stop_ids for decode, matching the main branch behavior.
    eos_stop_ids: Vec<usize>,
    /// Monotonic order for /v1/internal/reference_test requests.
    reference_test_request_order: u64,
}

/// Parsed HTTP request.
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

fn prepare_store_for_rust_prefill(
    store: &mut GpuDecodeStore,
    engine: &mut crate::gpu_prefill::PrefillEngine,
) -> Result<bool, String> {
    let has_hqq = store.prepare_runtime_for_prefill_rust()?;
    if has_hqq {
        let patches = store.hqq_prefill_pointer_patches_rust()?;
        engine.refresh_hqq_prefill_tensor_pointers(&patches)?;
    }
    Ok(has_hqq)
}

fn create_prefill_engine_for_server(
    store: &mut GpuDecodeStore,
    max_context_tokens: usize,
) -> Result<crate::gpu_prefill::PrefillEngine, String> {
    let has_hqq = store.has_hqq_runtime_slots();
    if has_hqq {
        store.prepare_runtime_for_prefill_rust()?;
    }
    let engine = match store.create_prefill_engine(max_context_tokens) {
        Ok(engine) => engine,
        Err(e) => {
            if has_hqq {
                let _ = store.prepare_runtime_for_decode_rust();
            }
            return Err(e);
        }
    };
    if has_hqq {
        store.prepare_runtime_for_decode_rust()?;
    }
    Ok(engine)
}

fn restore_store_after_rust_prefill(
    store: &mut GpuDecodeStore,
    prompt_len: usize,
) -> Result<(), String> {
    store.set_kv_position_rust(prompt_len);
    store.prepare_runtime_for_decode_rust()
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

fn fnv1a_token_hash(token_ids: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &token in token_ids {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn reference_logit_trace_json(
    logits: &[f32],
    vocab_size: usize,
    selected_token: usize,
    top_n: usize,
) -> serde_json::Value {
    let vocab_size = vocab_size.min(logits.len());
    if vocab_size == 0 {
        return serde_json::json!({
            "available": false,
            "reason": "empty_logits",
        });
    }

    let mut finite_count = 0usize;
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    let mut max_logit = f32::NEG_INFINITY;
    let mut max_token = 0usize;
    let mut min_logit = f32::INFINITY;
    let mut min_token = 0usize;

    for (idx, &value) in logits[..vocab_size].iter().enumerate() {
        if value.is_nan() {
            nan_count += 1;
            continue;
        }
        if value == f32::INFINITY {
            pos_inf_count += 1;
        } else if value == f32::NEG_INFINITY {
            neg_inf_count += 1;
        } else {
            finite_count += 1;
        }
        if value > max_logit {
            max_logit = value;
            max_token = idx;
        }
        if value < min_logit {
            min_logit = value;
            min_token = idx;
        }
    }

    let sum_exp: f64 = logits[..vocab_size]
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum();
    let log_sum_exp = max_logit as f64 + sum_exp.ln();
    let selected_raw_logit = logits.get(selected_token).copied().unwrap_or(f32::NAN);
    let selected_logprob = selected_raw_logit as f64 - log_sum_exp;

    let mut top_logits: Vec<(usize, f32)> = Vec::with_capacity(top_n.saturating_add(1));
    for (idx, &value) in logits[..vocab_size].iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if top_logits.len() < top_n {
            top_logits.push((idx, value));
            if top_logits.len() == top_n {
                top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if top_n > 0 && value > top_logits[top_n - 1].1 {
            top_logits[top_n - 1] = (idx, value);
            top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    top_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_entries: Vec<serde_json::Value> = top_logits
        .iter()
        .enumerate()
        .map(|(rank, &(token_id, raw_logit))| {
            serde_json::json!({
                "rank": rank + 1,
                "token_id": token_id,
                "raw_logit": raw_logit as f64,
                "logprob": raw_logit as f64 - log_sum_exp,
                "softmax_prob": (raw_logit as f64 - log_sum_exp).exp(),
            })
        })
        .collect();

    serde_json::json!({
        "available": true,
        "source": "prefill_engine.h_logits_after_lm_head_download_and_suppression",
        "dtype": "f32",
        "device_before_download": "cuda",
        "host_buffer": "engine.h_logits",
        "vocab_size": vocab_size,
        "selected_token_id": selected_token,
        "selected_raw_logit": selected_raw_logit as f64,
        "selected_logprob_from_raw": selected_logprob,
        "selected_softmax_prob_from_raw": selected_logprob.exp(),
        "max_logit": max_logit as f64,
        "max_token_id": max_token,
        "min_logit": min_logit as f64,
        "min_token_id": min_token,
        "sum_exp_shifted": sum_exp,
        "logsumexp": log_sum_exp,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "top_logits_before_logprob": top_entries,
    })
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
    logprobs: Option<&[(u32, f32)]>,
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
    let logprobs_str = if let Some(lps) = logprobs {
        // OpenAI format: {"content": [{"token": "...", "logprob": -0.5, "top_logprobs": [{"token": "...", "logprob": -0.5}, ...]}]}
        let mut top_entries = Vec::new();
        for &(tid, lp) in lps.iter() {
            top_entries.push(format!(r#"{{"token_id":{},"logprob":{:.6}}}"#, tid, lp));
        }
        let top_str = top_entries.join(",");
        // The first entry is the selected token
        let selected_lp = if !lps.is_empty() { lps[0].1 } else { 0.0 };
        format!(r#","logprobs":{{"content":[{{"logprob":{:.6},"top_logprobs":[{}]}}]}}"#, selected_lp, top_str)
    } else {
        String::new()
    };
    format!(
        r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[{{"index":0,"delta":{},"finish_reason":{}{}}}]}}"#,
        request_id, created, model_name, delta, fr, logprobs_str
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
    state: &mut ServerState,
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

        ("POST", "/v1/internal/prefill_logits") => {
            if state.test_endpoints {
                handle_prefill_logits(&mut tcp_stream, &request.body, state);
            } else {
                let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#);
            }
        }

        ("POST", "/v1/internal/reference_test") => {
            if state.test_endpoints {
                handle_reference_test(&mut tcp_stream, &request.body, state);
            } else {
                let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Test endpoints not enabled. Start server with --test-endpoints"}"#);
            }
        }

        _ => {
            let _ = send_json(&mut tcp_stream, 404, r#"{"error":"Not found"}"#);
        }
    }
}

/// Overhead timings collected during request setup (before decode).
struct RequestOverhead {
    parse_ms: f64,           // HTTP parse + JSON parse + tokenization
    evict_ms: f64,           // HCS soft-tier eviction
    prefill_ms: f64,         // GIL acquire + Python prefill
    reload_ms: f64,          // HCS soft-tier reload (wall-clock, includes sync if enabled)
    real_reload_dma_ms: f64, // Actual DMA time when sync is on (0.0 if async)
}

/// Handle /v1/chat/completions request.
fn handle_chat_completion(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
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
    let req_logprobs = req.get("logprobs").and_then(|v| v.as_bool()).unwrap_or(false);
    let req_top_logprobs = req.get("top_logprobs").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let logprobs_top_n = if req_logprobs { req_top_logprobs } else { 0 };
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

    // ── Render chat template (reused for both token estimation and Rust prefill) ──
    let rendered = match state.chat_template.apply_with_tools(
        &messages_json,
        &tools_json,
        true,
        enable_thinking,
    ) {
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
    let estimated_tokens = {
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

    // ── Prefill: Rust path (zero GIL) or Python fallback ──
    crate::vram_monitor::report_event("prefill_start");
    let t_prefill_gil = Instant::now();

    let prefill_result: Result<(usize, usize, Vec<usize>, bool), String> = {
        // ── Rust prefill: zero GIL, zero Python ──
        let token_ids: Vec<u32> = match state.tokenizer.encode(rendered.as_str(), true) {
            Ok(e) => e.get_ids().to_vec(),
            Err(e) => {
                let _ = send_json(stream, 500, &format!(r#"{{"error":"Tokenize: {}"}}"#, e));
                return;
            }
        };
        let mut engine_guard = state.rust_prefill.lock().unwrap();
        let engine = engine_guard.as_mut().unwrap();

        // Update HCS snapshot so prefill can use GPU-resident experts directly
        {
            let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
            let (cache_fast, ne) = store.export_hcs_snapshot();
            engine.update_hcs_snapshot(cache_fast, ne);
        }

        let kv_max_seq = engine.kv_max_seq;
        let kv_overflow = token_ids.len() > kv_max_seq;

        let _has_hqq_runtime_slots = {
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            match prepare_store_for_rust_prefill(store, engine) {
                Ok(has_hqq) => has_hqq,
                Err(e) => {
                    let _ = send_json(stream, 500, &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e));
                    return;
                }
            }
        };

        engine.set_prefill_hcs_guard_store_addr(state.gpu_store_addr);

        // Dynamically allocate scratch sized for this prompt
        if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
            engine.clear_prefill_hcs_guard_store_addr();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = send_json(stream, 500, &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e));
            return;
        }

        let suppress_tokens = {
            let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
            store.suppress_tokens_clone()
        };
        let result = match engine.run_prefill(
            &token_ids,
            temperature,
            &suppress_tokens,
        ) {
            Ok(r) => match engine.finalize_stage_exact_prefill_kv(r.prompt_len) {
                Ok(()) => Ok(r),
                Err(e) => Err(format!("KV stage export failed: {}", e)),
            },
            Err(e) => Err(e),
        };

        // Release scratch to free VRAM for decode/HCS
        if let Err(e) = engine.release_scratch() {
            log::error!("Failed to release scratch: {}", e);
        }
        engine.clear_prefill_hcs_guard_store_addr();

        // Convert stop token strings to IDs, and always include model's EOS tokens
        let mut stop_ids: Vec<usize> = state.eos_stop_ids.clone();
        for s in &stop_tokens {
            if let Some(id) = state.tokenizer.token_to_id(s) {
                let id = id as usize;
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }

        match result {
            Ok(r) => {
                // Set KV cache position on decode store so decode knows where to continue
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                if let Err(e) = restore_store_after_rust_prefill(store, r.prompt_len) {
                    log::error!("Failed to restore decode runtime after prefill: {}", e);
                }
                Ok((r.first_token as usize, r.prompt_len, stop_ids, kv_overflow))
            }
            Err(e) => {
                let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
                let _ = store.prepare_runtime_for_decode_rust();
                Err(e)
            }
        }
    };

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

    // ── Reload soft HCS after prefill ──
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
    // Always sync: wait for reload DMA to complete before decode starts.
    // Decode must never begin with an incomplete HCS — with dynamic chunking
    // the entire soft tier is evicted for prefill and must be fully restored.
    let (activated, real_reload_ms) = store.hcs_sync_soft_reload();
    if activated > 0 {
        log::info!("Request {}: HCS reload complete: {} experts, {:.1}ms DMA",
            request_id, activated, real_reload_ms);
    }
    // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
    // ── Multi-GPU: copy KV+LA state from primary to all aux GPUs after prefill ──
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
            // Copy LA recurrent state (conv_state + recur_state) for linear attention layers
            if let Err(e) = store.copy_la_states_to_aux(aux_store, layer_start, layer_end) {
                log::error!("Request {}: LA state copy to aux GPU{} failed: {}", request_id, i + 1, e);
            }
        }
        let kvcopy_ms = t_kvcopy.elapsed().as_secs_f64() * 1000.0;
        log::info!("Request {}: KV+LA state copied to {} aux GPUs in {:.1}ms", request_id, num_aux, kvcopy_ms);
    }
    let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;

    let overhead = RequestOverhead {
        parse_ms,
        evict_ms,
        prefill_ms: prefill_gil_ms,
        reload_ms, // includes sync wait
        real_reload_dma_ms: real_reload_ms, // actual DMA time (0 if async)
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
        &overhead, has_tools, enable_thinking, logprobs_top_n,
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

/// Handle /v1/internal/prefill_logits endpoint.
/// Runs a full prefill pass and extracts top-k logprobs at sampled positions.
fn handle_prefill_logits(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
) {
    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(stream, 400, &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e));
            return;
        }
    };

    let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let sample_every = req.get("sample_every").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

    // Accept either raw input_token_ids or messages (with chat template + tokenization)
    let token_ids: Vec<u32> = if let Some(serde_json::Value::Array(arr)) = req.get("input_token_ids") {
        arr.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()
    } else if let Some(messages) = req.get("messages") {
        let messages_json = messages.to_string();
        let enable_thinking = req.get("enable_thinking").and_then(|v| v.as_bool()).unwrap_or(false);
        let rendered = match state.chat_template.apply(&messages_json, true, enable_thinking) {
            Ok(r) => r,
            Err(e) => {
                let _ = send_json(stream, 500, &format!(r#"{{"error":"Chat template: {}"}}"#, e));
                return;
            }
        };
        match state.tokenizer.encode(rendered.as_str(), true) {
            Ok(e) => e.get_ids().to_vec(),
            Err(e) => {
                let _ = send_json(stream, 500, &format!(r#"{{"error":"Tokenize: {}"}}"#, e));
                return;
            }
        }
    } else {
        let _ = send_json(stream, 400, r#"{"error":"Missing input_token_ids or messages"}"#);
        return;
    };

    log::info!("prefill_logits: {} tokens, top_k={}, sample_every={}", token_ids.len(), top_k, sample_every);

    // Evict soft HCS before diagnostic prefill so this endpoint uses the same
    // conservative VRAM budget as the production and reference-test paths.
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (_evicted, _freed_mb) = store_for_evict.hcs_evict_for_prefill(token_ids.len());

    // Run prefill logits extraction
    let mut engine_guard = state.rust_prefill.lock().unwrap();
    let engine = match engine_guard.as_mut() {
        Some(e) => e,
        None => {
            let _ = send_json(stream, 500, r#"{"error":"Rust prefill engine not available"}"#);
            return;
        }
    };

    // Update HCS snapshot
    {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        engine.update_hcs_snapshot(cache_fast, ne);
    }

    let _has_hqq_runtime_slots = {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match prepare_store_for_rust_prefill(store, engine) {
            Ok(has_hqq) => has_hqq,
            Err(e) => {
                let _ = send_json(stream, 500, &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e));
                return;
            }
        }
    };

    // Dynamically allocate scratch for this prompt
    // run_prefill_logits needs scratch sized for all tokens (no chunking)
    if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        let _ = store.prepare_runtime_for_decode_rust();
        store.invalidate_cuda_graph();
        log::info!("prefill_logits: invalidated CUDA graphs after failed scratch allocation restore");
        let _ = send_json(stream, 500, &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e));
        return;
    }

    let positions = match engine.run_prefill_logits(&token_ids, top_k, sample_every) {
        Ok(p) => p,
        Err(e) => {
            // Release scratch even on error
            let _ = engine.release_scratch();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = store.hcs_reload_after_prefill_async(token_ids.len());
            let _ = store.hcs_sync_soft_reload();
            store.invalidate_cuda_graph();
            log::info!("prefill_logits: invalidated CUDA graphs after failed diagnostic prefill restore");
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            let _ = send_json(stream, 500, &format!(r#"{{"error":"Prefill logits: {}"}}"#, e));
            return;
        }
    };

    // Release scratch after logits extraction
    if let Err(e) = engine.release_scratch() {
        log::error!("Failed to release scratch after prefill_logits: {}", e);
    }

    // Restore evicted soft HCS so the next decode/reference request starts
    // from the normal steady-state cache residency.
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let _ = store.prepare_runtime_for_decode_rust();
    let _ = store.hcs_reload_after_prefill_async(token_ids.len());
    let _ = store.hcs_sync_soft_reload();
    log::info!("prefill_logits: restored decode runtime after diagnostic prefill");

    // Match the normal reference/inference cleanup path so diagnostic prefill
    // requests do not leak sequence state into the next prompt.
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });

    // Format response: {positions: [{position, top_k: [{token_id, logprob}]}]}
    let mut pos_json = Vec::new();
    for p in &positions {
        let mut tk_json = Vec::new();
        for &(tid, lp) in &p.top_k {
            tk_json.push(format!(r#"{{"token_id":{},"logprob":{:.6}}}"#, tid, lp));
        }
        pos_json.push(format!(
            r#"{{"position":{},"top_k":[{}]}}"#,
            p.position,
            tk_json.join(",")
        ));
    }
    let response = format!(r#"{{"positions":[{}]}}"#, pos_json.join(","));
    let _ = send_json(stream, 200, &response);
}

/// Handle /v1/internal/reference_test endpoint.
/// Accepts raw input_token_ids, runs greedy prefill + decode, returns output tokens with logprobs.
/// Used for comparing engine output against BF16 reference data.
fn handle_reference_test(
    stream: &mut TcpStream,
    body: &str,
    state: &mut ServerState,
) {
    let t_start = Instant::now();
    state.reference_test_request_order = state.reference_test_request_order.saturating_add(1);
    let reference_request_order = state.reference_test_request_order;

    // Parse request
    let req: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = send_json(stream, 400, &format!(r#"{{"error":"Invalid JSON: {}"}}"#, e));
            return;
        }
    };

    // Required: input_token_ids (raw token IDs, no tokenization or template applied)
    let input_token_ids: Vec<u32> = match req.get("input_token_ids") {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()
        }
        _ => {
            let _ = send_json(stream, 400, r#"{"error":"Missing or invalid input_token_ids array"}"#);
            return;
        }
    };

    let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
    let top_logprobs = req.get("top_logprobs").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let debug_reference_trace = req
        .get("debug_reference_trace")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let client_request_id = req
        .get("debug_request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let input_token_hash = fnv1a_token_hash(&input_token_ids);

    // Stop token IDs (from reference data's eos_token_ids)
    let stop_ids: Vec<usize> = match req.get("stop_token_ids") {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect()
        }
        _ => state.eos_stop_ids.clone(),
    };

    log::info!("reference_test: {} input tokens, max_tokens={}, top_logprobs={}, stop_ids={:?}",
        input_token_ids.len(), max_tokens, top_logprobs, stop_ids);

    // ── Evict soft HCS before prefill ──
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (evicted, freed_mb) = store_for_evict.hcs_evict_for_prefill(input_token_ids.len());

    // ── Prefill with raw token IDs (no tokenization, no chat template) ──
    let mut engine_guard = state.rust_prefill.lock().unwrap();
    let engine = match engine_guard.as_mut() {
        Some(e) => e,
        None => {
            let _ = send_json(stream, 500, r#"{"error":"Rust prefill engine not available"}"#);
            return;
        }
    };

    // Update HCS snapshot
    {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        engine.update_hcs_snapshot(cache_fast, ne);
    }

    let (hcs_snapshot_entries, hcs_num_experts_per_layer) = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        let (cache_fast, ne) = store.export_hcs_snapshot();
        (cache_fast.len(), ne)
    };

    let has_hqq_runtime_slots = {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        match prepare_store_for_rust_prefill(store, engine) {
            Ok(has_hqq) => has_hqq,
            Err(e) => {
                let _ = send_json(stream, 500, &format!(r#"{{"error":"Prefill prepare failed: {}"}}"#, e));
                return;
            }
        }
    };

    let hqq_prefill_materialized = false;

    // Dynamically allocate scratch for this prompt
    if let Err(e) = engine.prepare_for_prefill(input_token_ids.len()) {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        let _ = store.prepare_runtime_for_decode_rust();
        let _ = send_json(stream, 500, &format!(r#"{{"error":"Scratch alloc failed: {}"}}"#, e));
        return;
    }
    let scratch_tokens_after_prepare = engine.scratch.max_tokens;
    let prefill_chunk_size_after_prepare = engine.config.prefill_chunk_size;

    // Run prefill with temperature=0 (greedy)
    let suppress_tokens = {
        let store = unsafe { &*(state.gpu_store_addr as *const GpuDecodeStore) };
        store.suppress_tokens_clone()
    };
    engine.set_reference_debug_trace_enabled(debug_reference_trace);
    let prefill_result = match engine.run_prefill(
        &input_token_ids,
        0.0, // temperature=0 for greedy
        &suppress_tokens,
    ) {
        Ok(r) => match engine.finalize_stage_exact_prefill_kv(r.prompt_len) {
            Ok(()) => Ok(r),
            Err(e) => Err(format!("KV stage export failed: {}", e)),
        },
        Err(e) => Err(e),
    };
    let debug_prefill_stage_trace = if debug_reference_trace {
        engine.take_reference_debug_trace()
    } else {
        None
    };
    engine.set_reference_debug_trace_enabled(false);

    let (first_token, prompt_len, first_token_top_k, debug_prefill_logits) = match prefill_result {
        Ok(r) => {
            let first_token = r.first_token as usize;
            let first_token_top_k = crate::decode::extract_top_logprobs(&engine.h_logits, engine.h_logits.len(), top_logprobs);
            let debug_prefill_logits = if debug_reference_trace {
                Some(reference_logit_trace_json(
                    &engine.h_logits,
                    engine.h_logits.len(),
                    first_token,
                    top_logprobs,
                ))
            } else {
                None
            };
            (first_token, r.prompt_len, first_token_top_k, debug_prefill_logits)
        },
        Err(e) => {
            let _ = engine.release_scratch();
            let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
            let _ = store.prepare_runtime_for_decode_rust();
            let _ = send_json(stream, 500, &format!(r#"{{"error":"Prefill failed: {}"}}"#, e));
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

    // Release scratch to free VRAM for decode/HCS
    if let Err(e) = engine.release_scratch() {
        log::error!("reference_test: Failed to release scratch: {}", e);
    }

    // Set KV position and swap to simple INT4 for decode
    {
        let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
        if let Err(e) = restore_store_after_rust_prefill(store, prompt_len) {
            log::error!("reference_test: Failed to restore decode runtime: {}", e);
        }
    }

    let prefill_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // ── Reload soft HCS after prefill ──
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (queued, alloc_mb) = {
        let (queued, alloc_mb) = store.hcs_reload_after_prefill_async(prompt_len);
        if queued > 0 {
            log::info!("reference_test: HCS soft reload queued {} experts", queued);
        }
        (queued, alloc_mb)
    };
    // Always sync: decode must not start with incomplete HCS
    let (activated, dma_ms) = store.hcs_sync_soft_reload();
    if activated > 0 {
        log::info!("reference_test: HCS reload complete: {} experts, {:.1}ms DMA", activated, dma_ms);
    }

    // Disable thinking suppression for reference test (greedy, no thinking budget logic)
    store.set_think_end_suppress(None, 0);
    store.set_min_new_tokens_ext(0, vec![]);
    let gqa_diag_layer = std::env::var("KRASIS_GQA_DIAG_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(layer_idx) = gqa_diag_layer {
        store.set_debug_gqa_diag_layer(Some(layer_idx));
        log::info!("reference_test: enabled GQA decode diagnostic capture for layer {}", layer_idx);
    }

    // ── Greedy decode with logprobs collection ──
    let t_decode = Instant::now();
    let tokenizer = &state.tokenizer;

    // Collect all output tokens and their top-k logprobs
    let mut output_tokens: Vec<(usize, Vec<(u32, f32)>)> = Vec::new();
    let mut all_text = String::new();
    let mut finish_reason = "length".to_string();

    // First token
    let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
    all_text.push_str(&first_text);
    output_tokens.push((first_token, first_token_top_k.clone()));

    let decode_budget = max_tokens.saturating_sub(1);

    {
        let mut on_token = |token_id: usize, text: &str, fr: Option<&str>, token_logprobs: Option<&[(u32, f32)]>| -> bool {
            all_text.push_str(text);
            let lps = token_logprobs.map(|s| s.to_vec()).unwrap_or_default();
            output_tokens.push((token_id, lps));
            if let Some(r) = fr {
                finish_reason = r.to_string();
            }
            true
        };

        store.gpu_generate_stream(
            first_token,
            prompt_len,
            decode_budget,
            0.0,  // temperature=0 (greedy)
            1,    // top_k=1 (greedy)
            1.0,  // top_p=1.0
            &stop_ids,
            tokenizer,
            0.0,  // no presence penalty
            top_logprobs,
            Some("reference_test".to_string()),
            on_token,
        );
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    if gqa_diag_layer.is_some() {
        if let Ok(path) = std::env::var("KRASIS_GQA_DIAG_DUMP") {
            match store.debug_gqa_diag_json() {
                Ok(payload) => {
                    if let Err(e) = std::fs::write(&path, payload) {
                        log::error!("reference_test: failed to write GQA diagnostic {}: {}", path, e);
                    } else {
                        log::info!("reference_test: wrote GQA diagnostic {}", path);
                    }
                }
                Err(e) => {
                    log::error!("reference_test: failed to capture GQA diagnostic: {}", e);
                }
            }
        }
        store.set_debug_gqa_diag_layer(None);
    }

    // ── Cleanup ──
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    let server_cleanup_called = true;

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    // ── Format response ──
    let mut per_token_json = Vec::new();
    for (tid, logprobs) in &output_tokens {
        let mut tk_json = Vec::new();
        for &(lp_tid, lp_val) in logprobs {
            tk_json.push(format!(r#"{{"token_id":{},"log_prob":{:.6}}}"#, lp_tid, lp_val));
        }
        // Get log_prob for the selected token (first in top-k if available)
        let selected_lp = logprobs.iter()
            .find(|&&(t, _)| t == *tid as u32)
            .map(|&(_, lp)| lp)
            .unwrap_or(0.0);
        per_token_json.push(format!(
            r#"{{"token_id":{},"log_prob":{:.6},"top_k":[{}]}}"#,
            tid, selected_lp, tk_json.join(",")
        ));
    }

    // Escape text for JSON
    let text_escaped = serde_json::to_string(&all_text).unwrap_or_else(|_| "\"\"".to_string());

    let mut first_topk_json = Vec::new();
    for &(lp_tid, lp_val) in &first_token_top_k {
        first_topk_json.push(format!(r#"{{"token_id":{},"log_prob":{:.6}}}"#, lp_tid, lp_val));
    }

    let debug_json_suffix = if debug_reference_trace {
        let final_top_logprobs: Vec<serde_json::Value> = first_token_top_k
            .iter()
            .enumerate()
            .map(|(rank, &(token_id, log_prob))| {
                serde_json::json!({
                    "rank": rank + 1,
                    "token_id": token_id,
                    "log_prob": log_prob as f64,
                })
            })
            .collect();
        let selected_logprob_from_endpoint = first_token_top_k
            .iter()
            .find(|&&(token_id, _)| token_id == first_token as u32)
            .map(|&(_, log_prob)| log_prob as f64);
        let trace = serde_json::json!({
            "schema": "krasis_reference_test_debug_v1",
            "request_order": reference_request_order,
            "client_request_id": client_request_id,
            "input_token_count": input_token_ids.len(),
            "input_token_hash_fnv1a64": format!("0x{:016x}", input_token_hash),
            "max_tokens": max_tokens,
            "top_logprobs": top_logprobs,
            "stop_token_ids": stop_ids,
            "selected_token_id": first_token,
            "prompt_len": prompt_len,
            "state_reset_proof": {
                "fresh_prefill_run": true,
                "run_prefill_zeroes_la_state": true,
                "hcs_evict_for_prefill_called": true,
                "hcs_evicted_experts": evicted,
                "hcs_freed_mb": freed_mb,
                "hcs_snapshot_entries": hcs_snapshot_entries,
                "hcs_num_experts_per_layer": hcs_num_experts_per_layer,
                "prepare_runtime_for_prefill_called": true,
                "has_hqq_runtime_slots": has_hqq_runtime_slots,
                "hqq_prefill_materialized": hqq_prefill_materialized,
                "prepare_for_prefill_prompt_tokens": input_token_ids.len(),
                "scratch_tokens_after_prepare": scratch_tokens_after_prepare,
                "prefill_chunk_size_after_prepare": prefill_chunk_size_after_prepare,
                "release_scratch_called": true,
                "restore_runtime_for_decode_called": true,
                "decode_kv_position_set_to_prompt_len": prompt_len,
                "hcs_reload_after_prefill_queued": queued,
                "hcs_reload_after_prefill_alloc_mb": alloc_mb,
                "hcs_sync_soft_reload_activated": activated,
                "hcs_sync_soft_reload_dma_ms": dma_ms,
                "server_cleanup_called": server_cleanup_called
            },
            "prefill_stage_trace": debug_prefill_stage_trace.unwrap_or_else(|| serde_json::json!({"available": false})),
            "prefill_logits": debug_prefill_logits.unwrap_or_else(|| serde_json::json!({"available": false})),
            "final_top_logprobs": final_top_logprobs,
            "selected_logprob_from_endpoint": selected_logprob_from_endpoint,
            "timing": {
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "prompt_tokens": prompt_len
            }
        });
        format!(r#","debug_reference_trace":{}"#, trace)
    } else {
        String::new()
    };

    let response = format!(
        r#"{{"token_ids":[{}],"text":{},"num_tokens":{},"per_token_data":[{}],"first_token_top_k":[{}],"finish_reason":"{}","timing":{{"prefill_ms":{:.1},"decode_ms":{:.1},"total_ms":{:.1},"prompt_tokens":{}}}{}}}"#,
        output_tokens.iter().map(|(t, _)| t.to_string()).collect::<Vec<_>>().join(","),
        text_escaped,
        output_tokens.len(),
        per_token_json.join(","),
        first_topk_json.join(","),
        finish_reason,
        prefill_ms, decode_ms, total_ms, prompt_len,
        debug_json_suffix
    );

    log::info!("reference_test: {} output tokens in {:.0}ms (prefill={:.0}ms decode={:.0}ms), finish={}",
        output_tokens.len(), total_ms, prefill_ms, decode_ms, finish_reason);

    let _ = send_json(stream, 200, &response);
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
    logprobs_top_n: usize,
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
            let think_chunk = format_sse_token(request_id, model_name, "<think>", None, created, None);
            let _ = send_sse_chunk(stream, &think_chunk);
        }

        // When tool use is active, buffer first token (might need tool call parsing).
        // Otherwise send immediately for lowest latency.
        if !has_tools {
            let chunk = format_sse_token(request_id, model_name, &first_text, None, created, None);
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
                        let chunk = format_sse_token(request_id, model_name, before, None, created, None);
                        let _ = tx.send(format!("data: {}\n\n", chunk));
                    }
                }
            } else if !first_text.is_empty() {
                let chunk = format_sse_token(request_id, model_name, &first_text, None, created, None);
                let _ = tx.send(format!("data: {}\n\n", chunk));
            }
        }

        // Shared callback for both single-GPU and multi-GPU decode
        let mut on_token = |token_id: usize, text: &str, finish_reason: Option<&str>, token_logprobs: Option<&[(u32, f32)]>| -> bool {
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
                                request_id, model_name, before, None, created, None,
                            );
                            let _ = tx.send(format!("data: {}\n\n", chunk));
                        }
                    }
                } else {
                    // Normal content — stream it (no finish_reason; handled post-generation)
                    if !text.is_empty() {
                        let chunk = format_sse_token(
                            request_id, model_name, text, None, created, token_logprobs,
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
                    request_id, model_name, text, effective_finish, created, token_logprobs,
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
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
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
                logprobs_top_n,
                Some(format!("chat_{}", request_id)),
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
                    request_id, model_name, "", Some("tool_calls"), created, None,
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
                    request_id, model_name, "", Some(fr), created, None,
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
            r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[],"krasis_timing":{{"decode_tokens":{},"decode_time_ms":{:.1},"decode_tok_s":{:.2},"thinking_tokens":{},"answer_tokens":{},"total_generated":{},"prompt_tokens":{},"prefill_tok_s":{:.1},"overhead_ms":{:.1},"overhead":{{"parse_ms":{:.1},"evict_ms":{:.1},"prefill_ms":{:.1},"reload_ms":{:.1},"real_reload_dma_ms":{:.1}}}}}}}"#,
            request_id, created, model_name,
            decode_token_count, decode_ms, decode_tok_s,
            thinking_token_count, answer_token_count,
            total_gen, prompt_len, prefill_tok_s,
            overhead_total_ms,
            overhead.parse_ms, overhead.evict_ms, overhead.prefill_ms, overhead.reload_ms,
            overhead.real_reload_dma_ms
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
            let mut on_token = |token_id: usize, text: &str, finish_reason: Option<&str>, _logprobs: Option<&[(u32, f32)]>| -> bool {
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
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
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
                    logprobs_top_n,
                    Some(format!("chat_{}_nosse", request_id)),
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
    /// Shared Rust prefill engine — used by both serve_forever (HTTP requests)
    /// and benchmark_request (engine benchmarks). Arc+Mutex allows both paths
    /// to share the single pre-allocated engine without moving it.
    prefill_engine: Arc<std::sync::Mutex<Option<crate::gpu_prefill::PrefillEngine>>>,
    /// Enable test-only endpoints (/v1/internal/prefill_logits)
    test_endpoints: bool,
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens, enable_thinking=true, thinking_end_token_id=0, gpu_store_addr=0, aux_gpu_store_addrs=Vec::new(), multi_gpu_split_layers=Vec::new(), multi_gpu_gqa_offsets=Vec::new(), test_endpoints=false))]
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
        test_endpoints: bool,
    ) -> Self {
        // Take the pre-allocated Rust prefill engine from the decode store.
        // The engine was pre-allocated from Python (before HCS pool loading)
        // so it already has its VRAM allocated. Creating a new one here would
        // fail because HCS has consumed most remaining VRAM.
        let prefill_engine = if gpu_store_addr != 0 {
            let store = unsafe { &mut *(gpu_store_addr as *mut GpuDecodeStore) };
            match store.take_prefill_engine() {
                Some(engine) => {
                    log::info!("RustServer: took pre-allocated prefill engine for benchmarks");
                    Some(engine)
                }
                None => {
                    log::warn!("RustServer: no pre-allocated prefill engine, creating on demand");
                    match create_prefill_engine_for_server(store, max_context_tokens) {
                        Ok(engine) => {
                            log::info!("RustServer: prefill engine created on demand (max_tokens={})", max_context_tokens);
                            Some(engine)
                        }
                        Err(e) => {
                            log::error!("RustServer: prefill engine failed: {}", e);
                            None
                        }
                    }
                }
            }
        } else {
            None
        };

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
            prefill_engine: Arc::new(std::sync::Mutex::new(prefill_engine)),
            test_endpoints,
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
        let test_endpoints = self.test_endpoints;
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

            // Load EOS token IDs from generation_config.json (same directory as tokenizer.json).
            // These are required for the model to stop generating — without them, decode
            // never terminates (the main branch gets these from Python prefill result).
            let eos_stop_ids = {
                let p = std::path::Path::new(&tokenizer_path);
                let gen_cfg_path = p.parent().unwrap_or(p).join("generation_config.json");
                let mut ids = Vec::new();
                if let Ok(data) = std::fs::read_to_string(&gen_cfg_path) {
                    if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
                        match cfg.get("eos_token_id") {
                            Some(serde_json::Value::Number(n)) => {
                                if let Some(id) = n.as_u64() {
                                    ids.push(id as usize);
                                }
                            }
                            Some(serde_json::Value::Array(arr)) => {
                                for v in arr {
                                    if let Some(id) = v.as_u64() {
                                        ids.push(id as usize);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if ids.is_empty() {
                    log::warn!("No eos_token_id found in generation_config.json — decode may not stop");
                } else {
                    log::info!("EOS stop tokens: {:?}", ids);
                }
                ids
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

            // Share the prefill engine from the RustServer via Arc clone.
            // If RustServer::new() took the pre-allocated engine (it should have),
            // it's already in the shared Mutex. If not, try the decode store.
            let rust_prefill = {
                let has_engine = self.prefill_engine.lock().unwrap().is_some();
                if has_engine {
                    log::info!("Rust prefill engine shared via Arc (was pre-allocated)");
                    self.prefill_engine.clone()
                } else {
                    // Not in the shared Mutex — try the decode store
                    let store = unsafe { &mut *(gpu_store_addr as *mut GpuDecodeStore) };
                    match store.take_prefill_engine() {
                        Some(engine) => {
                            log::info!("Rust prefill engine taken from decode store pre-allocated slot");
                            let arc = Arc::new(std::sync::Mutex::new(Some(engine)));
                            arc
                        }
                        None => {
                            log::warn!("No pre-allocated prefill engine — creating on demand");
                            match create_prefill_engine_for_server(store, max_context_tokens) {
                                Ok(engine) => {
                                    log::info!("Rust prefill engine created on demand (max_tokens={})", max_context_tokens);
                                    Arc::new(std::sync::Mutex::new(Some(engine)))
                                }
                                Err(e) => {
                                    log::error!("Rust prefill engine failed: {}", e);
                                    log::error!("Cannot start server without Rust prefill engine");
                                    return;
                                }
                            }
                        }
                    }
                }
            };

            let mut state = ServerState {
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
                rust_prefill,
                test_endpoints,
                eos_stop_ids,
                reference_test_request_order: 0,
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
                        handle_request(stream, &mut state);
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

        // Estimate tokens by applying the same chat template mode the request will use.
        let estimated_tokens = {
            let rendered = chat_template.apply(&messages_json, true, enable_thinking)
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

        // Prefill (Rust, zero GIL)
        crate::vram_monitor::report_event("prefill_start");
        let t_prefill = Instant::now();

        let mut engine_guard = self.prefill_engine.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Prefill engine lock poisoned: {}", e))
        })?;
        let engine = engine_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Rust prefill engine not available for benchmark")
        })?;

        // Update HCS snapshot
        {
            let store_ref = unsafe { &*(self.gpu_store_addr as *const GpuDecodeStore) };
            let (cache_fast, ne) = store_ref.export_hcs_snapshot();
            engine.update_hcs_snapshot(cache_fast, ne);
        }

        // Tokenize using Rust tokenizer (always with generation prompt)
        let token_ids: Vec<u32> = {
            let rendered = chat_template.apply(&messages_json, true, enable_thinking)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Chat template failed: {}", e)))?;
            let encoding = tokenizer.encode(rendered.as_str(), true)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Tokenizer failed: {}", e)))?;
            encoding.get_ids().to_vec()
        };

        let kv_overflow = token_ids.len() > engine.kv_max_seq;

        let _has_hqq_runtime_slots = prepare_store_for_rust_prefill(store, engine)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to prepare runtime for prefill: {}", e)))?;

        engine.set_prefill_hcs_guard_store_addr(self.gpu_store_addr);

        // Dynamically allocate scratch for this prompt
        if let Err(e) = engine.prepare_for_prefill(token_ids.len()) {
            engine.clear_prefill_hcs_guard_store_addr();
            let _ = store.prepare_runtime_for_decode_rust();
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Scratch alloc failed: {}", e)));
        }

        let suppress_tokens = store.suppress_tokens_clone();
        let prefill_result = match engine.run_prefill(
            &token_ids,
            temperature,
            &suppress_tokens,
        ) {
            Ok(r) => match engine.finalize_stage_exact_prefill_kv(r.prompt_len) {
                Ok(()) => Ok(r),
                Err(e) => Err(format!("KV stage export failed: {}", e)),
            },
            Err(e) => Err(e),
        }.map_err(|e| {
            engine.clear_prefill_hcs_guard_store_addr();
            let _ = engine.release_scratch();
            let _ = store.prepare_runtime_for_decode_rust();
            pyo3::exceptions::PyRuntimeError::new_err(
                format!("Rust prefill failed: {}", e))
        })?;

        // Release scratch to free VRAM for decode/HCS
        if let Err(e) = engine.release_scratch() {
            log::error!("Failed to release scratch: {}", e);
        }
        engine.clear_prefill_hcs_guard_store_addr();

        let first_token = prefill_result.first_token as usize;
        let prompt_len = prefill_result.prompt_len;
        // Load EOS tokens for benchmark path (same logic as serve_forever)
        let stop_ids: Vec<usize> = {
            let p = std::path::Path::new(&self.tokenizer_path);
            let gen_cfg_path = p.parent().unwrap_or(p).join("generation_config.json");
            let mut ids = Vec::new();
            if let Ok(data) = std::fs::read_to_string(&gen_cfg_path) {
                if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
                    match cfg.get("eos_token_id") {
                        Some(serde_json::Value::Number(n)) => {
                            if let Some(id) = n.as_u64() { ids.push(id as usize); }
                        }
                        Some(serde_json::Value::Array(arr)) => {
                            for v in arr {
                                if let Some(id) = v.as_u64() { ids.push(id as usize); }
                            }
                        }
                        _ => {}
                    }
                }
            }
            ids
        };

        if let Err(e) = restore_store_after_rust_prefill(store, prompt_len) {
            log::error!("Failed to restore decode runtime after prefill: {}", e);
        }

        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("prefill_end");

        if kv_overflow || max_new_tokens <= 1 {
            self.py_model.call_method0(py, "server_cleanup")?;

            let prefill_tok_s = if prefill_ms > 0.0 {
                prompt_len as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            };
            let (min_free_vram_mb, mut hcs_loaded, mut hcs_total, _) = store.benchmark_stats();
            let safety_margin_mb = store.hcs_safety_margin_mb();
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

            let result = serde_json::json!({
                "prefill_ms": prefill_ms,
                "prefill_tok_s": prefill_tok_s,
                "prompt_tokens": prompt_len,
                "decode_ms": 0.0,
                "decode_tok_s": 0.0,
                "decode_tokens": 1,
                "evict_ms": evict_ms,
                "reload_ms": 0.0,
                "real_reload_dma_ms": 0.0,
                "min_free_vram_mb": min_free_vram_mb,
                "hcs_loaded": hcs_loaded,
                "hcs_total": hcs_total,
                "hcs_pct": hcs_pct,
                "safety_margin_mb": safety_margin_mb,
            });

            return Ok(result.to_string());
        }

        // Reload soft HCS after prefill
        crate::vram_monitor::report_event("hcs_soft_load_start");
        let t_reload = Instant::now();
        let (queued, _alloc_mb) = store.hcs_reload_after_prefill_async(prompt_len);
        if queued > 0 {
            log::info!("Benchmark: HCS soft async reload queued {} experts ({} tokens)",
                queued, prompt_len);
        }
        // Always sync: decode must not start with incomplete HCS
        let (activated, real_reload_dma_ms) = store.hcs_sync_soft_reload();
        if activated > 0 {
            log::info!("Benchmark: HCS reload complete: {} experts, {:.1}ms DMA",
                activated, real_reload_dma_ms);
        }
        let reload_pending_at_decode_start = store.hcs_soft_reload_pending();
        // NOTE: aux GPUs have no soft tier (100% hard), no eviction/reload needed
        let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;
        crate::vram_monitor::report_event("hcs_soft_load_end");

        // Match the live request path's per-request decode suppression setup.
        if enable_thinking {
            if self.thinking_end_token_id > 0 {
                store.set_think_end_suppress(Some(self.thinking_end_token_id), 4096);
                store.set_min_new_tokens_ext(0, stop_ids.clone());
            } else {
                store.set_think_end_suppress(None, 0);
                store.set_min_new_tokens_ext(0, vec![]);
            }
        } else {
            store.set_think_end_suppress(None, 0);
            store.set_min_new_tokens_ext(0, vec![]);
        }

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
                if let Err(e) = store.copy_la_states_to_aux(aux_store, layer_start, layer_end) {
                    log::error!("benchmark_request: LA state copy to aux GPU{} failed: {}", i + 1, e);
                }
            }
        }

        // Decode (pure Rust, GIL held but unused by decode loop)
        crate::vram_monitor::report_event("decode_start");
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
                0,      // logprobs_top_n
                Some("benchmark".to_string()),
                |_token_id: usize, _text: &str, _finish_reason: Option<&str>, _logprobs: Option<&[(u32, f32)]>| {
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
                0,      // logprobs_top_n
                Some("benchmark".to_string()),
                |_token_id, _text, _finish_reason, _logprobs: Option<&[(u32, f32)]>| {
                    count += 1;
                    true
                },
            );
        }
        let elapsed = decode_start.elapsed().as_secs_f64();
        let decode_tokens = count + 1; // includes first_token from prefill
        let decode_tok_s = if elapsed > 0.0 && count > 0 {
            count as f64 / elapsed
        } else {
            0.0
        };
        let decode_ms = elapsed * 1000.0;

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
        let state_validation_env = std::env::var("KRASIS_STATE_VALIDATION").ok();
        let config_validation_env = std::env::var("KRASIS_CONFIG_VALIDATION").ok();
        let state_validation_enabled = state_validation_env
            .as_deref()
            .map(|v| v != "0")
            .unwrap_or(false)
            || config_validation_env
                .as_deref()
                .map(|v| v != "0")
                .unwrap_or(false);
        let state_validation = if state_validation_enabled {
            let raw = store.config_validation_snapshot_json(
                prompt_len,
                true, // sync is always on
                reload_pending_at_decode_start,
            );
            match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(v) => {
                    log::info!("STATE_VALIDATION {}", v);
                    Some(v)
                }
                Err(e) => {
                    log::warn!("STATE_VALIDATION parse failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let mut result = serde_json::json!({
            "prefill_ms": prefill_ms,
            "prefill_tok_s": prefill_tok_s,
            "prompt_tokens": prompt_len,
            "decode_ms": decode_ms,
            "decode_tok_s": decode_tok_s,
            "decode_tokens": decode_tokens,
            "evict_ms": evict_ms,
            "reload_ms": reload_ms,
            "real_reload_dma_ms": real_reload_dma_ms,
            "min_free_vram_mb": min_free_vram_mb,
            "hcs_loaded": hcs_loaded,
            "hcs_total": hcs_total,
            "hcs_pct": hcs_pct,
            "safety_margin_mb": safety_margin_mb,
        });
        if let Some(v) = state_validation {
            result["state_validation"] = v;
        }

        Ok(result.to_string())
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
