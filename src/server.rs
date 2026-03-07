//! Rust HTTP server for Krasis — replaces Python FastAPI/uvicorn.
//!
//! Handles tokenization, HTTP parsing, and SSE streaming entirely in Rust.
//! Python is called only for GPU prefill (unavoidable — PyTorch/CUDA).
//! The decode loop runs GIL-free with zero Python involvement.
//!
//! Single-request at a time (matches our hardware constraint).

use crate::gpu_decode::GpuDecodeStore;
use pyo3::prelude::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::sync::mpsc;

/// Global pointer to the server's `running` flag so the raw SIGINT handler
/// can set it to `false` without going through Python's signal mechanism.
/// This is only written once (before the accept loop) and read from the
/// signal handler, so the raw pointer is safe in practice.
static SIGINT_RUNNING: AtomicBool = AtomicBool::new(false);
static SIGINT_FLAG_PTR: std::sync::atomic::AtomicPtr<AtomicBool> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

extern "C" fn sigint_handler(_sig: libc::c_int) {
    let ptr = SIGINT_FLAG_PTR.load(Ordering::Acquire);
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
    max_context_tokens: usize,
    default_enable_thinking: bool,
    /// Raw pointer to a GpuDecodeStore instance (set from Python during server init).
    /// Safety: single-request guarantee means no concurrent access.
    gpu_store_addr: usize,
    /// When true, log wall-clock time for each Python GIL acquisition.
    /// Enabled by KRASIS_GIL_TIMING=1. Zero cost when off (branch only).
    gil_timing: bool,
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
    let delta = if finish_reason.is_some() {
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
                r#"{{"object":"list","data":[{{"id":"{}","object":"model","owned_by":"krasis"}}]}}"#,
                state.model_name
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

    let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(2048) as usize;
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

    // ── Estimate prompt tokens for soft-tier HCS eviction ──
    let estimated_tokens = {
        // Extract text from messages JSON and tokenize with Rust tokenizer
        let mut text = String::new();
        if let Ok(msgs) = serde_json::from_str::<Vec<serde_json::Value>>(&messages_json) {
            for msg in &msgs {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    text.push_str(content);
                    text.push(' ');
                }
            }
        }
        let text_len = text.len();
        let base_count = state.tokenizer.encode(text, false)
            .map(|e| e.len())
            .unwrap_or(text_len / 4); // fallback: ~4 chars/token
        let est = base_count + 200; // approximate chat template overhead
        log::info!("Soft HCS: estimated {} tokens (text_len={}, base_count={})",
            est, text_len, base_count);
        est
    };
    let parse_ms = t_request.elapsed().as_secs_f64() * 1000.0;

    // ── Evict soft HCS before prefill to free VRAM ──
    let t_evict = Instant::now();
    let store_for_evict = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    let (evicted, _freed_mb) = store_for_evict.hcs_evict_for_prefill(estimated_tokens);
    let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;

    // ── Call Python for prefill (GIL required) ──
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

    let (first_token, prompt_len, stop_ids, kv_overflow) = match prefill_result {
        Ok(v) => v,
        Err(e) => {
            log::error!("Prefill failed: {}", e);
            let _ = send_json(
                stream,
                500,
                &format!(r#"{{"error":"Prefill failed: {}"}}"#, e),
            );
            // Cleanup on error
            Python::with_gil(|py| {
                let _ = state.py_model.call_method0(py, "server_cleanup");
            });
            return;
        }
    };

    // If prompt exceeded Rust KV cache, skip decode (return first token only)
    if kv_overflow {
        log::warn!("Request {}: prompt {} tokens exceeds Rust KV cache, returning first token only",
            request_id, prompt_len);
        let text = state.tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        let _ = send_json(
            stream,
            200,
            &format!(
                r#"{{"id":"chatcmpl-{}","object":"chat.completion","created":{},"model":"{}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{}"}},"finish_reason":"length"}}],"usage":{{"prompt_tokens":{},"completion_tokens":1,"total_tokens":{}}}}}"#,
                request_id, created, &state.model_name,
                text.replace('\\', "\\\\").replace('"', "\\\""),
                prompt_len, prompt_len + 1,
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
    let t_reload = Instant::now();
    let store = unsafe { &mut *(state.gpu_store_addr as *mut GpuDecodeStore) };
    if evicted > 0 {
        let (queued, _alloc_mb) = store.hcs_reload_after_prefill_async();
        log::info!("Request {}: HCS soft async reload queued {} experts (evicted {} for prefill)",
            request_id, queued, evicted);
    }
    let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;

    let overhead = RequestOverhead {
        parse_ms,
        evict_ms,
        prefill_ms: prefill_gil_ms,
        reload_ms,
    };

    // ── GPU decode: GIL-free Rust decode via GpuDecodeStore ──
    handle_gpu_decode(
        stream, is_stream, state, store, tokenizer,
        first_token, prompt_len, max_tokens, temperature,
        top_k, top_p, presence_penalty, &stop_ids,
        &request_id, &state.model_name, created,
        &overhead,
    );

    // ── Cleanup (GIL required) ──
    let t_cleanup_gil = Instant::now();
    Python::with_gil(|py| {
        let _ = state.py_model.call_method0(py, "server_cleanup");
    });
    let cleanup_gil_ms = t_cleanup_gil.elapsed().as_secs_f64() * 1000.0;

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
) {
    if is_stream {
        if let Err(e) = begin_sse(stream) {
            log::error!("Failed to send SSE headers: {}", e);
            return;
        }

        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        let chunk = format_sse_token(request_id, model_name, &first_text, None, created);
        let _ = send_sse_chunk(stream, &chunk);

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

        store.gpu_generate_stream(
            first_token,
            prompt_len,
            max_tokens.saturating_sub(1),
            temperature,
            top_k,
            top_p,
            stop_ids,
            tokenizer,
            presence_penalty,
            |_token_id, text, finish_reason| {
                decode_token_count += 1;
                let chunk = format_sse_token(request_id, model_name, text, finish_reason, created);
                let formatted = format!("data: {}\n\n", chunk);
                if tx.send(formatted).is_err() || writer_disconnected.load(Ordering::Acquire) {
                    return false;
                }
                true
            },
        );

        let elapsed = decode_start.elapsed().as_secs_f64();
        let total_gen = decode_token_count + 1;
        let decode_tok_s = if elapsed > 0.0 && decode_token_count > 0 {
            decode_token_count as f64 / elapsed
        } else {
            0.0
        };
        let decode_ms = elapsed * 1000.0;
        let overhead_total_ms = overhead.parse_ms + overhead.evict_ms + overhead.prefill_ms + overhead.reload_ms;
        let timing_chunk = format!(
            r#"{{"id":"{}","object":"chat.completion.chunk","created":{},"model":"{}","choices":[],"krasis_timing":{{"decode_tokens":{},"decode_time_ms":{:.1},"decode_tok_s":{:.2},"total_generated":{},"prompt_tokens":{},"overhead_ms":{:.1},"overhead":{{"parse_ms":{:.1},"evict_ms":{:.1},"prefill_ms":{:.1},"reload_ms":{:.1}}}}}}}"#,
            request_id, created, model_name,
            decode_token_count, decode_ms, decode_tok_s, total_gen, prompt_len,
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
        let mut all_text = String::new();
        let first_text = tokenizer.decode(&[first_token as u32], true).unwrap_or_default();
        all_text.push_str(&first_text);
        let mut total_tokens = 1usize;
        let mut finish = "length".to_string();

        store.gpu_generate_stream(
            first_token,
            prompt_len,
            max_tokens.saturating_sub(1),
            temperature,
            top_k,
            top_p,
            stop_ids,
            tokenizer,
            presence_penalty,
            |_token_id, text, finish_reason| {
                all_text.push_str(text);
                total_tokens += 1;
                if let Some(fr) = finish_reason {
                    finish = fr.to_string();
                }
                true
            },
        );

        let response = format_completion(
            request_id, model_name, &all_text, prompt_len,
            total_tokens, &finish, created,
        );
        let _ = send_json(stream, 200, &response);
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
    gpu_store_addr: usize,
    py_model: Py<PyAny>,
    running: Arc<AtomicBool>,
}

#[pymethods]
impl RustServer {
    #[new]
    #[pyo3(signature = (py_model, host, port, model_name, tokenizer_path, max_context_tokens, enable_thinking=true, gpu_store_addr=0))]
    fn new(
        py_model: PyObject,
        host: String,
        port: u16,
        model_name: String,
        tokenizer_path: String,
        max_context_tokens: usize,
        enable_thinking: bool,
        gpu_store_addr: usize,
    ) -> Self {
        Self {
            host,
            port,
            model_name,
            tokenizer_path,
            max_context_tokens,
            default_enable_thinking: enable_thinking,
            gpu_store_addr,
            py_model: py_model.into(),
            running: Arc::new(AtomicBool::new(false)),
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
        let gpu_store_addr = self.gpu_store_addr;
        let running = self.running.clone();

        // Install raw SIGINT handler BEFORE releasing the GIL.
        // Python's signal.signal handlers only dispatch between bytecodes,
        // but run() enters allow_threads (native Rust) so Python never gets
        // a chance to run the handler.  The raw handler sets `running` to
        // false directly, and the accept loop exits on the next 10ms poll.
        let running_ptr = Arc::as_ptr(&self.running) as *mut AtomicBool;
        SIGINT_FLAG_PTR.store(running_ptr, Ordering::Release);

        // Save previous handler so we can restore it
        let prev_handler;
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = sigint_handler as *const () as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = libc::SA_RESTART;
            let mut old_sa: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGINT, &sa, &mut old_sa);
            prev_handler = old_sa;
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

            let state = ServerState {
                py_model,
                model_name,
                tokenizer,
                max_context_tokens,
                default_enable_thinking,
                gpu_store_addr,
                gil_timing,
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

        // Restore previous SIGINT handler and clear global pointer
        SIGINT_FLAG_PTR.store(std::ptr::null_mut(), Ordering::Release);
        unsafe {
            libc::sigaction(libc::SIGINT, &prev_handler, std::ptr::null_mut());
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
        // Load tokenizer (same as server path)
        let tokenizer = tokenizers::Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load tokenizer: {}", e)))?;

        // Estimate tokens for HCS budget (same logic as server)
        let estimated_tokens = {
            let mut text = String::new();
            if let Ok(msgs) = serde_json::from_str::<Vec<serde_json::Value>>(&messages_json) {
                for msg in &msgs {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        text.push_str(content);
                        text.push(' ');
                    }
                }
            }
            let base_count = tokenizer.encode(text, false)
                .map(|e| e.len())
                .unwrap_or(200);
            base_count + 200
        };

        // Evict soft HCS before prefill
        let store = unsafe { &mut *(self.gpu_store_addr as *mut GpuDecodeStore) };
        let t_evict = Instant::now();
        let (evicted, _) = store.hcs_evict_for_prefill(estimated_tokens);
        let evict_ms = t_evict.elapsed().as_secs_f64() * 1000.0;

        // Prefill (Python, GIL held)
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

        // Reload soft HCS after prefill
        let t_reload = Instant::now();
        if evicted > 0 {
            store.hcs_reload_after_prefill_async();
        }
        let reload_ms = t_reload.elapsed().as_secs_f64() * 1000.0;

        // Decode (pure Rust, GIL held but unused by decode loop)
        let (decode_ms, decode_tokens, decode_tok_s) = if kv_overflow || max_new_tokens <= 1 {
            (0.0f64, 1usize, 0.0f64)
        } else {
            let decode_start = Instant::now();
            let mut count = 0usize;
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
            let elapsed = decode_start.elapsed().as_secs_f64();
            let total = count + 1; // includes first_token from prefill
            let tok_s = if elapsed > 0.0 && count > 0 {
                count as f64 / elapsed
            } else {
                0.0
            };
            (elapsed * 1000.0, total, tok_s)
        };

        // Cleanup
        self.py_model.call_method0(py, "server_cleanup")?;

        let prefill_tok_s = if prefill_ms > 0.0 {
            prompt_len as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };

        Ok(format!(
            r#"{{"prefill_ms":{:.1},"prefill_tok_s":{:.1},"prompt_tokens":{},"decode_ms":{:.1},"decode_tok_s":{:.2},"decode_tokens":{},"evict_ms":{:.1},"reload_ms":{:.1}}}"#,
            prefill_ms, prefill_tok_s, prompt_len,
            decode_ms, decode_tok_s, decode_tokens,
            evict_ms, reload_ms
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
