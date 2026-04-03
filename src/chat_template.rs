//! Chat template engine — applies Jinja2 chat templates in Rust.
//!
//! Uses minijinja to render HuggingFace-style chat templates. Replaces
//! the Python `tokenizer.apply_chat_template()` call so that the entire
//! request path (tokenize → prefill → decode → detokenize) can happen
//! without Python in the per-request hot path except for prefill.

use serde_json;

/// A Jinja2 chat template engine for HuggingFace models.
pub struct ChatTemplateEngine {
    template_source: String,
    bos_token: String,
    eos_token: String,
}

// DeepSeek-V2/V2-Lite chat format (plain text User:/Assistant:)
const DEEPSEEK_CHAT_TEMPLATE: &str = concat!(
    "{% if not add_generation_prompt is defined %}",
    "{% set add_generation_prompt = false %}{% endif %}",
    "{{ bos_token }}",
    "{% for message in messages %}",
    "{% if message['role'] == 'user' %}",
    "{{ 'User: ' + message['content'] + '\n\n' }}",
    "{% elif message['role'] == 'assistant' %}",
    "{{ 'Assistant: ' + message['content'] + eos_token }}",
    "{% elif message['role'] == 'system' %}",
    "{{ message['content'] + '\n\n' }}",
    "{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "{{ 'Assistant:' }}",
    "{% endif %}",
);

impl ChatTemplateEngine {
    /// Load a chat template from tokenizer_config.json.
    ///
    /// Reads the `chat_template` field from the config file.
    /// Falls back to DeepSeek format if no template is found.
    pub fn from_config(tokenizer_config_path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(tokenizer_config_path)
            .map_err(|e| format!("Failed to read {}: {}", tokenizer_config_path, e))?;
        let config: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        // Extract chat_template — can be a string or a list of {name, template} objects
        let template_source = if let Some(ct) = config.get("chat_template") {
            match ct {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Array(arr) => {
                    // List of templates — prefer "default" or first one
                    let mut default_tmpl = None;
                    let mut first_tmpl = None;
                    for item in arr {
                        if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                            if let Some(tmpl) = item.get("template").and_then(|t| t.as_str()) {
                                if first_tmpl.is_none() {
                                    first_tmpl = Some(tmpl.to_string());
                                }
                                if name == "default" {
                                    default_tmpl = Some(tmpl.to_string());
                                }
                            }
                        }
                    }
                    default_tmpl.or(first_tmpl)
                        .unwrap_or_else(|| DEEPSEEK_CHAT_TEMPLATE.to_string())
                }
                _ => DEEPSEEK_CHAT_TEMPLATE.to_string(),
            }
        } else {
            log::info!("No chat_template in config — using DeepSeek format fallback");
            DEEPSEEK_CHAT_TEMPLATE.to_string()
        };

        // Extract bos_token and eos_token
        let bos_token = extract_token(&config, "bos_token").unwrap_or_default();
        let eos_token = extract_token(&config, "eos_token").unwrap_or_default();

        log::info!(
            "ChatTemplateEngine: loaded template ({} chars), bos={:?}, eos={:?}",
            template_source.len(), bos_token, eos_token
        );

        Ok(ChatTemplateEngine {
            template_source,
            bos_token,
            eos_token,
        })
    }

    /// Apply the chat template to a list of messages.
    ///
    /// `messages_json` is a JSON array of {role, content} objects.
    /// `tools_json` is an optional JSON array of tool definitions (OpenAI format).
    /// Returns the rendered text string ready for tokenization.
    pub fn apply(
        &self,
        messages_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        self.apply_with_tools(messages_json, "", add_generation_prompt, enable_thinking)
    }

    /// Apply with optional tools array for accurate token estimation.
    pub fn apply_with_tools(
        &self,
        messages_json: &str,
        tools_json: &str,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        let mut messages: serde_json::Value = serde_json::from_str(messages_json)
            .map_err(|e| format!("Failed to parse messages JSON: {}", e))?;
        let tools: serde_json::Value = if tools_json.is_empty() {
            serde_json::Value::Array(vec![])
        } else {
            serde_json::from_str(tools_json)
                .unwrap_or(serde_json::Value::Array(vec![]))
        };

        // Pre-process: convert string tool_call arguments to objects.
        // OpenAI format sends arguments as a JSON string, but Jinja templates
        // use `arguments|items` which requires a dict/mapping.
        if let Some(msgs) = messages.as_array_mut() {
            for msg in msgs.iter_mut() {
                if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                    for tc in tool_calls.iter_mut() {
                        // Determine which object holds arguments: tc.function or tc itself
                        let has_function = tc.get("function").is_some();
                        let fn_obj = if has_function {
                            tc.get_mut("function").unwrap()
                        } else {
                            &mut *tc
                        };
                        if let Some(args_str) = fn_obj.get("arguments").and_then(|v| v.as_str()).map(String::from) {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&args_str) {
                                fn_obj["arguments"] = parsed;
                            }
                        }
                    }
                }
            }
        }

        let mut env = minijinja::Environment::new();

        // Register the template
        env.add_template("chat", &self.template_source)
            .map_err(|e| format!("Failed to compile chat template: {}", e))?;

        // Add tojson filter (used by Qwen templates to serialize tool parameters)
        env.add_filter("tojson", |value: minijinja::Value| -> String {
            // Convert minijinja Value to serde_json Value for proper JSON serialization
            let json_val = minijinja_value_to_json(&value);
            serde_json::to_string(&json_val).unwrap_or_else(|_| value.to_string())
        });

        // Add raise_exception function (used by some templates)
        env.add_function("raise_exception", raise_exception);

        // Add strftime_now function (used by some templates)
        env.add_function("strftime_now", strftime_now);

        // Handle Python string methods used by HuggingFace templates
        env.set_unknown_method_callback(|_state, value, method, args| {
            match method {
                "startswith" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "startswith requires a string")
                    })?;
                    let prefix = args.first()
                        .and_then(|a| a.as_str())
                        .ok_or_else(|| {
                            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "startswith requires a string argument")
                        })?;
                    Ok(minijinja::Value::from(s.starts_with(prefix)))
                }
                "endswith" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "endswith requires a string")
                    })?;
                    let suffix = args.first()
                        .and_then(|a| a.as_str())
                        .ok_or_else(|| {
                            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "endswith requires a string argument")
                        })?;
                    Ok(minijinja::Value::from(s.ends_with(suffix)))
                }
                "strip" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "strip requires a string")
                    })?;
                    Ok(minijinja::Value::from(s.trim()))
                }
                "lstrip" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "lstrip requires a string")
                    })?;
                    let chars = args.first().and_then(|a| a.as_str());
                    Ok(minijinja::Value::from(match chars {
                        Some(c) => s.trim_start_matches(|ch: char| c.contains(ch)),
                        None => s.trim_start(),
                    }))
                }
                "rstrip" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "rstrip requires a string")
                    })?;
                    let chars = args.first().and_then(|a| a.as_str());
                    Ok(minijinja::Value::from(match chars {
                        Some(c) => s.trim_end_matches(|ch: char| c.contains(ch)),
                        None => s.trim_end(),
                    }))
                }
                "split" => {
                    let s = value.as_str().ok_or_else(|| {
                        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "split requires a string")
                    })?;
                    let sep = args.first().and_then(|a| a.as_str());
                    let parts: Vec<minijinja::Value> = match sep {
                        Some(sep) => s.split(sep).map(minijinja::Value::from).collect(),
                        None => s.split_whitespace().map(minijinja::Value::from).collect(),
                    };
                    Ok(minijinja::Value::from(parts))
                }
                _ => Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("unknown method: {}", method),
                ))
            }
        });

        let tmpl = env.get_template("chat")
            .map_err(|e| format!("Failed to get template: {}", e))?;

        let ctx = minijinja::context! {
            messages => messages,
            tools => tools,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
        };

        tmpl.render(ctx)
            .map_err(|e| format!("Template render failed: {}", e))
    }
}

/// Extract a token string from tokenizer_config.json.
/// Handles both `"bos_token": "<s>"` and `"bos_token": {"content": "<s>", ...}`.
fn extract_token(config: &serde_json::Value, key: &str) -> Option<String> {
    match config.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(obj) => {
            obj.get("content").and_then(|v| v.as_str()).map(String::from)
        }
        _ => None,
    }
}

/// raise_exception function for Jinja2 templates.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// strftime_now function for Jinja2 templates (returns current date/time).
fn strftime_now(fmt: String) -> String {
    // Simple implementation covering common format strings
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // For the typical use case (%Y-%m-%d), we do a basic calculation
    // Full strftime is overkill — most templates just use %Y-%m-%d or similar
    if fmt.contains("%Y") || fmt.contains("%m") || fmt.contains("%d") {
        // Days since epoch
        let days = now / 86400;
        let (year, month, day) = days_to_ymd(days as i64);
        fmt.replace("%Y", &format!("{:04}", year))
            .replace("%m", &format!("{:02}", month))
            .replace("%d", &format!("{:02}", day))
    } else {
        format!("{}", now)
    }
}

/// Convert a minijinja Value to a serde_json Value for JSON serialization.
fn minijinja_value_to_json(value: &minijinja::Value) -> serde_json::Value {
    if value.is_none() || value.is_undefined() {
        serde_json::Value::Null
    } else if let Some(b) = value.as_str() {
        serde_json::Value::String(b.to_string())
    } else if let Ok(b) = bool::try_from(value.clone()) {
        serde_json::Value::Bool(b)
    } else if let Ok(n) = i64::try_from(value.clone()) {
        serde_json::json!(n)
    } else if let Ok(n) = f64::try_from(value.clone()) {
        serde_json::json!(n)
    } else if value.kind() == minijinja::value::ValueKind::Seq {
        let items: Vec<serde_json::Value> = value.try_iter()
            .map(|iter| iter.map(|v| minijinja_value_to_json(&v)).collect())
            .unwrap_or_default();
        serde_json::Value::Array(items)
    } else if value.kind() == minijinja::value::ValueKind::Map {
        let mut map = serde_json::Map::new();
        if let Ok(keys) = value.try_iter() {
            for key in keys {
                let key_str = key.to_string();
                if let Ok(val) = value.get_item(&key) {
                    map.insert(key_str, minijinja_value_to_json(&val));
                }
            }
        }
        serde_json::Value::Object(map)
    } else {
        // Fallback: use the display representation
        serde_json::Value::String(value.to_string())
    }
}

/// Convert days since epoch to (year, month, day).
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
