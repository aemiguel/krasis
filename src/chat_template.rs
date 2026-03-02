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
    /// Returns the rendered text string ready for tokenization.
    pub fn apply(&self, messages_json: &str, add_generation_prompt: bool) -> Result<String, String> {
        let messages: serde_json::Value = serde_json::from_str(messages_json)
            .map_err(|e| format!("Failed to parse messages JSON: {}", e))?;

        let mut env = minijinja::Environment::new();

        // Register the template
        env.add_template("chat", &self.template_source)
            .map_err(|e| format!("Failed to compile chat template: {}", e))?;

        // Add raise_exception function (used by some templates)
        env.add_function("raise_exception", raise_exception);

        // Add strftime_now function (used by some templates)
        env.add_function("strftime_now", strftime_now);

        let tmpl = env.get_template("chat")
            .map_err(|e| format!("Failed to get template: {}", e))?;

        let ctx = minijinja::context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => add_generation_prompt,
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
