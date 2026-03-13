use krasis::chat_template::ChatTemplateEngine;

fn main() {
    let engine = ChatTemplateEngine::from_config(
        "/home/main/.krasis/models/Qwen3.5-122B-A10B/tokenizer_config.json"
    ).expect("Failed to load template");
    
    let messages = r#"[{"role": "user", "content": "Hello, how are you?"}]"#;
    match engine.apply(messages, true) {
        Ok(rendered) => {
            println!("SUCCESS: rendered {} chars", rendered.len());
            println!("First 200 chars: {}", &rendered[..rendered.len().min(200)]);
        }
        Err(e) => {
            println!("ERROR: {}", e);
        }
    }
}
