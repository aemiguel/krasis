use krasis::chat_template::ChatTemplateEngine;

#[test]
fn qwen35_enable_thinking_false_multiturn_render_matches_expected() {
    let engine = ChatTemplateEngine::from_config(
        "/home/main/.krasis/models/Qwen3.5-35B-A3B/tokenizer_config.json",
    )
    .expect("Failed to load template");

    let messages = r#"[{"role":"user","content":"What is 7 times 8?"},{"role":"assistant","content":"56"},{"role":"user","content":"Now multiply that by 10"}]"#;

    let rendered = engine
        .apply(messages, true, false)
        .expect("Template render should succeed");

    let expected = concat!(
        "<|im_start|>user\n",
        "What is 7 times 8?<|im_end|>\n",
        "<|im_start|>assistant\n",
        "56<|im_end|>\n",
        "<|im_start|>user\n",
        "Now multiply that by 10<|im_end|>\n",
        "<|im_start|>assistant\n",
        "<think>\n\n</think>\n\n",
    );

    assert_eq!(rendered, expected);
}
