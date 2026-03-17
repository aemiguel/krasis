#!/bin/bash
# Test multi-turn conversations against llama.cpp server
# Reproduces the exact conversations that triggered bugs in Krasis

PORT=8080
URL="http://localhost:${PORT}/v1/chat/completions"

# Helper function to send a request and extract the response
send_msg() {
    local messages_json="$1"
    local response
    response=$(curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"qwen3.5-35b\",
            \"messages\": $messages_json,
            \"temperature\": 0.6,
            \"max_tokens\": 512,
            \"stream\": false
        }" 2>&1)

    local content
    content=$(echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d['choices'][0]['message']['content']
    # Count thinking vs answer tokens
    think_end = c.find('</think>')
    if think_end >= 0:
        think_part = c[:think_end]
        answer_part = c[think_end+8:].strip()
    elif '<think>' in c:
        think_part = c
        answer_part = ''
    else:
        think_part = ''
        answer_part = c.strip()
    print(f'THINK_LEN={len(think_part)} ANSWER_LEN={len(answer_part)}')
    print(f'ANSWER: {answer_part[:200]}')
except Exception as e:
    print(f'ERROR: {e}')
    print(sys.stdin.read()[:500] if hasattr(sys.stdin, 'read') else '')
" 2>&1)
    echo "$content"

    # Return the full content for building history
    echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except:
    print('')
" 2>&1
}

echo "=== Conversation 1: Paris (5 turns) ==="
echo ""

# Turn 1
echo "--- Turn 1: What is the capital of France? ---"
RESP1=$(curl -s "$URL" -H "Content-Type: application/json" -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.6, "max_tokens": 256, "stream": false
}')
CONTENT1=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
echo "$CONTENT1" | head -3
echo ""

# Turn 2
echo "--- Turn 2: What is the population of that city? ---"
MSGS2=$(python3 -c "
import json
msgs = [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': '''$CONTENT1'''},
    {'role': 'user', 'content': 'What is the population of that city?'}
]
print(json.dumps(msgs))
")
RESP2=$(curl -s "$URL" -H "Content-Type: application/json" -d "{
    \"model\": \"qwen3.5-35b\",
    \"messages\": $MSGS2,
    \"temperature\": 0.6, \"max_tokens\": 256, \"stream\": false
}")
CONTENT2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
echo "$CONTENT2" | head -3
echo ""

# Turn 3 - This is where Krasis failed
echo "--- Turn 3: Name three famous landmarks there ---"
MSGS3=$(python3 -c "
import json
msgs = [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': '''$CONTENT1'''},
    {'role': 'user', 'content': 'What is the population of that city?'},
    {'role': 'assistant', 'content': '''$CONTENT2'''},
    {'role': 'user', 'content': 'Name three famous landmarks there'}
]
print(json.dumps(msgs))
")
RESP3=$(curl -s "$URL" -H "Content-Type: application/json" -d "{
    \"model\": \"qwen3.5-35b\",
    \"messages\": $MSGS3,
    \"temperature\": 0.6, \"max_tokens\": 256, \"stream\": false
}")
CONTENT3=$(echo "$RESP3" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
echo "$CONTENT3" | head -5
echo ""

echo "=== Conversation 2: Math (6 turns) ==="
echo ""

# Turn 1
echo "--- Turn 1: What is 15 multiplied by 7? ---"
MRESP1=$(curl -s "$URL" -H "Content-Type: application/json" -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "What is 15 multiplied by 7?"}],
    "temperature": 0.6, "max_tokens": 256, "stream": false
}')
MCONTENT1=$(echo "$MRESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
echo "$MCONTENT1" | head -3
echo ""

echo "Done. If all turns produced valid answers, the model works correctly for multi-turn."
