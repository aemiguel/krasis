"""FastAPI HTTP server — OpenAI-compatible /v1/chat/completions.

Supports streaming (SSE) and blocking responses.

Usage:
    python -m krasis.server --model-path /path/to/Kimi-K2.5 --pp-partition 31,30
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from krasis.model import KrasisModel
from krasis.scheduler import GenerationRequest, Scheduler

logger = logging.getLogger("krasis.server")

app = FastAPI(title="Krasis", version="0.1.0")
_scheduler: Optional[Scheduler] = None
_model: Optional[KrasisModel] = None
_model_name: str = "unknown"


@app.get("/health")
async def health():
    if _model is None or not _model._loaded:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "owned_by": "krasis",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 256)
    temperature = body.get("temperature", 0.6)
    top_k = body.get("top_k", 50)
    top_p = body.get("top_p", 0.95)

    # Tokenize
    prompt_tokens = _model.tokenizer.apply_chat_template(messages)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    stop_ids = [_model.cfg.eos_token_id]
    # Handle custom stop tokens
    if "stop" in body:
        stop = body["stop"]
        if isinstance(stop, str):
            stop = [stop]
        for s in stop:
            ids = _model.tokenizer.encode(s, add_special_tokens=False)
            stop_ids.extend(ids)

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=stop_ids,
    )

    logger.info(
        "Request %s: %d prompt tokens, max_new=%d, stream=%s",
        request_id, len(prompt_tokens), max_tokens, stream,
    )

    if stream:
        return StreamingResponse(
            _stream_response(gen_request),
            media_type="text/event-stream",
        )
    else:
        return await _blocking_response(gen_request)


async def _stream_response(request: GenerationRequest):
    """SSE streaming response."""
    created = int(time.time())

    async for output in _scheduler.generate_stream(request):
        chunk = {
            "id": request.request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": output.text} if output.finish_reason is None else {},
                "finish_reason": output.finish_reason,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


async def _blocking_response(request: GenerationRequest):
    """Non-streaming response — collect all tokens then return."""
    created = int(time.time())
    chunks = []
    finish_reason = None

    async for output in _scheduler.generate_stream(request):
        chunks.append(output.text)
        if output.finish_reason:
            finish_reason = output.finish_reason

    full_text = "".join(chunks)

    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": finish_reason or "stop",
        }],
        "usage": {
            "prompt_tokens": len(request.prompt_tokens),
            "completion_tokens": len(chunks),
            "total_tokens": len(request.prompt_tokens) + len(chunks),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Krasis standalone LLM server")
    parser.add_argument("--model-path", required=True, help="Path to HF model")
    parser.add_argument("--pp-partition", default=None,
                        help="Comma-separated layer counts per GPU (e.g. 31,30)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--krasis-threads", type=int, default=16,
                        help="CPU threads for expert computation")
    parser.add_argument("--kv-dtype", default="bf16",
                        choices=["fp8_e4m3", "bf16"])
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    global _model, _scheduler, _model_name
    import torch

    pp_partition = None
    if args.pp_partition:
        pp_partition = [int(x.strip()) for x in args.pp_partition.split(",")]

    kv_dtype = torch.float8_e4m3fn if args.kv_dtype == "fp8_e4m3" else torch.bfloat16

    _model_name = args.model_path.rstrip("/").split("/")[-1]

    _model = KrasisModel(
        model_path=args.model_path,
        pp_partition=pp_partition,
        num_gpus=args.num_gpus,
        kv_dtype=kv_dtype,
        krasis_threads=args.krasis_threads,
    )

    logger.info("Loading model...")
    _model.load()
    logger.info("Model loaded, starting server on %s:%d", args.host, args.port)

    _scheduler = Scheduler(_model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
