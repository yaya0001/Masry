import os
import json
import time
import threading
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "./llama3-3b-finetuned")
MODEL_ID   = "masry-llama3"

app = FastAPI(title="Masry API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model      = None
_tokenizer  = None
_load_error = None

def load_model_once():
    global _model, _tokenizer, _load_error
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        has_gpu = torch.cuda.is_available()
        device  = "cuda" if has_gpu else "cpu"
        print(f"[Masry] Device: {device.upper()}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if has_gpu else torch.float32,
            device_map=device,
        )
        print("[Masry] Model loaded successfully")

    except Exception as exc:
        _load_error = str(exc)
        print(f"[Masry] Failed to load model: {exc}")


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_model_once, daemon=True).start()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

def build_prompt(messages: List[Message]) -> str:
    context_parts = []
    for msg in messages:
        if msg.role == "system":
            context_parts.append(f"[System]: {msg.content}")
        elif msg.role == "user":
            context_parts.append(f"[User]: {msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"[Assistant]: {msg.content}")

    context       = "\n".join(context_parts[:-1])
    last_user_msg = messages[-1].content

    full_instruction = f"{context}\n\n{last_user_msg}" if context else last_user_msg
    return f"### Instruction:\n{full_instruction}\n\n### Response:\n"


def run_generation(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    import torch
    from transformers import TextIteratorStreamer

    inputs   = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
    )
    threading.Thread(target=_model.generate, kwargs=gen_kwargs).start()
    return streamer

@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": _model is not None, "error": _load_error}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "masry"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _model is None:
        detail = f"Model failed to load: {_load_error}" if _load_error else "Model is still loading, please wait..."
        raise HTTPException(status_code=503, detail=detail)

    prompt  = build_prompt(req.messages)
    req_id  = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    if req.stream:
        async def stream_generator():
            streamer = run_generation(prompt, req.max_tokens or 512, req.temperature or 0.7, req.top_p or 0.9)
            for token in streamer:
                if not token:
                    continue
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)

            final_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    streamer  = run_generation(prompt, req.max_tokens or 512, req.temperature or 0.7, req.top_p or 0.9)
    full_text = "".join(streamer)

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }
