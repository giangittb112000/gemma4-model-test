"""Simple FastAPI service that parses Vietnamese e-commerce search queries.

Loads `google/gemma-4-e2b-it` from Hugging Face and returns a fixed JSON schema:
{category, product, brand, model, attributes}.

This serves the BASE model + prompt (no fine-tuning yet) to validate the
inference pipeline end-to-end.
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-e2b-it")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
REQUIRED_KEYS = ("category", "product", "brand", "model", "attributes")

SYSTEM_PROMPT = (
    "Bạn là bộ phân tích truy vấn tìm kiếm ecommerce tiếng Việt. "
    "Chuẩn hoá query (sửa lỗi chính tả, từ viết tắt, từ không dấu, từ viết dính) "
    "rồi trả về DUY NHẤT một JSON theo schema: "
    '{"category": string|null, "product": string|null, "brand": string|null, '
    '"model": string|null, "attributes": object}. '
    "Không tự bịa thông tin: trường thiếu để null, attributes thiếu để {}. "
    "Không giải thích gì thêm.\n\n"
    "Ví dụ:\n"
    'query: "dt ip 256" -> '
    '{"category": "điện thoại", "product": "iphone", "brand": "apple", '
    '"model": null, "attributes": {"storage": "256gb"}}'
)

STATE: dict[str, Any] = {"model": None, "tokenizer": None}


def _pick_dtype() -> "torch.dtype":
    """Choose the best dtype for the available hardware."""
    if torch.cuda.is_available():
        # Ampere+ (A100, RTX 30/40, L4...) hỗ trợ bf16; GPU cũ hơn (T4) dùng fp16.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # CPU: float32 thường nhanh hơn bf16 (bf16 bị emulate).
    return torch.float32


@asynccontextmanager
async def lifespan(_: FastAPI):
    use_cuda = torch.cuda.is_available()
    dtype = _pick_dtype()
    print(f"[startup] loading {MODEL_ID} (cuda={use_cuda}, dtype={dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model = model.to("cpu")
    model.eval()
    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    if use_cuda:
        STATE["device"] = f"cuda:{torch.cuda.get_device_name(0)}"
        print(f"[startup] model ready on GPU: {torch.cuda.get_device_name(0)}")
    else:
        STATE["device"] = "cpu"
        print("[startup] model ready on CPU")
    yield
    STATE.clear()


app = FastAPI(title="VN E-commerce Query Parser", lifespan=lifespan)


class ParseRequest(BaseModel):
    query: str


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first {...} block out of the generation and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in output: {text!r}")
    return json.loads(match.group(0))


def _normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    """Guarantee all required keys exist with sane defaults."""
    result = {k: parsed.get(k) for k in REQUIRED_KEYS}
    if not isinstance(result.get("attributes"), dict):
        result["attributes"] = {}
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    ready = STATE.get("model") is not None
    return {
        "status": "ok" if ready else "loading",
        "model": MODEL_ID,
        "device": STATE.get("device", "unknown"),
    }


@app.post("/parse")
def parse(req: ParseRequest) -> dict[str, Any]:
    model, tokenizer = STATE.get("model"), STATE.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model still loading")

    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nquery: \"{req.query}\""},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    generated = tokenizer.decode(
        output[0][input_len:], skip_special_tokens=True
    )

    try:
        parsed = _normalize(_extract_json(generated))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": str(exc), "raw": generated},
        ) from exc

    return {"query": req.query, "raw": generated, "parsed": parsed}
