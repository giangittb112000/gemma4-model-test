"""FastAPI wrapper phân tích truy vấn tìm kiếm ecommerce tiếng Việt.

Không tự load model — gọi tới một vLLM server (OpenAI-compatible) và bật
JSON schema (guided decoding) để output LUÔN đúng khuôn:
{category, product, brand, model, attributes}.

vLLM lo phần chạy model (nhanh, batching). Service này chỉ lo:
- dựng prompt,
- gọi vLLM với response_format = json_schema,
- chuẩn hoá và trả kết quả.
"""

import json
import os
import re
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000").rstrip("/")
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-e2b-it")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
REQUIRED_KEYS = ("category", "product", "brand", "model", "attributes")

# JSON schema ép output (vLLM guided decoding -> JSON valid 100%).
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": ["string", "null"]},
        "product": {"type": ["string", "null"]},
        "brand": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"]},
        "attributes": {"type": "object"},
    },
    "required": ["category", "product", "brand", "model", "attributes"],
    "additionalProperties": False,
}

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

app = FastAPI(title="VN E-commerce Query Parser (vLLM)")


class ParseRequest(BaseModel):
    query: str


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in output: {text!r}")
    return json.loads(match.group(0))


def _normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    result = {k: parsed.get(k) for k in REQUIRED_KEYS}
    if not isinstance(result.get("attributes"), dict):
        result["attributes"] = {}
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        resp = requests.get(f"{VLLM_URL}/health", timeout=5)
        ready = resp.status_code == 200
    except requests.RequestException:
        ready = False
    return {"status": "ok" if ready else "loading", "model": MODEL_ID, "vllm": VLLM_URL}


@app.post("/parse")
def parse(req: ParseRequest) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": [
            # Gemma không hỗ trợ role "system" riêng -> gộp vào user.
            {"role": "user", "content": f'{SYSTEM_PROMPT}\n\nquery: "{req.query}"'},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "query_parse", "schema": OUTPUT_SCHEMA},
        },
    }
    try:
        resp = requests.post(
            f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=120
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"vLLM error: {exc}") from exc

    try:
        parsed = _normalize(_extract_json(content))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail={"error": str(exc), "raw": content}
        ) from exc

    return {"query": req.query, "raw": content, "parsed": parsed}
