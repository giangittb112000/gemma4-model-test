#!/usr/bin/env python3
"""Gọi trực tiếp API vLLM (/v1/chat/completions) với JSON schema."""

import json
import os
import sys
import time
import urllib.error
import urllib.request

VLLM = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL_ID", "google/gemma-4-e2b-it")

SCHEMA = {
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

PROMPT = (
    "Bạn là bộ phân tích truy vấn tìm kiếm ecommerce tiếng Việt. "
    "Chuẩn hoá query (sửa lỗi chính tả, viết tắt, không dấu, viết dính) "
    "rồi trả về DUY NHẤT một JSON theo schema cố định. "
    "Thiếu thông tin thì null / {}. Không giải thích.\n\n"
    'Ví dụ: query "dt ip 256" -> '
    '{"category":"điện thoại","product":"iphone","brand":"apple",'
    '"model":null,"attributes":{"storage":"256gb"}}'
)

QUERIES = [
    "dt ip 256",
    "laptop dell i5 16gb",
    "ss s24 ultra 512",
    "tai nghe bluetooth",
    "ao thun nam mau den",
]


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM}{path}",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def wait_ready(timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{VLLM}/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"[ok] vLLM ready at {VLLM}")
                    return
        except Exception:
            pass
        print("[..] waiting for vLLM ...")
        time.sleep(5)
    raise SystemExit(f"[fail] vLLM not ready at {VLLM}")


def parse(query: str) -> str:
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 128,
        "messages": [
            {"role": "user", "content": f'{PROMPT}\n\nquery: "{query}"'},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "query_parse", "schema": SCHEMA},
        },
    }
    result = post("/v1/chat/completions", body)
    return result["choices"][0]["message"]["content"]


def main() -> int:
    wait = "--no-wait" not in sys.argv
    if wait:
        wait_ready()

    for query in QUERIES:
        start = time.time()
        try:
            raw = parse(query)
            ms = (time.time() - start) * 1000
            print(f"[ok] ({ms:6.0f} ms) query={query!r}")
            print(f"     raw={raw}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            print(f"[fail] query={query!r} http={exc.code} {body}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] query={query!r} {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
