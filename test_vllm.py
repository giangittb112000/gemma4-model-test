#!/usr/bin/env python3
"""Gọi trực tiếp API vLLM (/v1/chat/completions) với JSON schema.

Cách dùng:
  python3 test_vllm.py                  # chạy bộ query mặc định
  python3 test_vllm.py "dt ip 256"      # test 1 query
  python3 test_vllm.py "q1" "q2"        # test nhiều query
  make test Q="dt ip 256"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

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

# Prompt rút từ search_report.csv + synony.json + soft-intent (pin khủng, sinh viên...).
PROMPT = """\
Bạn là bộ phân tích truy vấn tìm kiếm ecommerce tiếng Việt (điện thoại, tablet, laptop, phụ kiện).
Nhiệm vụ: đọc query người dùng (thường viết tắt / không dấu / viết dính / sai chính tả / mô tả nhu cầu) \
rồi trả về DUY NHẤT một JSON đúng schema sau, không giải thích thêm:
{"category": string|null, "product": string|null, "brand": string|null, "model": string|null, "attributes": object}

Quy tắc:
1) Chuẩn hoá alias phổ biến:
   - dt, đt → điện thoại
   - ip, iph, iphon, i phone → iphone
   - ss, sámung → samsung
   - mi → xiaomi
   - airpod, air pod → airpods
   - prm, promax, prom → pro max
   - 128/128g, 256/256g, 512/512g → 128gb, 256gb, 512gb
2) category = ngành hàng rõ (điện thoại, tablet, laptop, tai nghe, phụ kiện...).
3) product = dòng sản phẩm; brand = hãng; model chỉ khi có đời máy rõ.
4) attributes: key viết thường. Gồm cả thông số tường minh và nhu cầu suy ra được.
5) Soft-intent (nhu cầu mô tả) → map sang attributes có ngưỡng rõ:
   - pin khủng / pin trâu / pin lâu → battery_mah_min = 5000
   - cho sinh viên / giá rẻ / giá thấp / tầm trung thấp → price_max = 20000000
   - chơi game / gaming → usage = gaming
   - chụp đẹp / camera đẹp → camera = good
   - mỏng nhẹ → form_factor = thin_light
6) Chỉ điền thông tin suy ra được từ query. Thiếu → null / {}. Không bịa brand/product/model.

Ví dụ:
query: "dt ip 256"
→ {"category":"điện thoại","product":"iphone","brand":"apple","model":null,"attributes":{"storage":"256gb"}}

query: "ip 16 promax"
→ {"category":"điện thoại","product":"iphone","brand":"apple","model":"16 pro max","attributes":{}}

query: "điện thoại pin khủng"
→ {"category":"điện thoại","product":null,"brand":null,"model":null,"attributes":{"battery_mah_min":5000}}

query: "điện thoại cho sinh viên"
→ {"category":"điện thoại","product":null,"brand":null,"model":null,"attributes":{"price_max":20000000}}

query: "dt pin trâu dưới 10 triệu"
→ {"category":"điện thoại","product":null,"brand":null,"model":null,"attributes":{"battery_mah_min":5000,"price_max":10000000}}

query: "tai nghe"
→ {"category":"tai nghe","product":null,"brand":null,"model":null,"attributes":{}}
"""

DEFAULT_QUERIES = [
    "dt ip 256",
    "ip 16 promax",
    "ss s25ultra 512",
    "điện thoại pin khủng",
    "điện thoại cho sinh viên",
    "dt pin trâu dưới 10 triệu",
    "laptop mỏng nhẹ cho sinh viên",
    "tai nghe bluetooth",
]

LINE = "─" * 64


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
                    print(f"✓  vLLM ready  →  {VLLM}")
                    return
        except Exception:
            pass
        print("…  waiting for vLLM ...")
        time.sleep(5)
    raise SystemExit(f"✗  vLLM not ready at {VLLM}")


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


def pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return raw


def print_result(index: int, total: int, query: str, raw: str, ms: float) -> None:
    print(LINE)
    print(f"[{index}/{total}]  query     : {query}")
    print(f"         latency    : {ms:,.0f} ms  ({ms / 1000:.2f} s)")
    print("         result     :")
    for line in pretty_json(raw).splitlines():
        print(f"           {line}")


def print_fail(index: int, total: int, query: str, err: str) -> None:
    print(LINE)
    print(f"[{index}/{total}]  query     : {query}")
    print(f"         status     : FAIL")
    print(f"         error      : {err}")


def run_queries(queries: list[str]) -> int:
    print(LINE)
    print(f" model   : {MODEL}")
    print(f" endpoint: {VLLM}/v1/chat/completions")
    print(f" queries : {len(queries)}")
    print(LINE)

    ok = 0
    times: list[float] = []
    total = len(queries)

    for i, query in enumerate(queries, 1):
        start = time.time()
        try:
            raw = parse(query)
            ms = (time.time() - start) * 1000
            times.append(ms)
            print_result(i, total, query, raw, ms)
            ok += 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            print_fail(i, total, query, f"HTTP {exc.code}: {body}")
        except Exception as exc:  # noqa: BLE001
            print_fail(i, total, query, str(exc))

    print(LINE)
    if times:
        avg = sum(times) / len(times)
        print(
            f" summary : {ok}/{total} ok"
            f"  |  avg {avg:,.0f} ms"
            f"  |  min {min(times):,.0f} ms"
            f"  |  max {max(times):,.0f} ms"
        )
    else:
        print(f" summary : {ok}/{total} ok")
    print(LINE)
    return 0 if ok == total else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test vLLM query parser (JSON schema)."
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help='Query tùy chọn, ví dụ: "dt ip 256". Không truyền thì chạy bộ mặc định.',
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Không chờ /health (giả định vLLM đã sẵn sàng).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    queries = args.queries or DEFAULT_QUERIES

    if not args.no_wait:
        wait_ready()

    return run_queries(queries)


if __name__ == "__main__":
    raise SystemExit(main())
