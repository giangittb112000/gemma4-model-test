#!/usr/bin/env python3
"""Gọi vLLM /v1/chat/completions + JSON schema.

Đo latency chuẩn cho perf search:
  - model_ms  = TTFT + generation (thời gian engine sinh đủ response)
  - e2e_ms    = client đo (gồm mạng)
  - PERF {...} JSON 1 dòng / request — grep/parse sau này

  make test Q="iphoooen 17 256"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

VLLM = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
# Serve: base + LoRA (train = QLoRA). Test chỉ gọi adapter.
DEFAULT_MODEL = os.environ.get("MODEL_ID", "query-parser-ft")

# Cùng prompt với finetune/prompt.py (train ↔ serve).
sys.path.insert(0, str(Path(__file__).resolve().parent / "finetune"))
from prompt import SYSTEM_PROMPT  # noqa: E402

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
# JSON parse ~30–50 token; 64 đủ, cắt decode thừa.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))


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


def chat_complete(query: str, model: str) -> tuple[str, dict[str, Any], float]:
    """Trả (content, api_json, e2e_ms). e2e = client perf_counter đến khi nhận đủ body."""
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'query: "{query}"'},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "query_parse", "schema": SCHEMA},
        },
        # vLLM với --enable-per-request-metrics → field metrics trong body.
        "include_metrics": True,
    }
    t0 = time.perf_counter()
    try:
        result = post("/v1/chat/completions", body)
    except urllib.error.HTTPError as exc:
        # Image vLLM cũ: bỏ include_metrics rồi thử lại.
        if exc.code in (400, 422) and "include_metrics" in body:
            err = exc.read().decode(errors="replace")
            if "include_metrics" in err or "metrics" in err.lower():
                body.pop("include_metrics", None)
                t0 = time.perf_counter()
                result = post("/v1/chat/completions", body)
            else:
                raise
        else:
            raise
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    content = result["choices"][0]["message"]["content"]
    return content, result, e2e_ms


def extract_perf(api: dict[str, Any], e2e_ms: float) -> dict[str, Any]:
    """Chuẩn hoá metrics để log / tính perf search.

    model_ms = TTFT + generation = thời gian engine từ lúc schedule đến token cuối
    (không gồm queue; không gồm RTT mạng). Đây là số chính để đánh giá model.
    """
    usage = api.get("usage") or {}
    metrics = api.get("metrics") or {}

    ttft = metrics.get("time_to_first_token_ms")
    gen = metrics.get("generation_time_ms")
    queue = metrics.get("queue_time_ms")
    tps = metrics.get("tokens_per_second")
    mean_itl = metrics.get("mean_itl_ms")

    model_ms: float | None = None
    if isinstance(ttft, (int, float)) and isinstance(gen, (int, float)):
        model_ms = float(ttft) + float(gen)
    elif isinstance(ttft, (int, float)) and gen is None:
        # 1 token / metrics thiếu generation: dùng TTFT ≈ toàn bộ
        model_ms = float(ttft)

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    return {
        "model_ms": model_ms,
        "e2e_ms": round(e2e_ms, 2),
        "ttft_ms": ttft,
        "generation_ms": gen,
        "queue_ms": queue,
        "mean_itl_ms": mean_itl,
        "tokens_per_second": tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "metrics_source": "vllm" if metrics else "client_e2e_only",
    }


def pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return raw


def print_result(
    index: int,
    total: int,
    query: str,
    model: str,
    raw: str,
    perf: dict[str, Any],
) -> None:
    print(LINE)
    print(f"[{index}/{total}]  query     : {query}")
    print(f"         model      : {model}")
    # Số chính cho search SLA / so model
    if perf["model_ms"] is not None:
        print(
            f"         model_ms   : {perf['model_ms']:,.1f} ms"
            f"  (ttft={perf['ttft_ms']} + gen={perf['generation_ms']})"
        )
    else:
        print(
            "         model_ms   : n/a"
            "  (vLLM chưa trả metrics — restart với --enable-per-request-metrics)"
        )
    print(f"         e2e_ms     : {perf['e2e_ms']:,.1f} ms  (client, gồm mạng)")
    if perf.get("queue_ms") is not None:
        print(f"         queue_ms   : {perf['queue_ms']}")
    if perf.get("tokens_per_second") is not None:
        print(f"         tok/s      : {perf['tokens_per_second']}")
    print(
        f"         tokens     : prompt={perf.get('prompt_tokens')} "
        f"completion={perf.get('completion_tokens')}"
    )
    print("         result     :")
    for line in pretty_json(raw).splitlines():
        print(f"           {line}")
    # 1 dòng machine-readable — grep '^PERF '
    print(
        "PERF "
        + json.dumps(
            {"query": query, "model": model, **perf},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def print_fail(index: int, total: int, query: str, model: str, err: str) -> None:
    print(LINE)
    print(f"[{index}/{total}]  query     : {query}")
    print(f"         model      : {model}")
    print(f"         status     : FAIL")
    print(f"         error      : {err}")


def warmup(models: list[str]) -> None:
    print(LINE)
    print(" warmup  : 1 request/model (không tính vào summary)")
    for model in models:
        try:
            _, api, e2e_ms = chat_complete("__warmup__", model)
            perf = extract_perf(api, e2e_ms)
            label = (
                f"model_ms={perf['model_ms']:.1f}"
                if perf["model_ms"] is not None
                else f"e2e_ms={perf['e2e_ms']:.1f}"
            )
            print(f"         {model}: cold {label}")
        except Exception as exc:  # noqa: BLE001
            print(f"         {model}: warmup fail ({exc})", file=sys.stderr)


def run_queries(queries: list[str], models: list[str], do_warmup: bool) -> int:
    print(LINE)
    print(f" models  : {', '.join(models)}")
    print(f" endpoint: {VLLM}/v1/chat/completions")
    print(f" queries : {len(queries)}")
    print(" metric  : model_ms = vLLM TTFT+generation (chuẩn perf search)")
    print(LINE)

    if do_warmup:
        warmup(models)

    ok = 0
    model_times: list[float] = []
    e2e_times: list[float] = []
    total = len(queries) * len(models)
    n = 0

    for query in queries:
        for model in models:
            n += 1
            try:
                raw, api, e2e_ms = chat_complete(query, model)
                perf = extract_perf(api, e2e_ms)
                e2e_times.append(float(perf["e2e_ms"]))
                if perf["model_ms"] is not None:
                    model_times.append(float(perf["model_ms"]))
                print_result(n, total, query, model, raw, perf)
                ok += 1
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                print_fail(n, total, query, model, f"HTTP {exc.code}: {body}")
            except Exception as exc:  # noqa: BLE001
                print_fail(n, total, query, model, str(exc))

    print(LINE)
    print(f" summary : {ok}/{total} ok")
    if model_times:
        print(
            f" model_ms: avg {sum(model_times)/len(model_times):,.1f}"
            f"  |  min {min(model_times):,.1f}"
            f"  |  max {max(model_times):,.1f}"
            f"  |  target <2000"
        )
    else:
        print(" model_ms: (không có — dùng e2e_ms tạm thời)")
    if e2e_times:
        print(
            f" e2e_ms  : avg {sum(e2e_times)/len(e2e_times):,.1f}"
            f"  |  min {min(e2e_times):,.1f}"
            f"  |  max {max(e2e_times):,.1f}"
        )
    print(LINE)
    return 0 if ok == total else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test vLLM query parser (JSON schema)."
    )
    parser.add_argument("queries", nargs="*", help='Query, ví dụ: "dt ip 256"')
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Tên trên /v1/models (mặc định: {DEFAULT_MODEL})",
    )
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    queries = args.queries or DEFAULT_QUERIES

    if not args.no_wait:
        wait_ready()

    return run_queries(queries, [args.model], do_warmup=not args.no_warmup)


if __name__ == "__main__":
    raise SystemExit(main())
