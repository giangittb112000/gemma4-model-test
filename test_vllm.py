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


def _hdr_float(headers: dict[str, str], *names: str) -> float | None:
    for name in names:
        raw = headers.get(name) or headers.get(name.lower())
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def post(path: str, payload: dict) -> tuple[dict, dict[str, str]]:
    """Trả (json_body, response_headers lowercase)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM}{path}",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return body, headers


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


def chat_complete(
    query: str, model: str
) -> tuple[str, dict[str, Any], dict[str, str], float]:
    """Trả (content, api_json, headers, e2e_ms)."""
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
        # Body metrics (cần server --enable-per-request-metrics).
        "include_metrics": True,
    }
    t0 = time.perf_counter()
    try:
        result, headers = post("/v1/chat/completions", body)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 422) and "include_metrics" in body:
            err = exc.read().decode(errors="replace")
            if "include_metrics" in err or "metrics" in err.lower():
                body.pop("include_metrics", None)
                t0 = time.perf_counter()
                result, headers = post("/v1/chat/completions", body)
            else:
                raise
        else:
            raise
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    content = result["choices"][0]["message"]["content"]
    return content, result, headers, e2e_ms


def extract_perf(
    api: dict[str, Any], headers: dict[str, str], e2e_ms: float
) -> dict[str, Any]:
    """Gộp metrics từ body + header x-vllm-* + ước lượng client.

    e2e_ms   = đồng hồ client: gửi request → nhận đủ response (gồm mạng/proxy).
    model_ms = thời gian engine (TTFT+decode hoặc inference header) — không gồm mạng.
    """
    usage = api.get("usage") or {}
    metrics = api.get("metrics") or {}
    sources: list[str] = ["client_e2e"]

    # --- body metrics (include_metrics) ---
    ttft = metrics.get("time_to_first_token_ms")
    gen = metrics.get("generation_time_ms")
    queue = metrics.get("queue_time_ms")
    tps = metrics.get("tokens_per_second")
    mean_itl = metrics.get("mean_itl_ms")
    if metrics:
        sources.append("body_metrics")

    # --- HTTP headers (enable-request-stats-headers) ---
    # Hỗ trợ cả x-vllm-* và x-* (tuỳ version).
    h_total = _hdr_float(headers, "x-vllm-total-time", "x-total-time")
    h_queue = _hdr_float(headers, "x-vllm-queue-time", "x-queue-time")
    h_prefill = _hdr_float(headers, "x-vllm-prefill-time", "x-prefill-time")
    h_decode = _hdr_float(headers, "x-vllm-decode-time", "x-decode-time")
    h_infer = _hdr_float(headers, "x-vllm-inference-time", "x-inference-time")
    h_tpot = _hdr_float(
        headers, "x-vllm-time-per-output-token", "x-time-per-output-token"
    )
    h_cached = _hdr_float(headers, "x-vllm-cached-tokens", "x-cached-tokens")
    h_prompt = _hdr_float(headers, "x-vllm-prompt-tokens", "x-prompt-tokens")
    h_completion = _hdr_float(
        headers, "x-vllm-completion-tokens", "x-completion-tokens"
    )
    if any(
        v is not None
        for v in (h_total, h_queue, h_prefill, h_decode, h_infer, h_cached)
    ):
        sources.append("headers")

    if queue is None and h_queue is not None:
        queue = h_queue
    if ttft is None and h_prefill is not None:
        # prefill ≈ TTFT khi không có queue tách riêng trong body
        ttft = h_prefill
    if gen is None and h_decode is not None:
        gen = h_decode
    if mean_itl is None and h_tpot is not None:
        mean_itl = h_tpot

    model_ms: float | None = None
    if isinstance(ttft, (int, float)) and isinstance(gen, (int, float)):
        model_ms = float(ttft) + float(gen)
    elif h_infer is not None:
        model_ms = h_infer
    elif isinstance(ttft, (int, float)):
        model_ms = float(ttft)
    elif h_prefill is not None and h_decode is not None:
        model_ms = h_prefill + h_decode

    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None and h_prompt is not None:
        prompt_tokens = int(h_prompt)
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None and h_completion is not None:
        completion_tokens = int(h_completion)

    # usage chi tiết (một số bản vLLM/OpenAI)
    cached_tokens = h_cached
    ptd = usage.get("prompt_tokens_details") or {}
    if cached_tokens is None and isinstance(ptd, dict):
        cached_tokens = ptd.get("cached_tokens")

    # Ước lượng phía client khi thiếu server metrics
    e2e_tok_s: float | None = None
    if completion_tokens and e2e_ms > 0:
        e2e_tok_s = round(completion_tokens / (e2e_ms / 1000.0), 2)
    if tps is None:
        tps = e2e_tok_s

    network_ms: float | None = None
    if model_ms is not None:
        network_ms = round(max(e2e_ms - model_ms, 0.0), 2)

    return {
        "model_ms": round(model_ms, 2) if model_ms is not None else None,
        "e2e_ms": round(e2e_ms, 2),
        "network_ms_est": network_ms,
        "ttft_ms": ttft,
        "generation_ms": gen,
        "prefill_ms": h_prefill,
        "decode_ms": h_decode,
        "inference_ms": h_infer,
        "queue_ms": queue,
        "server_total_ms": h_total,
        "mean_itl_ms": mean_itl,
        "tokens_per_second": tps,
        "e2e_completion_tok_s": e2e_tok_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "metrics_source": "+".join(sources),
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
    # model_ms = engine; e2e_ms = client (+mạng)
    if perf["model_ms"] is not None:
        print(
            f"         model_ms   : {perf['model_ms']:,.1f} ms"
            f"  (engine: ttft/prefill + decode)"
        )
    else:
        print(
            "         model_ms   : n/a"
            "  → make down && make up  (bật per-request-metrics / stats-headers)"
        )
    print(f"         e2e_ms     : {perf['e2e_ms']:,.1f} ms  (client: mạng + engine)")
    if perf.get("network_ms_est") is not None:
        print(f"         network~   : {perf['network_ms_est']:,.1f} ms  (e2e - model)")
    bits = []
    for key, label in (
        ("queue_ms", "queue"),
        ("prefill_ms", "prefill"),
        ("ttft_ms", "ttft"),
        ("decode_ms", "decode"),
        ("generation_ms", "gen"),
        ("inference_ms", "infer"),
        ("server_total_ms", "srv_total"),
        ("mean_itl_ms", "itl"),
    ):
        if perf.get(key) is not None:
            bits.append(f"{label}={perf[key]}")
    if bits:
        print(f"         detail     : {' '.join(bits)}")
    if perf.get("tokens_per_second") is not None:
        print(f"         tok/s      : {perf['tokens_per_second']}")
    print(
        f"         tokens     : prompt={perf.get('prompt_tokens')} "
        f"completion={perf.get('completion_tokens')} "
        f"cached={perf.get('cached_tokens')}"
    )
    print(f"         source     : {perf.get('metrics_source')}")
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
            _, api, headers, e2e_ms = chat_complete("__warmup__", model)
            perf = extract_perf(api, headers, e2e_ms)
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
    print(" metric  : model_ms=engine | e2e_ms=client(+mạng)")
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
                raw, api, headers, e2e_ms = chat_complete(query, model)
                perf = extract_perf(api, headers, e2e_ms)
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
