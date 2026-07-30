#!/usr/bin/env bash
# Entrypoint vLLM: base + LoRA. Knobs tối ưu latency search.
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-4-e2b-it}"
# Prompt ngắn + JSON ngắn → 512 đủ, KV nhỏ hơn 1024.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
# Gemma4-E2B (multimodal weights) + CUDA graph profile trên 16GB dễ hết chỗ KV
# nếu util quá thấp. Log vLLM gợi ý ~0.86; dùng 0.92 cho A4000 16GB.
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.92}"
# Khớp LoRA r=16 lúc train (đừng để 64 thừa VRAM).
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
ADAPTERS_DIR="${ADAPTERS_DIR:-/adapters}"
# Ít seq đồng thời → ít KV hơn lúc profile/start.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

help_txt="$(python3 -m vllm.entrypoints.openai.api_server --help 2>&1 || true)"

args=(
  --model="${MODEL_ID}"
  --max-model-len="${MAX_MODEL_LEN}"
  --gpu-memory-utilization="${GPU_MEM}"
  --max-num-seqs="${MAX_NUM_SEQS}"
  --enable-prefix-caching
  --port=8000
  --limit-mm-per-prompt
  '{"image":0,"video":0,"audio":0}'
)

# Thu thập timing per-request (body metrics +/hoặc HTTP headers).
if grep -q -- '--enable-per-request-metrics' <<<"${help_txt}"; then
  args+=(--enable-per-request-metrics)
  echo "[vllm] --enable-per-request-metrics"
fi
if grep -q -- '--enable-request-stats-headers' <<<"${help_txt}"; then
  args+=(--enable-request-stats-headers)
  echo "[vllm] --enable-request-stats-headers"
fi
# Cần log-stats bật thì per-request metrics mới có (mặc định on; chỉ tránh disable).
if grep -q -- '--disable-log-stats' <<<"${help_txt}"; then
  : # không truyền --disable-log-stats
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  args+=(--enforce-eager)
  echo "[vllm] ENFORCE_EAGER=1"
fi

if grep -q -- '--language-model-only' <<<"${help_txt}"; then
  args+=(--language-model-only)
  echo "[vllm] --language-model-only"
fi

loras=()
if [[ -d "${ADAPTERS_DIR}" ]]; then
  for d in "${ADAPTERS_DIR}"/*/; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}adapter_config.json" ]] || continue
    name="$(basename "${d}")"
    loras+=("${name}=${d%/}")
  done
fi

if ((${#loras[@]} > 0)); then
  echo "[vllm] enable-lora: ${loras[*]}"
  args+=(--enable-lora --max-lora-rank="${MAX_LORA_RANK}" --lora-modules "${loras[@]}")
else
  echo "[vllm] WARN: không thấy adapter trong ${ADAPTERS_DIR} — chỉ base."
  echo "[vllm]       Cần: make train  →  models/adapters/query-parser-ft/"
fi

echo "[vllm] starting: ${MODEL_ID} max_len=${MAX_MODEL_LEN} lora_rank=${MAX_LORA_RANK} eager=${ENFORCE_EAGER}"
exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
