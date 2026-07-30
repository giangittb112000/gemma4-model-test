#!/usr/bin/env bash
# Entrypoint vLLM: luôn load base; nếu có LoRA adapter trong /adapters thì bật --enable-lora.
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-4-e2b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
ADAPTERS_DIR="${ADAPTERS_DIR:-/adapters}"

args=(
  --model="${MODEL_ID}"
  --max-model-len="${MAX_MODEL_LEN}"
  --gpu-memory-utilization="${GPU_MEM}"
  --enforce-eager
  --port=8000
)

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
  echo "[vllm] no adapters in ${ADAPTERS_DIR} — serving base only"
fi

exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
