#!/usr/bin/env bash
# Entrypoint vLLM: base (+ LoRA nếu có). Text-only — bỏ qua MM processor/video.
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
  # Gemma 4 là multimodal; workload query-parser chỉ cần text.
  # Tránh tải/init video+image+audio processor (dễ treo HF hub / bị docker stop giữa chừng).
  --limit-mm-per-prompt
  '{"image":0,"video":0,"audio":0}'
)

# Flag có trên vLLM mới — bỏ qua nếu image cũ không nhận.
if python3 -c "import vllm,sys; sys.exit(0)" 2>/dev/null; then
  if python3 -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- '--language-model-only'; then
    args+=(--language-model-only)
    echo "[vllm] --language-model-only"
  fi
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
  echo "[vllm] no adapters in ${ADAPTERS_DIR} — serving base only"
fi

echo "[vllm] starting: ${MODEL_ID}"
exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
