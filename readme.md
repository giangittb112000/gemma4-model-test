# Query Parser (test) — vLLM + QLoRA Docker

Một API `:8000` (base + LoRA). Train tách file compose, chạy xong container thoát.

## Setup

```bash
cp .env.example .env   # HF_TOKEN=hf_xxx
```

## Serve

```bash
docker compose up -d          # hoặc: make up
make wait
make models-list                                   # xem model nào đang serve
make test                                          # base — mỗi dòng in `model:`
make test Q="dt ip 256"
make compare Q="dt ip 256"                         # base + LoRA cùng query

# 2 lệnh test từng model (đổi Q tuỳ ý):
make test MODEL=google/gemma-4-e2b-it Q="dt ip 256"
make test MODEL=query-parser-ft Q="dt ip 256"
```

## Train (one-shot)

Tắt serve / process GPU khác trước, rồi:

```bash
docker compose stop
# nếu còn Ollama: sudo systemctl stop ollama  (hoặc docker stop ...)
nvidia-smi   # free VRAM nên ~15GiB
make train   # hoặc: docker compose -f compose.train.yaml run --rm train
```

Xong → `./models/adapters/query-parser-ft/` → `docker compose up -d` lại.

Nếu OOM lúc prepare: script đã dùng prepare nhẹ (không upcast embedding). Vẫn OOM → tắt hết process GPU, hoặc giảm `--max-seq-length`.

## Cấu trúc

```text
compose.yaml          # chỉ vLLM
compose.train.yaml    # chỉ train, thoát khi xong
finetune/             # Dockerfile + train_qlora.py + data/train.json
models/adapters/      # output LoRA (mount)
scripts/vllm-serve.sh # auto --enable-lora nếu có adapter
test_vllm.py
```
