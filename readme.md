# Query Parser (test) — vLLM + QLoRA Docker

- **Train:** QLoRA → `./models/adapters/query-parser-ft/`
- **Serve:** base + LoRA adapter (không merge)
- **Test:** chỉ `query-parser-ft`

## Checklist đẩy server

```bash
cp .env.example .env          # HF_TOKEN=...
# lần đầu: HF_HUB_OFFLINE=0 ; cache đủ rồi có thể =1

make train                    # cần GPU trống
make up && make ready         # wait + warmup (bỏ cold start)
make test Q="iphoooen 17 256" # xem model_ms < 2000
```

## Serve / test

```bash
make up && make ready
make test
make test Q="dt ip 256"
```

## Latency đã tối ưu sẵn

| Knob | Giá trị |
|---|---|
| `enforce-eager` | tắt (steady-state nhanh) |
| prefix cache | bật |
| `max_model_len` | 512 |
| `max_lora_rank` | 16 (= train r) |
| `max_tokens` (client) | 64 |
| text-only / language-model-only | bật nếu image hỗ trợ |
| prompt | ngắn, chung `finetune/prompt.py` |
| `make ready` | warmup 1 request |

Log: `model_ms` (chuẩn SLA), `e2e_ms`, dòng `PERF {...}`.

## Data train

`finetune/data/train.json` — format TRL `messages` (`role`/`content`). Gemma: `role=model`.

## Cấu trúc

```text
compose.yaml / compose.train.yaml
finetune/   # QLoRA, prompt.py, data/train.json
models/adapters/
scripts/vllm-serve.sh
test_vllm.py
```
