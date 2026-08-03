# Query Parser (test) — vLLM + QLoRA Docker

- **Train:** QLoRA → `./models/adapters/query-parser-ft/`
- **Serve:** base + LoRA adapter (không merge)
- **Test:** chỉ `query-parser-ft`

Giải thích chỉ số (`r`, latency, vLLM…): [`docs/chi-so-du-an.md`](docs/chi-so-du-an.md)

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
| `gpu_memory_utilization` | **0.92** (tránh lỗi hết KV cache trên 16GB) |
| `max_num_seqs` | 4 |
| `max_lora_rank` | 16 (= train r) |
| `max_tokens` (client) | 64 |
| text-only / language-model-only | bật nếu image hỗ trợ |
| prompt | ngắn, chung `finetune/prompt.py` |
| `make ready` | warmup 1 request |

Nếu vẫn `No available memory for the cache blocks`: trong `.env` thử `GPU_MEMORY_UTILIZATION=0.95` hoặc `ENFORCE_EAGER=1`, rồi `make down && make up`.

### `e2e_ms` vs `model_ms`

| Chỉ số | Đo ở đâu | Gồm gì |
|---|---|---|
| **`e2e_ms`** | Client (`perf_counter`) | Mạng + queue + prefill + decode + serialize |
| **`model_ms`** | vLLM (body/headers) | Prefill/TTFT + decode trên GPU (không gồm mạng) |

SLA search nội bộ nên ưu tiên **`model_ms`**; đo trải nghiệm user/API gateway dùng **`e2e_ms`**.

Sau `make down && make up`, log thêm: queue/prefill/decode/cached_tokens/`PERF {...}`.

## Data train

Schema response:

```json
{"category": string|null, "product": string|null, "spec": string[]}
```

- Seed (agent làm giàu, **gitignored**): `finetune/data/finetune/*.json`
- Train file (commit cái này): `finetune/data/train.json` — TRL `messages` (`role=model` cho Gemma)

Local (có seeds): `make data` → commit/push `train.json`. Server: chỉ `make train` (không gom data).

```bash
# local
make data

# server
make train
make up && make ready
make test Q="ip17 256"
```

## Cấu trúc

```text
compose.yaml / compose.train.yaml
finetune/
  prompt.py
  build_train_from_seeds.py
  train_qlora.py
  data/finetune/*.json   # seeds
  data/train.json
models/adapters/
scripts/vllm-serve.sh
test_vllm.py
```
