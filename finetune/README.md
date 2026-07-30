# Fine-tune QLoRA (tách khỏi luồng test vLLM base)

Folder này **độc lập** với `compose.yaml` / `make test` ở root.

Pipeline:

```text
data/train.json  →  QLoRA train  →  outputs/adapter/  →  merge  →  outputs/merged/
                                                                  ↓
                                                  compose.merged.yaml (vLLM :8001)
```

## Data

Chỉ dùng **một file**: [`data/train.json`](data/train.json) — mảng vài mẫu (mô phỏng train thật).

Mỗi phần tử:

```json
{
  "messages": [
    {"role": "user", "content": "<prompt + query>"},
    {"role": "model", "content": "{\"category\":...}"}
  ]
}
```

Sửa/thêm mẫu trực tiếp trong `train.json`. Không có script sinh data.

## 1. Cài môi trường (server GPU)

```bash
cd finetune
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Cần `HF_TOKEN` trong `.env` ở root repo. Tắt vLLM/Ollama đang chiếm GPU trước khi train.

## 2. Train + merge

```bash
make train
```

Kết quả:

| Thư mục | Ý nghĩa |
|---|---|
| `outputs/adapter/` | LoRA adapter (trung gian) |
| `outputs/merged/` | Model đã merge — dùng cho vLLM |

## 3. Chạy merged bằng vLLM

Từ **root repo**:

```bash
docker compose -f compose.merged.yaml up -d
VLLM_URL=http://localhost:8001 MODEL_ID=query-parser-ft make test
```
