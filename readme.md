# Query Parser Test — vLLM trực tiếp + JSON schema

Chỉ chạy **1 service**: `vllm/vllm-openai` với model `google/gemma-4-e2b-it`.  
Test gọi thẳng API OpenAI-compatible của vLLM (`/v1/chat/completions`) kèm `response_format: json_schema`. Không có service API trung gian.

## Chuẩn bị

```bash
cp .env.example .env
# sửa: HF_TOKEN=hf_xxx
```

Tắt tạm Ollama (hoặc service đang chiếm GPU) trước khi chạy.

## Chạy trên server

```bash
make up                 # start vLLM :8000
make wait               # chờ model load xong
make test               # bộ query mặc định
make test Q="dt ip 256" # test 1 query tùy chọn
make logs
make down
```

Hoặc gọi script trực tiếp:

```bash
python3 test_vllm.py
python3 test_vllm.py "dt ip 256"
python3 test_vllm.py "ss s24" "tai nghe bluetooth"
```

## Gọi tay (curl)

```bash
curl -s localhost:8000/health

curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "google/gemma-4-e2b-it",
    "temperature": 0,
    "max_tokens": 128,
    "messages": [
      {
        "role": "user",
        "content": "Phân tích query ecommerce tiếng Việt, trả JSON schema cố định (category, product, brand, model, attributes). Thiếu thì null. query: \"dt ip 256\""
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "query_parse",
        "schema": {
          "type": "object",
          "properties": {
            "category": {"type": ["string", "null"]},
            "product": {"type": ["string", "null"]},
            "brand": {"type": ["string", "null"]},
            "model": {"type": ["string", "null"]},
            "attributes": {"type": "object"}
          },
          "required": ["category", "product", "brand", "model", "attributes"],
          "additionalProperties": false
        }
      }
    }
  }'
```

## Ghi chú

- Model cache tại `./hf-cache`. Sau lần tải đầu, đặt `HF_HUB_OFFLINE=1` trong `.env`.
- JSON schema do vLLM ép lúc sinh token — không có sẵn trong model.
- Script test: `test_vllm.py` (được `make test` gọi).
