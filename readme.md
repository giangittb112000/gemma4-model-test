# Vietnamese E-commerce Query Parser (Gemma 4)

Model chuyên phân tích truy vấn tìm kiếm ecommerce tiếng Việt, trả về JSON có cấu trúc cố định.

## Mục tiêu

- Xây dựng một model chuyên phân tích truy vấn tìm kiếm ecommerce tiếng Việt.
- Model phải sửa lỗi chính tả, từ viết tắt, từ không dấu và từ bị viết dính.
- Đầu ra phải phân tách rõ `category`, `product`, `brand`, `model` và `attributes`.
- Model luôn trả về JSON theo một schema cố định, không giải thích thêm.
- Không được tự bịa category, sản phẩm hoặc thuộc tính khi query không đủ thông tin (dùng `null`).

Ví dụ:

```
search "dt ip 256"  ->  category "điện thoại", product "iphone", spec "256gb"
```

## Chọn model & tối ưu tốc độ

Ưu tiên **tốc độ** nên chọn model nhỏ nhất họ Gemma 4:

- **Base model: `google/gemma-4-E2B-it` (~2.3B)** — bản nhỏ nhất, đủ mạnh cho task hẹp (trích xuất entity từ query ngắn). QLoRA 4-bit chỉ cần ~3-6 GB VRAM.
- Nếu sau khi eval thấy `category`/`attribute F1` chưa đạt, mới cân nhắc nâng lên `E4B` (~4.3B, <10 GB VRAM) — vẫn rất nhanh.

Tối ưu tốc độ khi inference:

- **Quantize GGUF `Q4_K_M`** (cân bằng) hoặc `Q4_0` (nhanh hơn); E2B nhỏ nên `Q5_K_M`/`Q6_K` vẫn nhẹ nếu cần chất lượng.
- **Giới hạn `max_new_tokens` ~64-128** vì JSON đầu ra ngắn.
- **System prompt gọn, cố định** để prefill nhanh.
- **Constrained decoding (JSON grammar)**: vừa đảm bảo JSON valid 100%, vừa tăng tốc do thu hẹp không gian token. Ollama hỗ trợ qua `format` (JSON schema).

> Lưu ý: Gemma 4 dùng role `model` (không phải `assistant`). Prompt lúc train và inference phải giống hệt nhau.

## Output schema

```json
{
  "category": "điện thoại",
  "product": "iphone",
  "brand": "apple",
  "model": null,
  "attributes": { "storage": "256gb" }
}
```

Trường thiếu thông tin để `null` (hoặc `{}` cho `attributes`). Không bịa giá trị.

## Chiến lược dữ liệu (TripleLearn)

Dùng 3 dataset bổ trợ nhau thay vì 1 tập hoàn hảo:

1. **Golden data** (~2K-16K, gán nhãn tay, lấy mẫu phân tầng theo pattern entity) — bootstrap + đo lường thật.
2. **Noisy data** (tự sinh từ search logs + click logs, match ngược với catalog/taxonomy) — bắt biến thể thật (`dt ip 256`, `ss s24`...).
3. **Synthetic data** (sinh từ catalog + alias) — phủ 100% category/brand/product.

Train lặp: bắt đầu từ Golden → dự đoán trên Noisy, chỉ giữ mẫu prediction khớp nhãn log (consensus lọc nhiễu) → thêm vào tập train → lặp lại. Refresh model khi có sản phẩm mới chỉ cần update Synthetic.

Augmentation tiếng Việt (sinh 3-5 biến thể/query): bỏ dấu (`điện thoại`→`dien thoai`), viết tắt/alias (`dt`→điện thoại, `ip`→iphone), viết dính (`iphone256gb`), typo telex (`iphon`, `samsng`).

## Fine-tune (QLoRA)

QLoRA 4-bit với Unsloth (nhanh ~2x, ít VRAM ~70%):

- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `target_modules=[q,k,v,o,gate,up,down]_proj`
- `lr=2e-4`, cosine scheduler, `warmup_ratio=0.05`, `epochs=3`, `optim="adamw_8bit"`, effective batch = 8

Đánh giá: JSON valid rate, category accuracy, attribute F1, latency.

Deploy: merge LoRA → `save_pretrained_gguf(q4_k_m)` → `ollama create` → inference bằng Ollama.

## Chạy thử (API + Docker + Make)

Một API FastAPI nhỏ load thẳng `google/gemma-4-e2b-it` từ Hugging Face (base model + prompt, chưa fine-tune) để kiểm tra pipeline inference đầu-cuối.

Token Hugging Face đặt trong file `.env` (đã `.gitignore`, KHÔNG commit):

```bash
cp .env.example .env
# rồi sửa .env: HF_TOKEN=hf_xxx
```

Đây là model gated nên token phải có quyền truy cập repo.

Các lệnh:

```bash
make up      # chạy API nền, lần đầu tải trọng số (~5GB)
make test    # gửi các query mẫu (curl) tới API đang chạy
make logs    # xem log
make down    # dừng
make clean   # dừng + xoá cache model
```

`make test` giả định API đã chạy sẵn (`make up` và `/health` = `ok`), chỉ lo việc gửi curl các query mẫu.

Gọi trực tiếp:

```bash
curl -s localhost:8000/parse -H 'content-type: application/json' \
  -d '{"query":"dt ip 256"}'
```

Endpoint:
- `GET /health` — trạng thái model (`loading`/`ok`) và `device` (GPU/CPU đang dùng).
- `POST /parse` — body `{"query": "..."}` → `{"raw": "...", "parsed": {category, product, brand, model, attributes}}`.

### Chạy trên server GPU NVIDIA (khuyến nghị)

Mặc định cấu hình đã bật GPU. Yêu cầu trên host:

- NVIDIA driver.
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Chạy:

```bash
docker compose up --build -d
make test        # hoặc: curl localhost:8000/health  -> xem "device": "cuda:..."
```

Dockerfile cài `torch` bản CUDA `cu124` (đổi sang `cu121`/`cu118` trong `Dockerfile` nếu driver cũ hơn). Code tự chọn dtype: `bfloat16` cho GPU Ampere+ (A100/RTX 30/40/L4), `float16` cho GPU cũ (T4). Latency kỳ vọng ~0.2–1s/query (thay vì ~80s trên CPU).

### Chạy trên máy không có GPU (ví dụ Mac, chỉ để thử)

Xoá (hoặc comment) khối `deploy` trong `compose.yaml` để không yêu cầu GPU; code sẽ tự chạy CPU (`float32`) — nhưng chậm (~vài chục giây/query).

Cấu trúc:

```
.
├── app/
│   ├── main.py            # FastAPI: load model HF, /parse, /health
│   ├── test_api.py        # smoke test gọi API, validate JSON
│   └── requirements.txt
├── Dockerfile
├── compose.yaml           # đọc HF_TOKEN từ .env
├── Makefile
├── .env.example           # mẫu, copy thành .env
└── readme.md
```
