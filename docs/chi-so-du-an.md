# Giải thích các chỉ số trong dự án (dễ hiểu)

Dự án: phân tích query search ecommerce tiếng Việt → JSON, train bằng **QLoRA**, serve bằng **vLLM** (base + LoRA adapter).

---

## Trả lời nhanh: `r` là gì? Tăng `r` có thông minh hơn không?

**`r` (LoRA rank)** = “độ rộng” của miếng kiến thức thêm vào model.

- Giống số “kênh” nhỏ được phép học thêm trên nền base.
- **`r` càng lớn** → adapter học được phức tạp hơn (dung lượng biểu diễn lớn hơn), file adapter nặng hơn một chút, train/serve tốn VRAM hơn, latency có thể tăng nhẹ.
- **`r` càng nhỏ** → học ít “độ tự do” hơn, nhẹ và nhanh hơn.

**Không phải** “tăng `r` là chắc chắn thông minh hơn”.

| Tăng `r` giúp khi… | Không giúp / phản tác dụng khi… |
|---|---|
| Data lớn, bài khó, `r` nhỏ bị underfit | Data ít → dễ nhớ máy / overfitting |
| Cần học nhiều pattern phức tạp | Chỉ cần map alias + JSON đơn giản (như bài này) |

Với query parser hiện tại, **`r = 16` thường đủ**. Data tốt 100k mẫu quan trọng hơn việc tăng `r` lên 64/128.

Liên quan: `lora_alpha` thường ~ `2 × r` (ở đây alpha=32 khi r=16) — hệ số scale độ mạnh cập nhật LoRA.

---

## 1. Nhóm model & luồng dự án

| Tên | Công dụng |
|---|---|
| **Base model** (`google/gemma-4-e2b-it`) | Model gốc đã có sẵn kiến thức + biết làm theo hướng dẫn. |
| **E2B** | Bản Gemma 4 cỡ nhỏ (~2B tham số hiệu dụng), ưu tiên tốc độ / ít VRAM. |
| **`-it`** | Instruction-tuned — đã được dạy “làm theo prompt”. |
| **QLoRA** | Cách **train**: quantize base 4-bit + học LoRA → tiết kiệm VRAM lúc fine-tune. |
| **LoRA adapter** | File nhỏ chứa phần “học thêm”; lúc serve gắn lên base. |
| **Merge** | Cộng adapter vào base thành 1 model full (dự án hiện **không** dùng lúc serve). |
| **Train vs Serve** | Train = dạy (QLoRA). Serve = trả lời user (vLLM + adapter). |

---

## 2. Nhóm fine-tune / LoRA (trong `train_qlora.py`)

| Chỉ số | Công dụng (nói đơn giản) |
|---|---|
| **`lora-r` / `r`** | Độ rộng adapter. Lớn = học phức tạp hơn, nặng hơn. |
| **`lora-alpha`** | Độ “mạnh” khi áp LoRA (thường ~2×r). |
| **`lora_dropout`** | Bỏ ngẫu nhiên một phần kết nối lúc train → giảm overfitting. |
| **`target_modules`** | Layer nào được gắn LoRA (vd. `q_proj`, `v_proj`… trong `language_model`). |
| **`max_seq_length`** | Độ dài tối đa 1 mẫu train (token). Dài hơn = tốn VRAM train hơn. |
| **`epochs`** | Số vòng duyệt hết tập train. Nhiều quá với data nhỏ → dễ nhớ máy. |
| **`batch_size`** | Số mẫu xử lý song song trên GPU mỗi bước. |
| **`grad_accum`** | Gom gradient nhiều bước nhỏ = “batch lớn giả”. |
| **`lr` (learning rate)** | Bước học mỗi lần cập nhật. Cao = học nhanh nhưng dễ loạn; thấp = ổn nhưng chậm. |
| **`optim` (paged_adamw_8bit)** | Bộ tối ưu tiết kiệm VRAM lúc train. |
| **4-bit / NF4 / double quant** | Nén base lúc train để vừa GPU 16GB. |
| **Gradient checkpointing** | Đổi thời gian lấy VRAM (chậm hơn một chút, đỡ OOM). |

**Lưu ý quan trọng:** Số mẫu train (200 hay 100k) **không** làm file adapter to tương ứng — chỉ `r` + số module LoRA quyết định độ nặng adapter.

---

## 3. Nhóm serve / vLLM (latency & VRAM)

| Chỉ số | Công dụng |
|---|---|
| **`max_model_len`** | Độ dài tối đa 1 request (prompt + output) tính bằng token. Nhỏ hơn → KV cache nhẹ hơn. |
| **`gpu_memory_utilization`** | % VRAM vLLM được dùng. Cao hơn → còn chỗ cho KV cache (tránh lỗi hết bộ nhớ lúc start). |
| **`max_num_seqs`** | Số request xử lý song song tối đa. Cao = throughput tốt hơn, dễ tăng latency/p99. |
| **`max_lora_rank`** | Rank LoRA tối đa vLLM dành chỗ. Nên ≥ `r` lúc train (đang 16). |
| **`ENFORCE_EAGER`** | `1` = tắt CUDA graph → start dễ hơn, chạy chậm hơn. `0` = tối ưu tốc độ steady-state. |
| **Prefix cache** | Nhớ phần prompt trùng (system) → request sau prefill nhanh hơn. |
| **`max_tokens` (client)** | Số token được sinh tối đa. JSON ngắn → để thấp (64) để khỏi decode thừa. |
| **Text-only / `language-model-only`** | Bỏ đường ảnh/audio (Gemma 4 multimodal) cho bài search text. |
| **JSON schema (`response_format`)** | Ép output đúng cấu trúc JSON — ổn định format, hơi tốn decode. |
| **KV cache** | Bộ nhớ “ngữ cảnh” lúc sinh token. Hết chỗ KV → engine không start / không chạy được. |
| **CUDA graph** | Tối ưu chạy GPU lặp lại; lúc start có thể chiếm thêm VRAM. |

---

## 4. Nhóm đo latency / performance

| Chỉ số | Công dụng |
|---|---|
| **`e2e_ms`** | Thời gian **1 HTTP call đầy đủ** từ client → vLLM → nhận hết response (gồm mạng). |
| **`model_ms`** | Thời gian **engine** (prefill/TTFT + decode) trên GPU — không gồm mạng. |
| **`ttft_ms` / prefill** | Thời gian đến token đầu (đọc prompt). Prompt dài → TTFT lớn. |
| **`generation_ms` / decode** | Thời gian sinh các token còn lại (JSON). |
| **`queue_ms`** | Chờ trong hàng đợi khi GPU bận. |
| **`network_ms_est`** | Ước lượng `e2e - model` (mạng + overhead client). |
| **`prompt_tokens` / `completion_tokens`** | Độ dài input / output (token). |
| **`cached_tokens`** | Token trúng prefix cache (không phải prefill lại). |
| **`tokens_per_second`** | Tốc độ sinh token. |
| **p50 / p95 / p99** | Latency tại mức 50% / 95% / 99% số request (đo từ nhiều mẫu, không đoán suông). |
| **Cold vs warm** | Request đầu sau khi lên model thường chậm (compile/graph); warm mới gần thực tế prod. |
| **Dòng `PERF {...}`** | Log JSON 1 dòng mỗi request — dùng `grep` để tính p95 sau này. |

**Gợi ý SLA search:** ưu tiên **`model_ms` warm**; nhìn **`e2e_ms`** khi đo cả đường API.

---

## 5. Nhóm data & schema output

| Tên | Công dụng |
|---|---|
| **`messages`** | Format train chuẩn HF/TRL: mảng `{role, content}`. |
| **`role: system`** | Hướng dẫn cố định (prompt). |
| **`role: user`** | Query người dùng. |
| **`role: model`** | Câu trả lời (Gemma dùng `model`, không phải `assistant`). |
| **`category`** | Ngành hàng (điện thoại, laptop, …). |
| **`product`** | SP đã chuẩn, gồm dòng + đời máy nếu có (`iphone 16 pro max`, `galaxy s25`, `samsung`, …). |
| **`spec`** | List string token chuẩn có trong query (`256gb`, `128gb`, `120hz`, …). |
| **`finetune/data/finetune/*.json`** | Seed labeled (`query/category/product/spec`) — agent làm giàu. |
| **`synony.json`** | Luật alias / viết tắt (dùng khi làm giàu seed, không bắt buộc lúc gom train). |

---

## 6. Nhóm lệnh vận hành thường dùng

| Lệnh | Công dụng |
|---|---|
| `make train` | QLoRA → ghi adapter `models/adapters/query-parser-ft/`. |
| `make up` | Bật vLLM. |
| `make ready` | Chờ healthy + warmup 1 request (bỏ cold). |
| `make test Q="..."` | Gọi adapter, in latency + JSON. |
| `make models-list` | Xem model/adapter đang serve. |
| `make data` | Gom seeds `data/finetune/*.json` → `train.json`. |

---

## 7. Quan hệ nhanh (để khỏi nhầm)

```text
Data lớn hơn     → train lâu hơn, model thường giỏi hơn
                 ✗ không làm adapter nặng hơn

r lớn hơn        → adapter “rộng” hơn, có thể học khó hơn
                 → file/VRAM/latency có thể tăng nhẹ

Prompt/output dài → TTFT/decode tăng (latency tăng rõ)
GPU bận / QPS cao → queue_ms tăng → p95/p99 xấu
```

---

*File này mô tả các chỉ số đang dùng trong repo. Khi đổi default trong `compose.yaml` / `train_qlora.py`, ưu tiên tin code đang chạy.*
