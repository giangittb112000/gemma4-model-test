# Mô tả dự án: Phân tích truy vấn tìm kiếm ecommerce tiếng Việt

Dự án dùng AI để “đọc” câu tìm kiếm của người dùng (thường viết tắt, sai chính tả, không dấu) và trả về thông tin có cấu trúc, ví dụ:

```text
Người dùng gõ:  dt ip 256
Hệ thống hiểu:  ngành hàng = điện thoại
                sản phẩm   = iphone
                dung lượng = 256gb
```

Tài liệu này chỉ tập trung vào 3 quyết định chính của dự án.

---

## 1. Model: `google/gemma-4-e2b-it`

Đây là model gốc (base model) của dự án — bản nhỏ, nhanh trong họ Gemma 4 của Google, phù hợp bài toán search cần phản hồi nhanh.

### Tên model nghĩa là gì?

| Thành phần trong tên | Ý nghĩa đơn giản |
|---|---|
| `google/gemma-4` | Họ model Gemma thế hệ 4 do Google phát hành trên Hugging Face |
| `e2b` | Bản **E2B** — khoảng **2.3 tỷ tham số hiệu dụng** (effective ~2B), tối ưu để chạy nhanh, tốn ít bộ nhớ |
| `-it` | **Instruction-tuned** — đã được huấn luyện thêm để **làm theo hướng dẫn** (ví dụ: “chỉ trả JSON, không giải thích”) |

### `E2B` khác gì với tên kiểu `gemma4-2b`?

Trong Gemma 4, Google **không đặt tên đơn giản là `2B`**, mà dùng **`E2B`** với chữ **E = Effective (hiệu dụng)**.

- Nói `gemma-4-e2b` ≈ “bản Gemma 4 cỡ ~2B, tối ưu hiệu năng”.
- Không nên hiểu là một model “2B kiểu cũ” hoàn toàn giống Gemma 2/3; đây là **dòng E riêng** của Gemma 4.
- Nếu cần mạnh hơn một bậc mà vẫn nhỏ, họ có **`E4B`** (~4B hiệu dụng) — dự án chỉ cân nhắc khi E2B chưa đủ chính xác.

### “2.3 tỷ tham số hiệu dụng” khác gì “2.3 tỷ tham số bình thường”?

**Tham số** = số “nút kiến thức” của model. Càng nhiều thường càng mạnh, nhưng cũng càng nặng (tốn GPU, chậm hơn).

| Cách nói | Ý nghĩa thực tế |
|---|---|
| **2.3B tham số bình thường** | Gần như toàn bộ ~2.3 tỷ tham số đều phải nằm trên GPU để tính toán mỗi lần trả lời |
| **2.3B tham số hiệu dụng (effective)** | Phần “não” thật sự phải tính trên GPU ~2.3B; còn thêm bảng tra cứu từ (embedding) làm tổng tham số trên giấy tờ lớn hơn (~5.1B) |

Với E2B, Google dùng kiến trúc **Per-Layer Embeddings (PLE)**:

- **Phần hiệu dụng (~2.3B):** quyết định tốc độ và VRAM chính khi chạy.
- **Phần bảng tra cứu từ:** dùng để “tra nghĩa token” nhanh; không cần hiểu như “não tính toán” full-time.

Ví dụ đời thường:

- **2.3B bình thường** = 2.3 triệu dụng cụ, tất cả phải để trên bếp.
- **2.3B hiệu dụng** = chỉ 2.3 triệu dụng cụ trên bếp để nấu; kho lớn ngoài (bảng từ) lấy khi cần, không chất hết lên bếp.

**Nhớ một câu:**

> Tham số bình thường ≈ “có bao nhiêu đồ”.  
> Tham số hiệu dụng ≈ “phải bật bao nhiêu đồ lên GPU mỗi lần chạy”.

Vì vậy khi chọn E2B cho search, ta quan tâm chính là **chi phí chạy gần như model ~2.3B**, dù tổng tham số đếm hết có thể lớn hơn.

### `-it` khác gì bản không có `-it`?

| Loại | Đặc điểm |
|---|---|
| Không `-it` (base/pretrained) | Model “thô”, giỏi tiếp tục câu văn, chưa quen làm theo lệnh cụ thể |
| Có `-it` (instruction-tuned) | Đã được dạy kiểu hỏi–đáp / làm theo yêu cầu → dễ bảo “trả đúng JSON schema” |

Với bài toán parse query, bản **`-it`** phù hợp hơn vì ta cần model **tuân thủ format đầu ra**, không phải viết văn dài.

### Vì sao chọn E2B-it cho search?

1. **Search cần nhanh** — query ngắn, kết quả phải về ngay; model nhỏ thắng model lớn về tốc độ.
2. **Bài toán hẹp** — chỉ cần chuẩn hoá câu tìm kiếm và tách category / product / brand / attributes, không cần “trí tuệ đa năng”.
3. **Dễ vận hành trên GPU ~16GB** — còn chỗ chạy serving, fine-tune sau này cũng nhẹ.
4. **Đủ tốt làm nền** — phần còn lại (độ chính xác domain tiếng Việt) sẽ được cải thiện bằng data + fine-tune, không nhất thiết phải dùng model to hơn.

---

## 2. Vì sao dùng thêm vLLM + JSON schema

Phần này giải thích theo hướng dễ hiểu, ít thuật ngữ.

### Bài toán thực tế

Có model giỏi chưa đủ. Khi đưa vào hệ thống search, ta còn cần 2 điều:

1. **Trả lời nhanh**, kể cả lúc nhiều người search cùng lúc.
2. **Trả lời đúng khuôn** — luôn là JSON có các ô cố định (`category`, `product`, `brand`…), không được “bày tỏ thêm” chữ thừa làm hệ thống phía sau bị lỗi.

Hai công cụ dưới đây giải quyết đúng hai việc đó.

### vLLM — “bếp phục vụ” model nhanh hơn

Có thể hình dung:

- Model giống **đầu bếp**.
- **vLLM** giống **cách tổ chức nhà bếp**: xếp hàng order thông minh, dùng bếp (GPU) hiệu quả, phục vụ nhiều khách cùng lúc thay vì làm từng món một cách chậm.

Nếu chỉ chạy model theo cách đơn giản (mỗi câu một lượt, lần lượt), hệ thống dễ chậm khi traffic tăng.  
vLLM giúp:

- xử lý nhanh hơn trên GPU,
- phục vụ nhiều request song song tốt hơn,
- dễ gắn vào API chuẩn (gọi giống kiểu ChatGPT API).

**Tóm lại cho non-tech:** vLLM không “dạy” model thông minh hơn — nó giúp **chạy model đã có một cách nhanh và ổn định hơn**.

### JSON schema — “khuôn cố định” cho câu trả lời

Nếu để model tự trả lời tự do, đôi khi nó sẽ:

- viết thêm giải thích,
- thiếu một trường,
- hoặc JSON bị lỗi cú pháp.

**JSON schema** giống việc đưa một **biểu mẫu có sẵn các ô**, và chỉ cho phép điền đúng các ô đó. Hệ thống không chấp nhận câu trả lời “lan man”.

Ví dụ khuôn dự án dùng:

```json
{
  "category": "... hoặc null",
  "product": "... hoặc null",
  "brand": "... hoặc null",
  "model": "... hoặc null",
  "attributes": { }
}
```

**Tóm lại cho non-tech:** JSON schema không làm model hiểu tiếng Việt tốt hơn — nó đảm bảo **mọi câu trả lời đều cùng một format**, để search phía sau dùng được ngay.

### JSON schema có sẵn trong model không? Dùng như thế nào?

Đây là câu hỏi rất hay — và câu trả lời thường gây hiểu nhầm:

| Thành phần | Vai trò |
|---|---|
| **Model Gemma** | Quyết định *nội dung* điền vào các ô (điện thoại / iphone / …) |
| **JSON schema** | Bản “luật khuôn” do **dự án tự định nghĩa** (các khóa, kiểu dữ liệu) |
| **vLLM** | Công cụ **thực thi luật đó** lúc chạy: chỉ cho sinh token nào còn hợp schema |

→ JSON schema **không nằm sẵn trong Gemma**.  
→ Đây là tính năng **structured output / guided decoding của vLLM** (nhiều engine khác cũng có tương tự).  
→ Không có vLLM (hoặc engine tương đương), model vẫn có thể “cố” viết JSON theo prompt, nhưng **không được đảm bảo 100% đúng khuôn**.

**Cách dùng trong dự án (luồng thật):**

1. Client gọi API: `POST /parse` với `{"query":"dt ip 256"}`.
2. Service `api` dựng prompt hướng dẫn + lấy `OUTPUT_SCHEMA` đã định nghĩa trong code.
3. `api` gửi sang vLLM kèm:

```json
"response_format": {
  "type": "json_schema",
  "json_schema": { "name": "query_parse", "schema": { /* OUTPUT_SCHEMA */ } }
}
```

4. vLLM sinh câu trả lời **bị ràng buộc bởi schema**.
5. `api` nhận JSON, chuẩn hoá nhẹ, trả về `{raw, parsed}`.

Hình dung dễ nhớ:

- Model = người điền form  
- JSON schema = mẫu form (các ô cố định)  
- vLLM = người giám sát: không cho viết ra ngoài form  

Prompt vẫn cần (để model biết điền *gì*).  
Schema đảm bảo (không được trả *sai format*).

### Vì sao cần cả hai?

| Công cụ | Giải quyết việc gì |
|---|---|
| **vLLM** | Nhanh, chịu được nhiều người dùng |
| **JSON schema** | Đúng khuôn, ít lỗi format |

Có thể hiểu ngắn:

> Model quyết định **nội dung đúng sai**.  
> vLLM quyết định **chạy nhanh thế nào**.  
> JSON schema quyết định **không được trả sai format**.

Ba lớp này bổ sung nhau, không thay thế nhau.

---

## 3. Fine-tune: thiết kế data và vai trò của QLoRA

Fine-tune = **dạy thêm** model gốc bằng dữ liệu đúng bài toán của mình (search ecommerce tiếng Việt), để nó hiểu alias/typo/domain tốt hơn.

Ví dụ trước fine-tune, model có thể nhầm `product` và `brand`. Sau khi học từ data thật của shop, nó ổn định hơn với các câu như `dt ip 256`, `ss s24`, `ao thun nam mau den`.

### 3.1 Cách thiết kế data (ý tưởng TripleLearn)

Không cố tạo ngay một bộ data “hoàn hảo và khổng lồ”. Dự án dùng **3 nguồn bổ trợ**:

#### A. Golden data (data vàng)

- Gán nhãn tay, chất lượng cao.
- Số lượng vừa phải (thường vài nghìn đến hơn mười nghìn mẫu).
- Dùng để **dạy khởi điểm** và **đo độ chính xác thật**.

#### B. Noisy data (data thật nhưng nhiễu)

- Lấy từ search logs + click logs thực tế.
- Ví dụ: người dùng gõ `dt ip 256` rồi click vào iPhone 256GB → suy ra nhãn gần đúng.
- Số lượng lớn, bắt được cách người dùng **thật sự** gõ (viết tắt, không dấu, viết dính).

#### C. Synthetic data (data sinh từ catalog)

- Sinh từ danh mục sản phẩm + bảng alias (`dt` = điện thoại, `ip` = iphone…).
- Mục tiêu: phủ đủ brand/product/category, kể cả thứ ít xuất hiện trong log.

**Cách dùng kết hợp:**

1. Bắt đầu train với Golden.
2. Dùng model tạm suy đoán trên Noisy, chỉ giữ những mẫu “model và log đồng thuận”.
3. Thêm Synthetic để phủ đủ danh mục.
4. Đo lại trên Golden holdout → lặp đến khi đủ tốt.

**Augmentation tiếng Việt** (mỗi câu gốc sinh thêm vài biến thể):

- bỏ dấu: `điện thoại` → `dien thoai`
- viết tắt: `dt`, `ip`, `ss`
- viết dính: `iphone256gb`
- lỗi gõ: `iphon`, `samsng`

Quy tắc quan trọng: thiếu thông tin thì để `null`, **không bịa**.

### 3.2 QLoRA làm gì? (giải thích dễ hiểu)

Fine-tune full (sửa toàn bộ model) thường **rất nặng**: tốn GPU, tốn tiền, tốn thời gian.

**QLoRA** là cách “dạy thêm” thông minh hơn:

1. **Nén** model gốc xuống dạng nhẹ (4-bit) để vừa GPU.
2. **Đóng băng** kiến thức gốc (không sửa lung tung toàn bộ).
3. Chỉ gắn và train các **miếng adapter nhỏ (LoRA)** — giống “ghi chú chuyên ngành” dán thêm vào model, thay vì viết lại cả cuốn sách.

**Tác dụng thực tế với dự án:**

- Train được trên GPU phổ thông (~16GB), chi phí thấp.
- Giữ được năng lực chung của Gemma, đồng thời học tốt domain search tiếng Việt.
- Sau train chỉ cần lưu adapter (nhẹ) hoặc gộp lại rồi đưa vào vLLM để phục vụ.

Có thể nhớ một câu:

> **Data** dạy model “hiểu đúng ngành hàng/sản phẩm thế nào”.  
> **QLoRA** là cách dạy đó **rẻ và vừa sức phần cứng**, không cần máy siêu mạnh.

### 3.3 Fine-tune nằm ở đâu trong toàn bộ hệ thống?

```text
Data (Golden + Noisy + Synthetic)
        ↓
   Fine-tune bằng QLoRA
        ↓
  Model hiểu domain tốt hơn
        ↓
Phục vụ bằng vLLM + JSON schema
        ↓
   API /parse cho search
```

- Fine-tune cải thiện **độ chính xác nội dung**.
- vLLM + JSON schema cải thiện **tốc độ và độ ổn định format**.
- Cả hai đều cần; không cái nào thay thế cái còn lại.

---

## Phụ lục — Câu hỏi thường gặp khi đọc tài liệu này

### E2B / tham số

**Hỏi: `e2b` khác gì tên kiểu `gemma4-2b`?**  
Đáp: Gemma 4 dùng chữ **E = Effective**. Đây là dòng tối ưu hiệu năng riêng, không phải cách đặt tên `2B` kiểu Gemma đời trước.

**Hỏi: “2.3 tỷ tham số hiệu dụng” khác gì “2.3 tỷ tham số bình thường”?**  
Đáp: Bình thường ≈ hầu hết phải nạp lên GPU để tính. Hiệu dụng ≈ phần não tính toán ~2.3B; tổng tham số trên giấy tờ có thể lớn hơn nhờ bảng tra cứu từ (PLE). Chi phí chạy gần như model ~2.3B.

**Hỏi: `-it` nghĩa là gì?**  
Đáp: Instruction-tuned — model đã được dạy làm theo hướng dẫn, phù hợp bài “trả đúng JSON”, không phải bản pretrained thô.

### vLLM + JSON schema

**Hỏi: vLLM có làm model thông minh hơn không?**  
Đáp: Không. vLLM chỉ giúp **chạy nhanh và ổn định hơn**. Độ hiểu domain đến từ model + fine-tune.

**Hỏi: JSON schema có sẵn trong Gemma không?**  
Đáp: Không. Schema do dự án định nghĩa; **vLLM** là nơi thực thi luật format lúc sinh câu trả lời.

**Hỏi: Vậy đang dùng JSON schema như thế nào?**  
Đáp: API gửi request sang vLLM kèm `response_format.type = json_schema` và bản `OUTPUT_SCHEMA`. vLLM chỉ cho phép output hợp khuôn đó.

**Hỏi: Chỉ cần prompt “hãy trả JSON” thôi có đủ không?**  
Đáp: Đủ để demo, không đủ để production. Prompt làm model *có xu hướng* đúng format; schema + vLLM mới **đảm bảo** format.

### Fine-tune / QLoRA

**Hỏi: Fine-tune có thay thế vLLM + JSON schema không?**  
Đáp: Không. Fine-tune tăng độ chính xác nội dung; vLLM + schema lo tốc độ và format.

**Hỏi: QLoRA khác fine-tune full thế nào?**  
Đáp: Không sửa cả model. Chỉ train adapter nhỏ trên bản nén 4-bit → rẻ hơn, vừa GPU 16GB, vẫn học tốt domain.
