# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phan Duy Thành
**Nhóm:** G16
**Ngày:** 2026-09-20

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

Hai văn bản có vector hướng gần như trùng nhau, tức chúng biểu diễn cùng một ý nghĩa (hoặc rất giống nhau), kể cả khi dùng từ ngữ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Tôi muốn trả lại món hàng này.
- Câu B: Tôi cần hoàn trả sản phẩm vừa mua.
- Tại sao tương đồng: hai câu dùng từ khác nhau ("trả lại" vs "hoàn trả", "món hàng" vs "sản phẩm") nhưng cùng nghĩa "đổi trả".

**Ví dụ có độ tương tự THẤP:**
- Câu A: Chính sách đổi trả trong 15 ngày.
- Câu B: Hướng dẫn nấu món bò kho.
- Tại sao khác: một câu về chính sách mua sắm, một câu về nấu ăn — khác hoàn toàn chủ đề.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

Cosine chỉ đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài vector (độ dài văn bản). Euclid lại nhạy với độ lớn vector, nên hai văn bản cùng nghĩa nhưng khác độ dài vẫn bị tính là xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23
> Đáp án: 23 chunks (đã kiểm chứng bằng `FixedSizeChunker(chunk_size=500, overlap=50)` — in ra 23).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

Tăng từ 50 → 100 thì số chunk tăng từ 23 → 25. Overlap lớn giúp thông tin nằm ở ranh giới cắt không bị đứt đôi giữa hai chunk — mỗi chunk giữ lại một phần ngữ cảnh của chunk trước nên retrieval bắt trọn nghĩa tốt hơn, đổi lại tốn thêm chunk.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

Dùng `re.split(r"(?<=[.!?])\s+", text)` — lookbehind tách ở vị trí ngay SAU dấu câu nên dấu câu không bị nuốt mất. Sau đó gom tối đa `max_sentences_per_chunk` câu thành một chunk và strip khoảng trắng thừa. Edge case chưa xử lý: chữ viết tắt (TS., v.v.) và số thập phân (3.14) sẽ bị cắt sai.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

Thử separator theo thứ tự ưu tiên ["\n\n", "\n", ". ", " ", ""]. Ba base case: (1) văn bản đã ≤ chunk_size; (2) hết separator → cắt cứng theo chunk_size; (3) separator rỗng "" → cắt cứng. Gồm cả hai chiều: đệ quy xuống (mảnh quá dài → gọi lại với separator còn lại) và gom lên (nối mảnh nhỏ tới sát chunk_size).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

Lưu in-memory trong list `_store`, mỗi record gồm id, content, metadata (bản sao, có kèm `doc_id`) và embedding. `search` nhúng câu truy vấn rồi tính dot product (vector đã chuẩn hoá nên dot = cosine), sắp xếp giảm dần và trả top_k (bỏ trường embedding).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

Filter TRƯỚC rồi mới search (nếu search trước lấy top-k rồi mới lọc thì k slot có thể bị tài liệu sai chiếm hết, để lại 0 kết quả dù store vẫn còn tài liệu hợp lệ). `delete_document` xoá mọi record có `metadata["doc_id"]` khớp, trả True nếu có xoá.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

Ba nhịp: truy xuất top-k → dựng prompt → gọi `llm_fn`. Prompt đánh số từng chunk `[1] [2] [3]` kèm nguồn `doc_id`, yêu cầu model trích dẫn số đó khi trả lời (truy vết nguồn), đồng thời cấm bịa và xử lý store rỗng (trả thông báo, không gọi LLM vô ích).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
# => 42 passed in 0.14s
# Toàn bộ 42 test PASS: 2 ProjectStructure + 2 ClassBasedInterfaces + 7 FixedSizeChunker
# + 4 SentenceChunker + 4 RecursiveChunker + 9 EmbeddingStore + 2 KnowledgeBaseAgent
# + 4 ComputeSimilarity + 3 CompareChunkingStrategies + 3 SearchWithFilter + 3 DeleteDocument
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A                              | Câu B                                       | Dự đoán | Điểm thực tế | Đúng? |
| --- | ---------------------------------- | ------------------------------------------- | ------- | ------------ | ----- |
| 1   | Tôi muốn trả lại hàng              | Tôi cần đổi trả sản phẩm                    | cao     | -0.0244      | Không |
| 2   | Chính sách đổi trả trong 15 ngày   | Hướng dẫn nấu món bò kho                    | thấp    | -0.1712      | Không |
| 3   | Người mua có 7 ngày để khiếu nại   | Khách hàng được khiếu nại trong vòng 7 ngày | cao     | -0.0865      | Không |
| 4   | Đồ bơi không được trả lại          | Bikini không thể hoàn trả                   | cao     | -0.1629      | Không |
| 5   | Hoàn tiền qua ví MoMo trong 48 giờ | Sản phẩm LazMall trả trong 30 ngày          | thấp    | -0.1425      | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

Toàn bộ 5 cặp đều ra gần 0 (thậm chí âm nhẹ), kể cả cặp cùng nghĩa. Lý do: lab đang dùng MockEmbedder băm MD5 rồi sinh số giả ngẫu nhiên nên KHÔNG mã hoá ngữ nghĩa — chỉ hai chuỗi giống hệt nhau mới ra 1.0. Đây chính là minh chứng phải dùng embedder thật (Local/OpenAI/Gemini) để đo độ tương tự.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| #   | Câu hỏi (Query)                                                | Top-1 Chunk truy xuất được (tóm tắt)                              | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
| --- | -------------------------------------------------------------- | ----------------------------------------------------------------- | ---------- | ------------------------------ | ------------------------------- |
| 1   | Sản phẩm LazMall & CHOICE được hoàn trả trong bao lâu?         | "Ảnh và video cần thiết..." (lazada-yeu-cau-hoan-tien)            | 0.3171     | Không (mock)                   | —                               |
| 2   | Thanh toán bằng ví MoMo thì bao lâu nhận được tiền hoàn?       | "...đến lấy hàng của bạn..." (lazada-cach-tra-hang)               | 0.3034     | Không (mock)                   | —                               |
| 3   | Những loại sản phẩm nào không được trả lại?                    | "Giao hàng không thành công..." (lazada-chinh-sach-hoan-tien)     | 0.2962     | Không (mock)                   | —                               |
| 4   | Để trả hàng với lý do "Tôi đổi ý", cần điều kiện gì?           | "...bằng chứng hợp lệ..." (lazada-cach-tra-hang)                  | 0.3184     | Không (mock)                   | —                               |
| 5   | Khi sản phẩm phải thu hồi, ai phải thông báo và trong bao lâu? | "...khiếu nại của bên thứ ba..." (lazada-dieu-khoan-nha-ban-hang) | 0.2213     | Đúng tài liệu, sai section     | —                               |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 0 / 5 (với MockEmbedder — số liệu là nhiễu)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> MockEmbedder băm MD5 nên không đo ngữ nghĩa; muốn benchmark thật phải dùng embedder thật (Local/OpenAI/Gemini). Riêng câu 5 có metadata filter vẫn lọc đúng về tài liệu `seller`, cho thấy filter hoạt động độc lập với chất lượng embedding.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------- |
| Khởi động (Warm-up)                             | / 5              |
| Hướng tiếp cận của tôi (My Approach)            | / 10             |
| Hoàn thiện code (Core Implementation — tests)   | / 30             |
| Dự đoán độ tương tự (Similarity Predictions)    | / 5              |
| Kết quả truy xuất của tôi (Competition Results) | / 10             |
| **Tổng phần cá nhân**                           | **/ 60**         |
