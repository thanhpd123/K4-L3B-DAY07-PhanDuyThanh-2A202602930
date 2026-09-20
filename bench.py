"""Benchmark retrieval cho Giai đoạn 2 — công cụ đo của riêng từng thành viên.

Bốn việc:
  1. Đọc .md, tách frontmatter (metadata) và phần thân (content)
  2. Chunk phần thân; mỗi chunk -> Document(id=f"{stem}#{i}", metadata={**fm, "doc_id": stem})
  3. Nạp vào EmbeddingStore, chạy 5 query qua search_with_filter()
  4. In top-3 kèm score + doc_id để đối chiếu gold answer

Để so sánh công bằng, MỖI NGƯỜI CHỈ ĐỔI MỘT DÒNG: dòng gán `CHUNKER`.
"""

from pathlib import Path
import re

from src import Document, EmbeddingStore
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker


DATA_DIR = Path("data/chinh-sach-doi-tra")
TOP_K = 3


class HeadingChunker:
    """Chunk theo tiêu đề/mục (## ...) — chiến lược bắt buộc của R3.

    Mỗi section (mục) đã là một đơn vị ngữ nghĩa trọn vẹn. Section dài quá
    ngưỡng thì hạ xuống RecursiveChunker, và GẮN LẠI tiêu đề vào từng mảnh con
    để mảnh thứ 2 trở đi không mất ngữ cảnh "mục này nói về cái gì".
    """

    def __init__(self, chunk_size: int = 400) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        parts = re.split(r"(?m)(?=^#{1,6}\s)", text)
        sections = [part.strip() for part in parts if part.strip()]
        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue
            lines = section.split("\n", 1)
            heading = lines[0].strip() if lines and lines[0].lstrip().startswith("#") else ""
            body = lines[1] if len(lines) > 1 and heading else section
            for piece in RecursiveChunker(chunk_size=self.chunk_size).chunk(body):
                chunks.append(f"{heading}\n{piece}" if heading else piece)
        return chunks


# ================= CHỌN CHIẾN LƯỢC — CHỈ ĐỔI DÒNG NÀY =================
CHUNKER = FixedSizeChunker(chunk_size=300, overlap=50)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=4)
# CHUNKER = RecursiveChunker(chunk_size=300)
# CHUNKER = HeadingChunker(chunk_size=400)
# =====================================================================

# (nhãn, câu hỏi, metadata_filter — None nếu không cần lọc)
QUERIES = [
    ("Q1 thoi han LazMall", "Sản phẩm LazMall & CHOICE được hoàn trả trong bao lâu?", None),
    ("Q2 hoan tien MoMo", "Thanh toán bằng ví MoMo thì bao lâu nhận được tiền hoàn?", None),
    ("Q3 liet ke", "Những loại sản phẩm nào không được trả lại?", None),
    ("Q4 dieu kien doi y", "Để trả hàng với lý do 'Tôi đổi ý', sản phẩm cần điều kiện gì?", None),
    ("Q5 filter seller", "Khi sản phẩm phải thu hồi, ai phải thông báo và trong bao lâu?", {"audience": "seller"}),
]


def parse_md(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        _, frontmatter, body = text.split("---", 2)
        metadata: dict = {}
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"')
        return metadata, body.strip()
    return {}, text.strip()


def main() -> None:
    store = EmbeddingStore(collection_name="bench")
    files = sorted(DATA_DIR.glob("*.md"))
    docs: list[Document] = []

    for path in files:
        frontmatter, body = parse_md(path)
        for index, chunk in enumerate(CHUNKER.chunk(body)):
            docs.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    metadata={**frontmatter, "doc_id": path.stem},
                )
            )

    store.add_documents(docs)
    print(f"Loaded {len(files)} files -> {len(docs)} chunks\n")

    for label, question, flt in QUERIES:
        results = store.search_with_filter(question, top_k=TOP_K, metadata_filter=flt)
        print(f"== {label} ==")
        print(f"   query : {question}")
        print(f"   filter: {flt}")
        for result in results:
            preview = " ".join(result["content"].split())[:90]
            print(f"   score={result['score']:.4f} doc={result['metadata'].get('doc_id')} | {preview}")
        print()


if __name__ == "__main__":
    main()
