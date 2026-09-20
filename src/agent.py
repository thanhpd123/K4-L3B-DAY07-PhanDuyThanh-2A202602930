from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan trong kho tri thức."

        context_parts: list[str] = []
        for index, result in enumerate(results, start=1):
            doc_id = result.get("metadata", {}).get("doc_id", result.get("id", "unknown"))
            context_parts.append(f"[{index}] (nguồn: {doc_id})\n{result['content']}")
        context = "\n\n".join(context_parts)

        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa CHỈ trên ngữ cảnh được cung cấp bên dưới.\n"
            "Nếu ngữ cảnh không chứa câu trả lời, hãy nói rõ là không tìm thấy, tuyệt đối không bịa đặt.\n"
            "Khi trả lời, hãy trích dẫn số của đoạn ngữ cảnh bạn sử dụng, ví dụ [1], [2].\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )
        return self.llm_fn(prompt)
