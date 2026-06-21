"""Build a ChromaDB index from local Markdown documents."""

from __future__ import annotations

from pathlib import Path

from config.settings import RAG_CONFIG


class KnowledgeBaseIndexer:
    """Build or rebuild the local knowledge base."""

    def __init__(
        self,
        docs_dir: str | None = None,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.docs_dir = Path(docs_dir or RAG_CONFIG["docs_dir"])
        self.persist_dir = Path(persist_dir or RAG_CONFIG["persist_dir"])
        self.embedding_model = embedding_model or RAG_CONFIG["embedding_model"]
        self.chunk_size = chunk_size or RAG_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or RAG_CONFIG["chunk_overlap"]

    def build(self) -> None:
        """切分本地 Markdown 文档、生成向量，并重建持久化 ChromaDB 集合。"""
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            client.delete_collection("mental_health_docs")
        except Exception:
            pass
        collection = client.get_or_create_collection("mental_health_docs")
        model = SentenceTransformer(self.embedding_model)

        ids, documents, metadatas = [], [], []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            for idx, chunk in enumerate(self._split_text(text)):
                ids.append(f"{path.stem}-{idx}")
                documents.append(chunk)
                metadatas.append({"source": path.name})

        if not documents:
            raise ValueError(f"No Markdown documents found in {self.docs_dir}")

        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def _split_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end
        return chunks
