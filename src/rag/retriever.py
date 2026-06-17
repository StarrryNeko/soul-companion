"""Knowledge retrieval with ChromaDB, plus a lightweight fallback."""

from __future__ import annotations

from pathlib import Path

from config.settings import RAG_CONFIG


class KnowledgeRetriever:
    """Retrieve relevant knowledge chunks."""

    def __init__(self, persist_dir: str | None = None, docs_dir: str | None = None) -> None:
        self.persist_dir = Path(persist_dir or RAG_CONFIG["persist_dir"])
        self.docs_dir = Path(docs_dir or RAG_CONFIG["docs_dir"])
        self.embedding_model = RAG_CONFIG["embedding_model"]
        self._embedding_model = None
        self._collection = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or RAG_CONFIG["top_k"]
        try:
            return self._retrieve_chroma(query, top_k)
        except Exception:
            return self._retrieve_keyword(query, top_k)

    def _retrieve_chroma(self, query: str, top_k: int) -> list[dict]:
        collection = self._get_collection()
        model = self._get_embedding_model()
        embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
        result = collection.query(query_embeddings=[embedding], n_results=top_k)

        rows = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, distance in zip(docs, metas, distances):
            rows.append({"content": doc, "source": meta.get("source", ""), "score": float(distance)})
        return rows

    def _retrieve_keyword(self, query: str, top_k: int) -> list[dict]:
        rows = []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            score = sum(1 for char in set(query) if char and char in text)
            if score > 0:
                rows.append({"content": text[:700], "source": path.name, "score": float(score)})
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[:top_k]

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_collection("mental_health_docs")
        return self._collection

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model)
        return self._embedding_model
