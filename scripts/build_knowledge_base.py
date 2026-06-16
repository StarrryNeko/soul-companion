"""Build the local ChromaDB knowledge base."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.indexer import KnowledgeBaseIndexer


def main() -> None:
    indexer = KnowledgeBaseIndexer()
    indexer.build()
    print("knowledge_base=ok")


if __name__ == "__main__":
    main()

