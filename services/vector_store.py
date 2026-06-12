"""
Vector Store Service (ChromaDB)

Manages a local, embedded ChromaDB instance for storing and retrieving
policy/documentation chunks using semantic search via Gemini embeddings.
"""

from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import get_settings


class VectorStoreService:
    """
    Wrapper around a persistent ChromaDB collection.

    ChromaDB is used here because it is:
    - Free and open-source
    - Embedded (no separate server process needed)
    - Persists to local disk (./chroma_db)
    - Sufficient for small-scale document collections (policies, FAQs)
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[chromadb.Client] = None
        self._collection = None

    def connect(self) -> None:
        """Initialize the persistent ChromaDB client and collection."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB connected: collection='{self.settings.chroma_collection_name}', "
                f"docs={self._collection.count()}"
            )

    @property
    def collection(self):
        if self._collection is None:
            self.connect()
        return self._collection

    # =========================================================================
    # INGESTION
    # =========================================================================

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """Insert or update documents with their embeddings."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(ids)} documents into vector store")

    def delete_all(self) -> None:
        """Clear the entire collection (used when re-ingesting from scratch)."""
        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} existing documents")

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 4,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant documents for a query embedding.

        Returns a list of dicts with 'document', 'metadata', and 'distance'.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        formatted = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append({
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            })

        return formatted

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self.collection.count()
