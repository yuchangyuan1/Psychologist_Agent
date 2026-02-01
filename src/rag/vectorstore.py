"""
Vector store abstraction for RAG system.

This module provides a unified interface for vector storage backends
(FAISS, ChromaDB) with support for MOCK mode testing.
"""

import os
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from src.utils.logging_config import setup_logging

logger = setup_logging("vectorstore")


class VectorStoreType(Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    CHROMADB = "chromadb"
    MOCK = "mock"


@dataclass
class Document:
    """A document with content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Result from vector search."""
    document: Document
    score: float
    rank: int


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Add documents with their embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of documents."""
        pass


class MockVectorStore(BaseVectorStore):
    """In-memory mock vector store for testing."""

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Add documents with embeddings."""
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            doc.doc_id = doc.doc_id or f"doc_{len(self.documents) + i}"
            self.documents.append(doc)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        logger.info(f"Added {len(documents)} documents to mock store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        corpus_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities
        similarities = np.dot(query_norm, corpus_norm.T).flatten()

        # Get top-k indices
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(similarities[idx]),
                rank=rank
            ))

        return results

    def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by ID."""
        indices_to_keep = [
            i for i, doc in enumerate(self.documents)
            if doc.doc_id not in doc_ids
        ]

        self.documents = [self.documents[i] for i in indices_to_keep]
        if self.embeddings is not None and indices_to_keep:
            self.embeddings = self.embeddings[indices_to_keep]
        elif not indices_to_keep:
            self.embeddings = None

    def clear(self) -> None:
        """Clear all documents."""
        self.documents = []
        self.embeddings = None

    def count(self) -> int:
        """Return number of documents."""
        return len(self.documents)


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store."""

    def __init__(
        self,
        dimension: int = 384,
        index_path: Optional[str] = None
    ):
        self.dimension = dimension
        self.index_path = index_path
        self.index = None
        self.documents: List[Document] = []
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load FAISS index."""
        try:
            import faiss
            if self.index_path and os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created new FAISS index (dim={self.dimension})")
        except ImportError:
            logger.warning("faiss not installed, using mock store")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Add documents with embeddings."""
        import faiss

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        for i, doc in enumerate(documents):
            doc.doc_id = doc.doc_id or f"doc_{len(self.documents) + i}"
            self.documents.append(doc)

        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Added {len(documents)} documents to FAISS index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for similar documents."""
        import faiss

        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        query = query_embedding.astype(np.float32)
        faiss.normalize_L2(query)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0 and idx < len(self.documents):
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank
                ))

        return results

    def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by ID (requires index rebuild)."""
        indices_to_keep = [
            i for i, doc in enumerate(self.documents)
            if doc.doc_id not in doc_ids
        ]

        if len(indices_to_keep) == len(self.documents):
            return

        # Rebuild index
        import faiss
        old_docs = self.documents
        self.documents = []
        self.index = faiss.IndexFlatIP(self.dimension)

        if indices_to_keep:
            kept_docs = [old_docs[i] for i in indices_to_keep]
            embeddings = np.array([d.embedding for d in kept_docs if d.embedding is not None])
            if len(embeddings) > 0:
                self.add_documents(kept_docs, embeddings)

    def clear(self) -> None:
        """Clear all documents."""
        import faiss
        self.documents = []
        self.index = faiss.IndexFlatIP(self.dimension)

    def count(self) -> int:
        """Return number of documents."""
        return self.index.ntotal

    def save(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        import faiss
        save_path = path or self.index_path
        if save_path:
            faiss.write_index(self.index, save_path)
            logger.info(f"Saved FAISS index to {save_path}")


class VectorStoreFactory:
    """Factory for creating vector stores."""

    @staticmethod
    def create(
        store_type: Optional[VectorStoreType] = None,
        dimension: int = 384,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store.

        Args:
            store_type: Type of vector store (auto-detected from LLM_TYPE if None)
            dimension: Embedding dimension
            **kwargs: Additional arguments for the store

        Returns:
            BaseVectorStore: The created vector store
        """
        if store_type is None:
            llm_type = os.getenv("LLM_TYPE", "MOCK").upper()
            if llm_type == "MOCK":
                store_type = VectorStoreType.MOCK
            else:
                store_type = VectorStoreType.FAISS

        if store_type == VectorStoreType.MOCK:
            return MockVectorStore()

        if store_type == VectorStoreType.FAISS:
            try:
                return FAISSVectorStore(dimension=dimension, **kwargs)
            except ImportError:
                logger.warning("FAISS not available, falling back to mock store")
                return MockVectorStore()

        if store_type == VectorStoreType.CHROMADB:
            # ChromaDB implementation would go here
            logger.warning("ChromaDB not implemented, using mock store")
            return MockVectorStore()

        return MockVectorStore()
