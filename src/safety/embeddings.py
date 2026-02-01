"""
Embedding Manager for semantic similarity operations.

This module provides a singleton EmbeddingManager class that handles text embeddings
using BGE-small model for semantic similarity matching in safety and RAG components.
"""

import os
from typing import List, Optional, Union
import numpy as np
from dataclasses import dataclass

from src.utils.logging_config import setup_logging

logger = setup_logging("embeddings")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32


class EmbeddingManager:
    """
    Singleton manager for text embeddings using BGE-small model.

    Shared between Safety Gateway and RAG system to avoid loading
    the model multiple times.
    """

    _instance: Optional["EmbeddingManager"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[EmbeddingConfig] = None, mock_mode: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[EmbeddingConfig] = None, mock_mode: bool = False):
        if self._initialized:
            return

        self.config = config or EmbeddingConfig()
        self.mock_mode = mock_mode or os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"
        self._model = None
        self._initialized = True

        logger.info(f"EmbeddingManager initialized (mock_mode={self.mock_mode})")

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        if self.mock_mode:
            logger.info("Using mock embedding model")
            self._model = MockEmbeddingModel()
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed, using mock mode")
            self._model = MockEmbeddingModel()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding (optional)

        Returns:
            numpy.ndarray: Embeddings array of shape (n_texts, embedding_dim)
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        if self.mock_mode:
            return self._model.encode(texts)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False
        )

        return np.array(embeddings)

    def similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus embeddings.

        Args:
            query_embedding: Query embedding (1, dim) or (dim,)
            corpus_embeddings: Corpus embeddings (n, dim)

        Returns:
            numpy.ndarray: Similarity scores (n,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if not already
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
        corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(query_norm, corpus_norm.T).flatten()
        return similarities

    def find_most_similar(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        Find most similar texts from corpus.

        Args:
            query: Query text
            corpus: List of corpus texts
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (text, similarity_score) tuples
        """
        if not corpus:
            return []

        query_emb = self.encode(query)
        corpus_emb = self.encode(corpus)

        similarities = self.similarity(query_emb, corpus_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                results.append((corpus[idx], float(score)))

        return results

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._initialized = False


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._cache = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate deterministic mock embeddings.

        Uses hash of text to generate consistent embeddings for the same input.
        """
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                # Generate deterministic embedding based on text hash
                np.random.seed(hash(text) % (2**32))
                emb = np.random.randn(self.embedding_dim).astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                self._cache[text] = emb
                embeddings.append(emb)

        return np.array(embeddings)
