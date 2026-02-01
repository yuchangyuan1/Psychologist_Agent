"""
RAG (Retrieval-Augmented Generation) module.

This module provides knowledge retrieval capabilities for the
psychologist agent using vector similarity search.
"""

from src.rag.retriever import RAGRetriever, RetrievalResult, RAGConfig, ContextBuilder
from src.rag.vectorstore import (
    VectorStoreFactory, BaseVectorStore, VectorStoreType,
    Document, SearchResult, MockVectorStore
)
from src.rag.document_loader import DocumentLoader, ChunkConfig

__all__ = [
    "RAGRetriever",
    "RetrievalResult",
    "RAGConfig",
    "ContextBuilder",
    "VectorStoreFactory",
    "BaseVectorStore",
    "VectorStoreType",
    "Document",
    "SearchResult",
    "MockVectorStore",
    "DocumentLoader",
    "ChunkConfig"
]
