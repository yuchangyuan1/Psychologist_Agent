"""
RAG Retriever for knowledge-augmented response generation.

This module provides the main RAGRetriever class that combines
document loading, embedding, and vector search for knowledge retrieval.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.safety.embeddings import EmbeddingManager, EmbeddingConfig
from src.rag.vectorstore import (
    VectorStoreFactory, BaseVectorStore, VectorStoreType,
    Document, SearchResult
)
from src.rag.document_loader import DocumentLoader, ChunkConfig
from src.utils.logging_config import setup_logging

logger = setup_logging("rag_retriever")


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    content: str
    source: str
    source_type: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    top_k: int = 5
    score_threshold: float = 0.3
    max_context_length: int = 2000
    include_metadata: bool = True


class RAGRetriever:
    """
    RAG retriever for knowledge-augmented response generation.

    Combines document loading, embedding generation, and vector search
    to retrieve relevant context for the psychologist agent.

    Example:
        retriever = RAGRetriever()
        await retriever.initialize()
        results = await retriever.retrieve("How can I manage anxiety?")
        context = retriever.format_context(results)
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        mock_mode: Optional[bool] = None,
        knowledge_dir: Optional[str] = None
    ):
        """
        Initialize RAG retriever.

        Args:
            config: RAG configuration
            embedding_config: Embedding model configuration
            mock_mode: Whether to use mock mode
            knowledge_dir: Path to knowledge base directory
        """
        self.config = config or RAGConfig()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        # Initialize components
        self.embedding_manager = EmbeddingManager(
            config=embedding_config,
            mock_mode=self.mock_mode
        )

        self.document_loader = DocumentLoader(knowledge_dir=knowledge_dir)

        store_type = VectorStoreType.MOCK if self.mock_mode else None
        self.vector_store: BaseVectorStore = VectorStoreFactory.create(store_type)

        self._initialized = False

        logger.info(f"RAGRetriever created (mock_mode={self.mock_mode})")

    async def initialize(self, force_reload: bool = False) -> None:
        """
        Initialize the retriever by loading and indexing documents.

        Args:
            force_reload: Force reload even if already initialized
        """
        if self._initialized and not force_reload:
            return

        # Load documents
        documents = self.document_loader.load_all()

        if not documents:
            logger.warning("No documents loaded from knowledge base")
            self._initialized = True
            return

        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_manager.encode(texts)

        # Add to vector store
        self.vector_store.clear()
        self.vector_store.add_documents(documents, embeddings)

        self._initialized = True
        logger.info(f"RAG retriever initialized with {len(documents)} documents")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_source_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Number of results to return
            score_threshold: Minimum relevance score
            filter_source_type: Filter by source type (cbt, dbt, etc.)

        Returns:
            List of RetrievalResult objects
        """
        if not self._initialized:
            await self.initialize()

        top_k = top_k or self.config.top_k
        threshold = score_threshold or self.config.score_threshold

        # Encode query
        query_embedding = self.embedding_manager.encode(query)

        # Search vector store
        search_results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # Filter and convert results
        results = []
        for sr in search_results:
            if sr.score < threshold:
                continue

            if filter_source_type and sr.document.metadata.get("source_type") != filter_source_type:
                continue

            results.append(RetrievalResult(
                content=sr.document.content,
                source=sr.document.metadata.get("source", "unknown"),
                source_type=sr.document.metadata.get("source_type", "general"),
                score=sr.score,
                metadata=sr.document.metadata
            ))

            if len(results) >= top_k:
                break

        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

    def format_context(
        self,
        results: List[RetrievalResult],
        max_length: Optional[int] = None,
        include_sources: bool = True
    ) -> str:
        """
        Format retrieval results as context for the model.

        Args:
            results: List of RetrievalResult objects
            max_length: Maximum context length
            include_sources: Whether to include source information

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        max_length = max_length or self.config.max_context_length
        context_parts = []
        current_length = 0

        for result in results:
            if include_sources:
                source_info = f"[{result.source_type.upper()}] "
                if "title" in result.metadata:
                    source_info += f"({result.metadata['title']}): "
            else:
                source_info = ""

            entry = f"{source_info}{result.content}"

            if current_length + len(entry) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length
                if remaining > 100:
                    entry = entry[:remaining] + "..."
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            current_length += len(entry) + 2  # +2 for separator

        return "\n\n".join(context_parts)

    async def add_document(
        self,
        content: str,
        source: str = "user_added",
        source_type: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new document to the knowledge base.

        Args:
            content: Document content
            source: Source identifier
            source_type: Type of source
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc = Document(
            content=content,
            metadata={
                "source": source,
                "source_type": source_type,
                **(metadata or {})
            },
            doc_id=f"{source}_{self.vector_store.count()}"
        )

        embedding = self.embedding_manager.encode(content)
        self.vector_store.add_documents([doc], embedding)

        logger.info(f"Added document: {doc.doc_id}")
        return doc.doc_id

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "initialized": self._initialized,
            "document_count": self.vector_store.count(),
            "mock_mode": self.mock_mode,
            "config": {
                "top_k": self.config.top_k,
                "score_threshold": self.config.score_threshold,
                "max_context_length": self.config.max_context_length
            }
        }


class ContextBuilder:
    """Helper class for building context from multiple sources."""

    def __init__(self, max_length: int = 4000):
        self.max_length = max_length
        self.sections: List[Dict[str, Any]] = []

    def add_section(
        self,
        title: str,
        content: str,
        priority: int = 1
    ) -> "ContextBuilder":
        """Add a section to the context."""
        self.sections.append({
            "title": title,
            "content": content,
            "priority": priority
        })
        return self

    def add_rag_results(
        self,
        results: List[RetrievalResult],
        title: str = "Relevant Knowledge",
        priority: int = 2
    ) -> "ContextBuilder":
        """Add RAG results as a section."""
        if not results:
            return self

        content_parts = []
        for r in results:
            source_info = f"[{r.source_type.upper()}]"
            content_parts.append(f"{source_info} {r.content}")

        self.add_section(title, "\n".join(content_parts), priority)
        return self

    def add_conversation_history(
        self,
        messages: List[Dict[str, str]],
        title: str = "Recent Conversation",
        max_messages: int = 5,
        priority: int = 3
    ) -> "ContextBuilder":
        """Add conversation history as a section."""
        if not messages:
            return self

        recent = messages[-max_messages:]
        content_parts = []
        for msg in recent:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            content_parts.append(f"{role}: {text}")

        self.add_section(title, "\n".join(content_parts), priority)
        return self

    def build(self) -> str:
        """Build the final context string."""
        # Sort by priority (lower = more important)
        sorted_sections = sorted(self.sections, key=lambda x: x["priority"])

        result_parts = []
        current_length = 0

        for section in sorted_sections:
            section_text = f"### {section['title']}\n{section['content']}"

            if current_length + len(section_text) > self.max_length:
                remaining = self.max_length - current_length
                if remaining > 200:
                    section_text = section_text[:remaining] + "\n[truncated]"
                    result_parts.append(section_text)
                break

            result_parts.append(section_text)
            current_length += len(section_text) + 2

        return "\n\n".join(result_parts)
