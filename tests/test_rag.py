"""
Tests for RAG (Retrieval-Augmented Generation) module.

All tests use MOCK mode - no real embeddings or vector stores.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.rag.retriever import RAGRetriever, RetrievalResult, RAGConfig, ContextBuilder
from src.rag.vectorstore import (
    MockVectorStore, Document, VectorStoreFactory, VectorStoreType
)
from src.rag.document_loader import DocumentLoader, ChunkConfig
from src.memory.store import MemoryStore, MemoryConfig, Message
from src.safety.embeddings import EmbeddingManager


class TestMockVectorStore:
    """Tests for MockVectorStore class."""

    def test_add_and_search(self):
        """Test adding documents and searching."""
        store = MockVectorStore()

        docs = [
            Document(content="Hello world", doc_id="1"),
            Document(content="Goodbye world", doc_id="2"),
            Document(content="Hello there", doc_id="3")
        ]

        # Create mock embeddings
        import numpy as np
        embeddings = np.random.randn(3, 384).astype(np.float32)

        store.add_documents(docs, embeddings)
        assert store.count() == 3

    def test_delete_documents(self):
        """Test deleting documents."""
        store = MockVectorStore()

        docs = [
            Document(content="Doc 1", doc_id="1"),
            Document(content="Doc 2", doc_id="2")
        ]

        import numpy as np
        embeddings = np.random.randn(2, 384).astype(np.float32)

        store.add_documents(docs, embeddings)
        assert store.count() == 2

        store.delete(["1"])
        assert store.count() == 1

    def test_clear_store(self):
        """Test clearing the store."""
        store = MockVectorStore()

        docs = [Document(content="Test", doc_id="1")]
        import numpy as np
        embeddings = np.random.randn(1, 384).astype(np.float32)

        store.add_documents(docs, embeddings)
        assert store.count() == 1

        store.clear()
        assert store.count() == 0


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_create_mock_store(self):
        """Test creating mock store."""
        store = VectorStoreFactory.create(VectorStoreType.MOCK)
        assert isinstance(store, MockVectorStore)

    def test_auto_detect_mock_mode(self):
        """Test auto-detection of mock mode."""
        os.environ["LLM_TYPE"] = "MOCK"
        store = VectorStoreFactory.create()
        assert isinstance(store, MockVectorStore)


class TestDocumentLoader:
    """Tests for DocumentLoader class."""

    def test_load_all_documents(self):
        """Test loading all documents from knowledge base."""
        loader = DocumentLoader()
        docs = loader.load_all()

        # Should load CBT and DBT documents
        assert len(docs) > 0
        source_types = set(d.metadata.get("source_type") for d in docs)
        assert "cbt" in source_types or "dbt" in source_types

    def test_chunk_config(self):
        """Test custom chunk configuration."""
        config = ChunkConfig(chunk_size=200, chunk_overlap=20)
        loader = DocumentLoader(chunk_config=config)

        assert loader.chunk_config.chunk_size == 200
        assert loader.chunk_config.chunk_overlap == 20

    def test_load_from_text(self):
        """Test loading document from raw text."""
        loader = DocumentLoader()
        text = "This is a test document with some content."
        docs = loader.load_from_text(text, source="test", source_type="test_type")

        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "test"
        assert docs[0].metadata["source_type"] == "test_type"


class TestRAGRetriever:
    """Tests for RAGRetriever class."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingManager.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingManager.reset_instance()

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test retriever initialization."""
        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        stats = retriever.get_stats()
        assert stats["initialized"]
        assert stats["mock_mode"]

    @pytest.mark.asyncio
    async def test_retrieve(self):
        """Test document retrieval."""
        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        results = await retriever.retrieve("anxiety management techniques")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_format_context(self):
        """Test context formatting."""
        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        results = await retriever.retrieve("CBT techniques")
        context = retriever.format_context(results)

        assert isinstance(context, str)

    @pytest.mark.asyncio
    async def test_add_document(self):
        """Test adding a document."""
        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        initial_count = retriever.vector_store.count()
        doc_id = await retriever.add_document(
            content="New therapeutic technique for anxiety",
            source="test",
            source_type="custom"
        )

        assert doc_id is not None
        assert retriever.vector_store.count() == initial_count + 1


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    def test_add_sections(self):
        """Test adding sections to context."""
        builder = ContextBuilder(max_length=1000)
        builder.add_section("Section 1", "Content 1", priority=1)
        builder.add_section("Section 2", "Content 2", priority=2)

        result = builder.build()
        assert "Section 1" in result
        assert "Section 2" in result

    def test_add_rag_results(self):
        """Test adding RAG results."""
        builder = ContextBuilder()
        results = [
            RetrievalResult(
                content="CBT technique for anxiety",
                source="cbt_techniques.md",
                source_type="cbt",
                score=0.9,
                metadata={}
            )
        ]

        builder.add_rag_results(results)
        result = builder.build()

        assert "[CBT]" in result
        assert "anxiety" in result

    def test_truncation(self):
        """Test context truncation."""
        builder = ContextBuilder(max_length=100)
        builder.add_section("Section", "A" * 200, priority=1)

        result = builder.build()
        assert len(result) <= 110  # Some tolerance for [truncated]


class TestMemoryStore:
    """Tests for MemoryStore class."""

    @pytest.fixture
    def store(self):
        """Create a fresh memory store."""
        return MemoryStore()

    @pytest.mark.asyncio
    async def test_add_and_get_history(self, store):
        """Test adding and retrieving history."""
        await store.add("session1", "Hello", "Hi there!")
        history = await store.get_history("session1")

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_history_limit(self, store):
        """Test history limit."""
        for i in range(10):
            await store.add("session1", f"User message {i}", f"Response {i}")

        history = await store.get_history("session1", limit=4)
        assert len(history) == 4

    @pytest.mark.asyncio
    async def test_formatted_history(self, store):
        """Test formatted history output."""
        await store.add("session1", "How are you?", "I'm here to help!")

        simple = await store.get_formatted_history("session1", format_type="simple")
        assert "User:" in simple
        assert "Assistant:" in simple

    @pytest.mark.asyncio
    async def test_session_metadata(self, store):
        """Test session metadata."""
        await store.set_session_metadata("session1", "risk_level", "low")
        value = await store.get_session_metadata("session1", "risk_level")

        assert value == "low"

    @pytest.mark.asyncio
    async def test_clear_session(self, store):
        """Test clearing a session."""
        await store.add("session1", "Test", "Response")
        assert await store.session_exists("session1")

        await store.clear_session("session1")
        assert not await store.session_exists("session1")

    @pytest.mark.asyncio
    async def test_get_last_n_turns(self, store):
        """Test getting last N turns."""
        for i in range(5):
            await store.add("session1", f"User {i}", f"Assistant {i}")

        turns = await store.get_last_n_turns("session1", n=2)
        assert len(turns) == 4  # 2 turns = 4 messages

    @pytest.mark.asyncio
    async def test_session_stats(self, store):
        """Test getting session statistics."""
        await store.add("session1", "Hello", "Hi!")
        await store.add("session1", "How are you?", "I'm good!")

        stats = await store.get_session_stats("session1")
        assert stats["message_count"] == 4
        assert stats["turn_count"] == 2


class TestIntegration:
    """Integration tests for RAG and Memory."""

    def setup_method(self):
        """Reset state before each test."""
        EmbeddingManager.reset_instance()

    @pytest.mark.asyncio
    async def test_rag_with_memory(self):
        """Test RAG retrieval with conversation memory."""
        retriever = RAGRetriever(mock_mode=True)
        memory = MemoryStore()

        await retriever.initialize()

        # Simulate a conversation
        await memory.add("session1", "I'm feeling anxious", "I understand...")

        # Retrieve relevant knowledge
        results = await retriever.retrieve("anxiety management")

        # Build context
        builder = ContextBuilder()
        history = await memory.get_last_n_turns("session1", n=3)
        builder.add_conversation_history(history)
        builder.add_rag_results(results)

        context = builder.build()
        assert isinstance(context, str)
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
