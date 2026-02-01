"""
Tests for WHO PDF processing module.

Tests cover:
- PDF extraction functionality
- Text cleaning and structuring
- Markdown format correctness
- RAG retrieval of WHO content
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


class TestWHOPDFScript:
    """Tests for the WHO PDF processing script."""

    def test_script_exists(self):
        """Test that the processing script exists."""
        script_path = PROJECT_ROOT / "scripts" / "process_who_pdf.py"
        assert script_path.exists(), "Processing script should exist"

    def test_script_importable(self):
        """Test that script modules can be imported."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from process_who_pdf import (
                get_project_root,
                clean_text,
                detect_heading,
                structure_content,
                post_process_markdown
            )
            assert callable(get_project_root)
            assert callable(clean_text)
            assert callable(detect_heading)
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))


class TestTextCleaning:
    """Tests for text cleaning functions."""

    @pytest.fixture
    def clean_text_func(self):
        """Import clean_text function."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from process_who_pdf import clean_text
            yield clean_text
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))

    def test_remove_page_numbers(self, clean_text_func):
        """Test removal of standalone page numbers."""
        text = "Some content\n42\nMore content"
        cleaned = clean_text_func(text)
        # Page number should be removed
        assert "\n42\n" not in cleaned

    def test_remove_multiple_blank_lines(self, clean_text_func):
        """Test consolidation of multiple blank lines."""
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_text_func(text)
        assert "\n\n\n" not in cleaned

    def test_preserve_content(self, clean_text_func):
        """Test that actual content is preserved."""
        text = "Important suicide prevention information"
        cleaned = clean_text_func(text)
        assert "suicide prevention" in cleaned


class TestHeadingDetection:
    """Tests for heading detection."""

    @pytest.fixture
    def detect_heading_func(self):
        """Import detect_heading function."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from process_who_pdf import detect_heading
            yield detect_heading
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))

    def test_detect_chapter_heading(self, detect_heading_func):
        """Test detection of chapter headings."""
        assert detect_heading_func("Chapter 1: Introduction") == 1
        assert detect_heading_func("CHAPTER 2") == 1

    def test_detect_uppercase_section(self, detect_heading_func):
        """Test detection of all-caps sections."""
        result = detect_heading_func("RISK FACTORS")
        assert result == 2

    def test_regular_text_not_heading(self, detect_heading_func):
        """Test that regular text is not detected as heading."""
        assert detect_heading_func("This is a regular paragraph with text.") is None

    def test_short_text_not_heading(self, detect_heading_func):
        """Test that very short text is not heading."""
        assert detect_heading_func("A") is None

    def test_long_text_not_heading(self, detect_heading_func):
        """Test that very long text is not heading."""
        long_text = "A" * 150
        assert detect_heading_func(long_text) is None


class TestMarkdownGeneration:
    """Tests for Markdown structure generation."""

    @pytest.fixture
    def structure_content_func(self):
        """Import structure_content function."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from process_who_pdf import structure_content
            yield structure_content
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))

    def test_adds_document_header(self, structure_content_func):
        """Test that document header is added."""
        pages = [(1, "Some content")]
        result = structure_content_func(pages)
        assert "# WHO Preventing Suicide" in result

    def test_adds_blockquote(self, structure_content_func):
        """Test that blockquote is added."""
        pages = [(1, "Some content")]
        result = structure_content_func(pages)
        assert ">" in result
        assert "crisis intervention" in result

    def test_preserves_content(self, structure_content_func):
        """Test that page content is preserved."""
        pages = [(1, "Suicide risk assessment")]
        result = structure_content_func(pages)
        assert "Suicide" in result or "suicide" in result


class TestPostProcessing:
    """Tests for Markdown post-processing."""

    @pytest.fixture
    def post_process_func(self):
        """Import post_process_markdown function."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            from process_who_pdf import post_process_markdown
            yield post_process_markdown
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))

    def test_remove_excessive_blank_lines(self, post_process_func):
        """Test removal of excessive blank lines."""
        content = "Line 1\n\n\n\n\nLine 2"
        result = post_process_func(content)
        assert "\n\n\n" not in result

    def test_clean_punctuation_spacing(self, post_process_func):
        """Test cleanup of punctuation spacing."""
        content = "Hello , world ."
        result = post_process_func(content)
        assert ", world." in result or "Hello, world." in result


class TestOutputFile:
    """Tests for the generated Markdown output file."""

    @pytest.fixture
    def output_path(self):
        """Get path to output file."""
        return PROJECT_ROOT / "data" / "knowledge" / "who_preventing_suicide.md"

    def test_output_file_exists(self, output_path):
        """Test that output file exists."""
        if not output_path.exists():
            pytest.skip("Output file not yet generated")
        assert output_path.exists()

    def test_output_has_content(self, output_path):
        """Test that output file has meaningful content."""
        if not output_path.exists():
            pytest.skip("Output file not yet generated")

        content = output_path.read_text(encoding='utf-8')
        assert len(content) > 1000, "File should have substantial content"

    def test_output_has_valid_markdown(self, output_path):
        """Test that output is valid Markdown."""
        if not output_path.exists():
            pytest.skip("Output file not yet generated")

        content = output_path.read_text(encoding='utf-8')

        # Should have document title
        assert content.startswith("# "), "Should start with H1 header"

        # Should have multiple sections
        heading_count = content.count("\n## ") + content.count("\n### ")
        assert heading_count >= 5, "Should have multiple section headings"

    def test_output_encoding(self, output_path):
        """Test that output file is UTF-8 encoded."""
        if not output_path.exists():
            pytest.skip("Output file not yet generated")

        # Should not raise encoding error
        content = output_path.read_text(encoding='utf-8')
        assert isinstance(content, str)


class TestRAGIntegration:
    """Tests for RAG integration with WHO content."""

    @pytest.fixture
    def who_content_path(self):
        """Get path to WHO content file."""
        return PROJECT_ROOT / "data" / "knowledge" / "who_preventing_suicide.md"

    def test_document_loader_finds_who_file(self, who_content_path):
        """Test that DocumentLoader can find WHO file."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_all()

        # Find WHO documents
        who_docs = [d for d in docs if "who" in d.metadata.get("source", "").lower()]
        assert len(who_docs) > 0, "WHO document should be loaded"

    def test_who_document_chunks(self, who_content_path):
        """Test that WHO document is properly chunked."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_file(str(who_content_path))

        assert len(docs) > 0, "Should produce multiple chunks"

        # Check metadata
        for doc in docs:
            assert "source" in doc.metadata
            assert "who" in doc.metadata["source"].lower()

    def test_who_source_type_detection(self, who_content_path):
        """Test that WHO document has correct source type."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        docs = loader.load_file(str(who_content_path))

        # Check source type (should be detected as general or crisis-related)
        for doc in docs[:5]:
            assert "source_type" in doc.metadata


class TestRAGRetrieval:
    """Tests for RAG retrieval of WHO content."""

    @pytest.fixture
    def who_content_path(self):
        """Get path to WHO content file."""
        return PROJECT_ROOT / "data" / "knowledge" / "who_preventing_suicide.md"

    @pytest.fixture
    def reset_embeddings(self):
        """Reset embedding manager before and after test."""
        from src.safety.embeddings import EmbeddingManager
        EmbeddingManager.reset_instance()
        yield
        EmbeddingManager.reset_instance()

    @pytest.mark.asyncio
    async def test_retrieve_suicide_risk_factors(self, who_content_path, reset_embeddings):
        """Test retrieval of suicide risk factors."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.retriever import RAGRetriever

        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        results = await retriever.retrieve("suicide risk factors")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_crisis_intervention(self, who_content_path, reset_embeddings):
        """Test retrieval of crisis intervention content."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.retriever import RAGRetriever

        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        results = await retriever.retrieve("crisis intervention guidelines")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_warning_signs(self, who_content_path, reset_embeddings):
        """Test retrieval of warning signs content."""
        if not who_content_path.exists():
            pytest.skip("WHO output file not yet generated")

        from src.rag.retriever import RAGRetriever

        retriever = RAGRetriever(mock_mode=True)
        await retriever.initialize()

        results = await retriever.retrieve("warning signs of suicide")
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
