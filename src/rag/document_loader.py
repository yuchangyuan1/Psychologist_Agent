"""
Document loader for RAG knowledge base.

This module handles loading and chunking documents from the knowledge base
(CBT techniques, DBT skills, etc.) for indexing in the vector store.
"""

import os
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass
from pathlib import Path
import re

from src.rag.vectorstore import Document
from src.utils.logging_config import setup_logging

logger = setup_logging("document_loader")


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    separator: str = "\n\n"


class DocumentLoader:
    """
    Loads and chunks documents from the knowledge base.

    Supports Markdown files with intelligent chunking based on
    headers and paragraphs.
    """

    def __init__(
        self,
        knowledge_dir: Optional[str] = None,
        chunk_config: Optional[ChunkConfig] = None
    ):
        """
        Initialize document loader.

        Args:
            knowledge_dir: Path to knowledge base directory
            chunk_config: Configuration for chunking
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.knowledge_dir = knowledge_dir or os.path.join(base_dir, "data", "knowledge")
        self.chunk_config = chunk_config or ChunkConfig()

    def load_all(self) -> List[Document]:
        """
        Load all documents from the knowledge base.

        Returns:
            List of Document objects
        """
        documents = []

        if not os.path.exists(self.knowledge_dir):
            logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            return documents

        for file_path in Path(self.knowledge_dir).glob("**/*.md"):
            try:
                docs = self.load_file(str(file_path))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} total document chunks")
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load and chunk a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of Document chunks
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_name = os.path.basename(file_path)
        source_type = self._detect_source_type(file_name)

        # Parse markdown structure
        sections = self._parse_markdown_sections(content)

        # Create document chunks
        documents = []
        for section in sections:
            chunks = self._chunk_section(section)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk["content"],
                    metadata={
                        "source": file_name,
                        "source_type": source_type,
                        "title": section["title"],
                        "heading_path": section["heading_path"],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    },
                    doc_id=f"{file_name}_{section['title']}_{i}"
                )
                documents.append(doc)

        return documents

    def _detect_source_type(self, file_name: str) -> str:
        """Detect the type of knowledge from file name."""
        name_lower = file_name.lower()
        if "cbt" in name_lower:
            return "cbt"
        elif "dbt" in name_lower:
            return "dbt"
        elif "crisis" in name_lower:
            return "crisis"
        elif "who" in name_lower or "suicide" in name_lower:
            return "who_guideline"
        else:
            return "general"

    def _parse_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse markdown content into sections based on headers.

        Args:
            content: Markdown content

        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = {
            "title": "Introduction",
            "heading_path": [],
            "content": "",
            "level": 0
        }
        heading_stack = []

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update heading stack
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()

                heading_stack.append({"level": level, "title": title})
                heading_path = [h["title"] for h in heading_stack]

                current_section = {
                    "title": title,
                    "heading_path": heading_path,
                    "content": "",
                    "level": level
                }
            else:
                current_section["content"] += line + "\n"

            i += 1

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a section into smaller pieces.

        Args:
            section: Section dictionary

        Returns:
            List of chunk dictionaries
        """
        content = section["content"].strip()

        if not content:
            return []

        if len(content) <= self.chunk_config.chunk_size:
            return [{"content": content}]

        chunks = []
        paragraphs = content.split(self.chunk_config.separator)

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph alone exceeds chunk size, split it further
            if len(para) > self.chunk_config.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append({"content": current_chunk.strip()})
                    current_chunk = ""

                # Split large paragraph
                sub_chunks = self._split_large_text(para)
                chunks.extend([{"content": c} for c in sub_chunks])
                continue

            # Check if adding paragraph exceeds limit
            test_chunk = current_chunk + self.chunk_config.separator + para if current_chunk else para

            if len(test_chunk) > self.chunk_config.chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append({"content": current_chunk.strip()})
                current_chunk = para
            else:
                current_chunk = test_chunk

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.chunk_config.min_chunk_size:
            chunks.append({"content": current_chunk.strip()})

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split large text into smaller chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) > self.chunk_config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def load_from_text(
        self,
        text: str,
        source: str = "user_input",
        source_type: str = "conversation"
    ) -> List[Document]:
        """
        Create document chunks from raw text.

        Args:
            text: Raw text content
            source: Source identifier
            source_type: Type of source

        Returns:
            List of Document chunks
        """
        if len(text) <= self.chunk_config.chunk_size:
            return [Document(
                content=text,
                metadata={"source": source, "source_type": source_type},
                doc_id=f"{source}_0"
            )]

        chunks = self._split_large_text(text)
        return [
            Document(
                content=chunk,
                metadata={"source": source, "source_type": source_type, "chunk_index": i},
                doc_id=f"{source}_{i}"
            )
            for i, chunk in enumerate(chunks)
        ]
