#!/usr/bin/env python
"""
Process WHO Preventing Suicide PDF into structured Markdown.

This script uses OCR (EasyOCR) to extract text from the WHO PDF since it has
non-standard font encoding, then cleans and structures the content for RAG indexing.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import io


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def extract_text_with_ocr(pdf_path: str, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Extract text from PDF using OCR.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process (None for all)

    Returns:
        List of tuples (page_number, text_content)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    try:
        import easyocr
    except ImportError:
        print("Error: EasyOCR not installed. Run: pip install easyocr")
        sys.exit(1)

    # Initialize OCR reader
    print("Initializing OCR reader...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    pages = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc) if max_pages is None else min(max_pages, len(doc))

    for page_num in range(total_pages):
        print(f"Processing page {page_num + 1}/{total_pages}...", end=' ')

        page = doc[page_num]
        # Render page to image at 200 DPI for good OCR quality
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        # Run OCR
        results = reader.readtext(img_bytes)

        # Extract text with confidence filtering
        page_text_parts = []
        for bbox, text, confidence in results:
            if confidence > 0.4:  # Filter low confidence text
                page_text_parts.append(text)

        page_text = '\n'.join(page_text_parts)
        pages.append((page_num + 1, page_text))
        print(f"({len(page_text_parts)} text blocks)")

    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Clean extracted OCR text.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    # Remove multiple consecutive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]

    # Remove lines that are just page numbers
    lines = [line for line in lines if not re.match(r'^Page\s+\d+$', line, re.IGNORECASE)]
    lines = [line for line in lines if not re.match(r'^\d{1,3}$', line)]

    # Remove common header/footer patterns
    header_patterns = [
        r'^PREVENTING SUICIDE:?\s*$',
        r'^A RESOURCE FOR COUNSELLORS\s*$',
        r'^Department of Mental Health.*$',
        r'^World Health Organization\s*$',
    ]

    cleaned_lines = []
    for line in lines:
        is_header = False
        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_header = True
                break
        if not is_header:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def detect_heading(line: str) -> Optional[int]:
    """
    Detect if a line is a heading and determine its level.

    Args:
        line: Text line to analyze

    Returns:
        Heading level (1-3) or None if not a heading
    """
    line = line.strip()

    if not line or len(line) < 3:
        return None

    # Skip very long lines
    if len(line) > 100:
        return None

    # Chapter/Section patterns (H1)
    h1_patterns = [
        r'^(Chapter|CHAPTER)\s+\d+[:\.]?\s*',
        r'^(Section|SECTION)\s+\d+[:\.]?\s*',
        r'^(Part|PART)\s+\d+[:\.]?\s*',
        r'^INTRODUCTION$',
        r'^REFERENCES$',
        r'^APPENDIX',
        r'^ACKNOWLEDGEMENTS$',
    ]

    for pattern in h1_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return 1

    # All caps lines that look like section headers (H2)
    if line.isupper() and 4 < len(line) < 60:
        # Filter out things that are clearly not headers
        if not any(c.isdigit() for c in line[:3]):
            return 2

    # Key topic patterns (H3)
    h3_keywords = [
        'risk factor', 'warning sign', 'protective factor',
        'assessment', 'intervention', 'prevention', 'treatment',
        'suicide attempt', 'suicidal ideation', 'crisis',
        'mental health', 'mental disorder', 'depression',
        'substance abuse', 'alcohol', 'communication',
        'training', 'resources', 'guidelines', 'recommendations'
    ]

    line_lower = line.lower()
    for keyword in h3_keywords:
        if line_lower.startswith(keyword) and len(line) < 60:
            return 3

    return None


def structure_content(pages: List[Tuple[int, str]]) -> str:
    """
    Structure the extracted content with Markdown formatting.

    Args:
        pages: List of (page_number, text) tuples

    Returns:
        Structured Markdown content
    """
    # Combine all pages
    all_text = '\n\n'.join([text for _, text in pages])
    cleaned_text = clean_text(all_text)

    lines = cleaned_text.split('\n')
    structured_lines = []

    # Add document header
    structured_lines.append("# WHO Preventing Suicide: A Resource for Counsellors")
    structured_lines.append("")
    structured_lines.append(
        "> This document is extracted from the WHO publication "
        "'Preventing Suicide: A Resource for Counsellors' for use in "
        "crisis intervention knowledge retrieval."
    )
    structured_lines.append("")
    structured_lines.append("---")
    structured_lines.append("")

    current_paragraph = []

    for line in lines:
        line = line.strip()

        if not line:
            # End of paragraph
            if current_paragraph:
                structured_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            structured_lines.append("")
            continue

        # Check if this is a heading
        heading_level = detect_heading(line)

        if heading_level:
            # Flush current paragraph
            if current_paragraph:
                structured_lines.append(' '.join(current_paragraph))
                current_paragraph = []
                structured_lines.append("")

            # Add heading
            prefix = '#' * (heading_level + 1)  # +1 because document title is H1
            structured_lines.append(f"{prefix} {line}")
            structured_lines.append("")
            continue

        # Check for list items
        list_match = re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•●○◦\-\*]\s*(.+)$', line)
        if list_match:
            if current_paragraph:
                structured_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            structured_lines.append(f"- {list_match.group(1)}")
            continue

        # Numbered list items
        numbered_match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line)
        if numbered_match:
            if current_paragraph:
                structured_lines.append(' '.join(current_paragraph))
                current_paragraph = []
            structured_lines.append(f"{numbered_match.group(1)}. {numbered_match.group(2)}")
            continue

        # Regular text - accumulate into paragraph
        current_paragraph.append(line)

    # Flush remaining paragraph
    if current_paragraph:
        structured_lines.append(' '.join(current_paragraph))

    return '\n'.join(structured_lines)


def post_process_markdown(content: str) -> str:
    """
    Post-process the Markdown for final cleanup.

    Args:
        content: Raw structured content

    Returns:
        Cleaned Markdown content
    """
    # Remove excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Ensure proper spacing around headings
    content = re.sub(r'(#{2,}\s+[^\n]+)\n([^#\n\-])', r'\1\n\n\2', content)

    # Clean up any OCR artifacts
    content = re.sub(r'\s+([,\.\!\?])', r'\1', content)

    return content.strip()


def main():
    """Main function to process WHO PDF."""
    project_root = get_project_root()

    # Input PDF path
    pdf_path = project_root / "data" / "raw" / "WHO_Preventing_suicide.pdf"

    # Output Markdown path
    output_dir = project_root / "data" / "knowledge"
    output_path = output_dir / "who_preventing_suicide.md"

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        print("Please ensure the WHO PDF is placed at: data/raw/WHO_Preventing_suicide.pdf")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {pdf_path}")
    print(f"Output: {output_path}")
    print()

    # Extract text using OCR
    print("Extracting text using OCR (this may take several minutes)...")
    pages = extract_text_with_ocr(str(pdf_path))
    print(f"\nExtracted text from {len(pages)} pages")

    # Structure content
    print("Structuring content...")
    structured_content = structure_content(pages)

    # Post-process
    print("Post-processing...")
    final_content = post_process_markdown(structured_content)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"\nSuccessfully wrote Markdown to: {output_path}")

    # Print statistics
    word_count = len(final_content.split())
    line_count = len(final_content.split('\n'))
    heading_count = len(re.findall(r'^#{2,}', final_content, re.MULTILINE))

    print(f"\nStatistics:")
    print(f"  - Words: {word_count}")
    print(f"  - Lines: {line_count}")
    print(f"  - Headings: {heading_count}")


if __name__ == "__main__":
    main()
