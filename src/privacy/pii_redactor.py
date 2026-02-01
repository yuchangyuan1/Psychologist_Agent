"""
PII (Personally Identifiable Information) Redactor.

This module provides the PIIRedactor class for detecting and masking
sensitive information in text before sending to cloud APIs.
"""

import os
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.privacy.patterns import (
    PIIEntityType, PIIPatterns, PIIPatternConfig,
    get_replacement, REPLACEMENT_TEMPLATES
)
from src.utils.logging_config import setup_logging

logger = setup_logging("pii_redactor")


@dataclass
class PIIEntity:
    """A detected PII entity."""
    entity_type: PIIEntityType
    text: str
    start: int
    end: int
    confidence: float
    replacement: str


@dataclass
class RedactionResult:
    """Result from PII redaction."""
    redacted_text: str
    entities: List[PIIEntity]
    mapping: Dict[str, str]  # replacement -> original
    entity_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "redacted_text": self.redacted_text,
            "entity_count": self.entity_count,
            "entities": [
                {
                    "type": e.entity_type.value,
                    "replacement": e.replacement,
                    "confidence": e.confidence
                }
                for e in self.entities
            ]
        }


class PIIRedactor:
    """
    PII Redactor for privacy-preserving text processing.

    Detects and masks sensitive information in user input before
    sending to cloud APIs, with support for restoring original values.

    Example:
        redactor = PIIRedactor()
        result = redactor.redact("My email is john@example.com")
        # result.redacted_text = "My email is [EMAIL_REDACTED]"
        restored = redactor.restore(result.redacted_text, result.mapping)
    """

    def __init__(
        self,
        config: Optional[PIIPatternConfig] = None,
        use_presidio: Optional[bool] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize PII Redactor.

        Args:
            config: Pattern configuration
            use_presidio: Whether to use presidio-analyzer (if available)
            mock_mode: Whether to use mock mode
        """
        self.config = config or PIIPatternConfig()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        # Determine whether to use presidio
        self._use_presidio = use_presidio if use_presidio is not None else self.config.use_presidio
        self._presidio_analyzer = None
        self._presidio_available = False

        if self._use_presidio and not self.mock_mode:
            self._initialize_presidio()

        # Compile regex patterns
        self._compiled_patterns = PIIPatterns.get_compiled_patterns()

        logger.info(f"PIIRedactor initialized (presidio={self._presidio_available}, mock={self.mock_mode})")

    def _initialize_presidio(self) -> None:
        """Initialize presidio analyzer if available."""
        try:
            from presidio_analyzer import AnalyzerEngine
            self._presidio_analyzer = AnalyzerEngine()
            self._presidio_available = True
            logger.info("Presidio analyzer loaded successfully")
        except ImportError:
            logger.warning("presidio-analyzer not installed, using regex-only detection")
            self._presidio_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize presidio: {e}")
            self._presidio_available = False

    def redact(self, text: str) -> RedactionResult:
        """
        Detect and redact PII from text.

        Args:
            text: Input text to redact

        Returns:
            RedactionResult with redacted text and entity information
        """
        if not text:
            return RedactionResult(
                redacted_text="",
                entities=[],
                mapping={},
                entity_count=0
            )

        entities = []

        # Use presidio if available
        if self._presidio_available and self._presidio_analyzer:
            entities = self._detect_with_presidio(text)

        # Also apply regex patterns
        regex_entities = self._detect_with_regex(text)

        # Merge entities (prefer presidio for overlapping)
        entities = self._merge_entities(entities, regex_entities)

        # Sort by position (reverse) for replacement
        entities.sort(key=lambda e: e.start, reverse=True)

        # Build mapping and redacted text
        redacted_text = text
        mapping = {}
        replacement_counter: Dict[str, int] = {}

        for entity in entities:
            # Create unique replacement if same type appears multiple times
            base_replacement = get_replacement(entity.entity_type)
            type_key = entity.entity_type.value

            if type_key in replacement_counter:
                replacement_counter[type_key] += 1
                unique_replacement = f"{base_replacement[:-1]}_{replacement_counter[type_key]}]"
            else:
                replacement_counter[type_key] = 1
                unique_replacement = base_replacement

            entity.replacement = unique_replacement
            mapping[unique_replacement] = entity.text

            # Replace in text
            redacted_text = (
                redacted_text[:entity.start] +
                unique_replacement +
                redacted_text[entity.end:]
            )

        return RedactionResult(
            redacted_text=redacted_text,
            entities=list(reversed(entities)),  # Restore original order
            mapping=mapping,
            entity_count=len(entities)
        )

    def _detect_with_presidio(self, text: str) -> List[PIIEntity]:
        """Detect PII using presidio analyzer."""
        entities = []

        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language='en',
                entities=None  # Detect all supported entities
            )

            for result in results:
                entity_type = self._map_presidio_entity_type(result.entity_type)
                if entity_type:
                    entities.append(PIIEntity(
                        entity_type=entity_type,
                        text=text[result.start:result.end],
                        start=result.start,
                        end=result.end,
                        confidence=result.score,
                        replacement=""
                    ))

        except Exception as e:
            logger.error(f"Presidio detection error: {e}")

        return entities

    def _detect_with_regex(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns."""
        entities = []

        for entity_type, pattern in self._compiled_patterns.items():
            if entity_type not in self.config.enabled_entities:
                continue

            for match in pattern.finditer(text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,  # Default confidence for regex
                    replacement=""
                ))

        return entities

    def _map_presidio_entity_type(self, presidio_type: str) -> Optional[PIIEntityType]:
        """Map presidio entity type to our entity type."""
        mapping = {
            "EMAIL_ADDRESS": PIIEntityType.EMAIL,
            "PHONE_NUMBER": PIIEntityType.PHONE,
            "US_SSN": PIIEntityType.SSN,
            "CREDIT_CARD": PIIEntityType.CREDIT_CARD,
            "IP_ADDRESS": PIIEntityType.IP_ADDRESS,
            "PERSON": PIIEntityType.PERSON,
            "LOCATION": PIIEntityType.LOCATION,
            "ORGANIZATION": PIIEntityType.ORGANIZATION,
            "DATE_TIME": PIIEntityType.DATE_OF_BIRTH
        }
        return mapping.get(presidio_type)

    def _merge_entities(
        self,
        primary: List[PIIEntity],
        secondary: List[PIIEntity]
    ) -> List[PIIEntity]:
        """Merge two lists of entities, preferring primary for overlaps."""
        if not primary:
            return secondary
        if not secondary:
            return primary

        merged = list(primary)
        primary_ranges = [(e.start, e.end) for e in primary]

        for entity in secondary:
            # Check for overlap with primary entities
            overlaps = any(
                not (entity.end <= start or entity.start >= end)
                for start, end in primary_ranges
            )
            if not overlaps:
                merged.append(entity)

        return merged

    def restore(self, redacted_text: str, mapping: Dict[str, str]) -> str:
        """
        Restore original text from redacted version.

        Args:
            redacted_text: Text with redacted PII
            mapping: Mapping from replacements to original values

        Returns:
            Restored original text
        """
        restored = redacted_text
        for replacement, original in mapping.items():
            restored = restored.replace(replacement, original)
        return restored

    def get_supported_entities(self) -> List[str]:
        """Get list of supported entity types."""
        return [e.value for e in PIIEntityType]


class MockPIIRedactor(PIIRedactor):
    """Mock PII redactor for testing."""

    def __init__(self):
        super().__init__(mock_mode=True, use_presidio=False)

    def redact(self, text: str) -> RedactionResult:
        """Simple mock redaction using regex only."""
        return super().redact(text)
