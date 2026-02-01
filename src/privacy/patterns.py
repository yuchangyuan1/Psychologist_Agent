"""
PII detection patterns and entity definitions.

This module provides regex patterns and entity configurations
for detecting personally identifiable information.
"""

import re
from typing import Dict, List, Pattern
from dataclasses import dataclass, field
from enum import Enum


class PIIEntityType(Enum):
    """Types of PII entities."""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    NAME = "NAME"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"


@dataclass
class PIIPattern:
    """A PII detection pattern."""
    entity_type: PIIEntityType
    pattern: str
    description: str
    confidence: float = 0.9
    replacement_template: str = "[{entity_type}]"


@dataclass
class PIIPatternConfig:
    """Configuration for PII patterns."""
    enabled_entities: List[PIIEntityType] = field(default_factory=lambda: list(PIIEntityType))
    custom_patterns: List[PIIPattern] = field(default_factory=list)
    use_presidio: bool = True


# Pre-compiled regex patterns for common PII types
class PIIPatterns:
    """Collection of PII detection patterns."""

    # Email pattern
    EMAIL = PIIPattern(
        entity_type=PIIEntityType.EMAIL,
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        description="Email address",
        confidence=0.95
    )

    # Phone patterns (US formats)
    PHONE_US = PIIPattern(
        entity_type=PIIEntityType.PHONE,
        pattern=r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        description="US phone number",
        confidence=0.85
    )

    # Social Security Number
    SSN = PIIPattern(
        entity_type=PIIEntityType.SSN,
        pattern=r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        description="Social Security Number",
        confidence=0.9
    )

    # Credit Card (basic pattern)
    CREDIT_CARD = PIIPattern(
        entity_type=PIIEntityType.CREDIT_CARD,
        pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        description="Credit card number",
        confidence=0.85
    )

    # IP Address
    IP_ADDRESS = PIIPattern(
        entity_type=PIIEntityType.IP_ADDRESS,
        pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        description="IPv4 address",
        confidence=0.9
    )

    # Date of Birth patterns
    DOB = PIIPattern(
        entity_type=PIIEntityType.DATE_OF_BIRTH,
        pattern=r'\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
        description="Date of birth (MM/DD/YYYY)",
        confidence=0.7
    )

    # Medical Record Number (basic pattern)
    MEDICAL_RECORD = PIIPattern(
        entity_type=PIIEntityType.MEDICAL_RECORD,
        pattern=r'\b(?:MRN|MR#|Medical Record)[:\s]*\d{6,10}\b',
        description="Medical record number",
        confidence=0.85
    )

    @classmethod
    def get_all_patterns(cls) -> List[PIIPattern]:
        """Get all defined patterns."""
        return [
            cls.EMAIL,
            cls.PHONE_US,
            cls.SSN,
            cls.CREDIT_CARD,
            cls.IP_ADDRESS,
            cls.DOB,
            cls.MEDICAL_RECORD
        ]

    @classmethod
    def get_compiled_patterns(cls) -> Dict[PIIEntityType, Pattern]:
        """Get compiled regex patterns."""
        return {
            p.entity_type: re.compile(p.pattern, re.IGNORECASE)
            for p in cls.get_all_patterns()
        }


# Replacement templates for different entity types
REPLACEMENT_TEMPLATES = {
    PIIEntityType.EMAIL: "[EMAIL_REDACTED]",
    PIIEntityType.PHONE: "[PHONE_REDACTED]",
    PIIEntityType.SSN: "[SSN_REDACTED]",
    PIIEntityType.CREDIT_CARD: "[CARD_REDACTED]",
    PIIEntityType.IP_ADDRESS: "[IP_REDACTED]",
    PIIEntityType.NAME: "[NAME_REDACTED]",
    PIIEntityType.ADDRESS: "[ADDRESS_REDACTED]",
    PIIEntityType.DATE_OF_BIRTH: "[DOB_REDACTED]",
    PIIEntityType.MEDICAL_RECORD: "[MRN_REDACTED]",
    PIIEntityType.PERSON: "[PERSON_REDACTED]",
    PIIEntityType.LOCATION: "[LOCATION_REDACTED]",
    PIIEntityType.ORGANIZATION: "[ORG_REDACTED]"
}


def get_replacement(entity_type: PIIEntityType) -> str:
    """Get replacement text for an entity type."""
    return REPLACEMENT_TEMPLATES.get(entity_type, f"[{entity_type.value}_REDACTED]")
