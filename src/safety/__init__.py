"""
Safety module for mental health content screening.

This module provides safety gateway functionality to detect high-risk
content in user messages and provide appropriate crisis responses.
"""

from src.safety.gateway import SafetyGateway, SafetyResult
from src.safety.patterns import RiskLevel, RiskPattern, PatternLoader
from src.safety.embeddings import EmbeddingManager, EmbeddingConfig

__all__ = [
    "SafetyGateway",
    "SafetyResult",
    "RiskLevel",
    "RiskPattern",
    "PatternLoader",
    "EmbeddingManager",
    "EmbeddingConfig"
]
