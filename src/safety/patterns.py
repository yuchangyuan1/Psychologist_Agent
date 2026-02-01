"""
Risk pattern loading and management for Safety Gateway.

This module handles loading and caching of risk patterns from JSON files
for semantic similarity matching in the safety checking pipeline.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logging_config import setup_logging

logger = setup_logging("patterns")


class RiskLevel(Enum):
    """Risk level enumeration."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskPattern:
    """A risk pattern with its category and risk level."""
    text: str
    category: str
    risk_level: RiskLevel


@dataclass
class RiskPatternDatabase:
    """Database of risk patterns organized by risk level."""
    high_risk: List[RiskPattern] = field(default_factory=list)
    moderate_risk: List[RiskPattern] = field(default_factory=list)
    low_risk: List[RiskPattern] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    def get_all_patterns(self) -> List[RiskPattern]:
        """Get all patterns in priority order (high first)."""
        return self.high_risk + self.moderate_risk + self.low_risk

    def get_patterns_by_level(self, level: RiskLevel) -> List[RiskPattern]:
        """Get patterns for a specific risk level."""
        level_map = {
            RiskLevel.HIGH: self.high_risk,
            RiskLevel.CRITICAL: self.high_risk,
            RiskLevel.MODERATE: self.moderate_risk,
            RiskLevel.LOW: self.low_risk
        }
        return level_map.get(level, [])


@dataclass
class CrisisResponse:
    """A crisis response template."""
    message: str
    resources: List[Dict[str, str]]
    action: str


@dataclass
class SafetyResponseDatabase:
    """Database of safety responses."""
    crisis_responses: Dict[str, CrisisResponse] = field(default_factory=dict)
    supportive_responses: Dict[str, Dict[str, str]] = field(default_factory=dict)
    follow_up_prompts: Dict[str, str] = field(default_factory=dict)


class PatternLoader:
    """
    Loader for risk patterns and safety responses.

    Handles loading from JSON files and caching the results.
    """

    _patterns_cache: Optional[RiskPatternDatabase] = None
    _responses_cache: Optional[SafetyResponseDatabase] = None

    def __init__(
        self,
        patterns_path: Optional[str] = None,
        responses_path: Optional[str] = None
    ):
        """
        Initialize pattern loader.

        Args:
            patterns_path: Path to risk_patterns.json
            responses_path: Path to safety_responses.json
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.patterns_path = patterns_path or os.path.join(
            base_dir, "data", "safety", "risk_patterns.json"
        )
        self.responses_path = responses_path or os.path.join(
            base_dir, "data", "safety", "safety_responses.json"
        )

    def load_patterns(self, force_reload: bool = False) -> RiskPatternDatabase:
        """
        Load risk patterns from JSON file.

        Args:
            force_reload: Force reload even if cached

        Returns:
            RiskPatternDatabase: Loaded patterns
        """
        if PatternLoader._patterns_cache is not None and not force_reload:
            return PatternLoader._patterns_cache

        try:
            with open(self.patterns_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Patterns file not found: {self.patterns_path}, using defaults")
            data = self._get_default_patterns()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing patterns file: {e}")
            data = self._get_default_patterns()

        db = RiskPatternDatabase()

        # Load high risk patterns
        for category, patterns in data.get("high_risk_patterns", {}).items():
            for text in patterns:
                db.high_risk.append(RiskPattern(
                    text=text,
                    category=category,
                    risk_level=RiskLevel.HIGH
                ))

        # Load moderate risk patterns
        for category, patterns in data.get("moderate_risk_patterns", {}).items():
            for text in patterns:
                db.moderate_risk.append(RiskPattern(
                    text=text,
                    category=category,
                    risk_level=RiskLevel.MODERATE
                ))

        # Load low risk patterns
        for category, patterns in data.get("low_risk_patterns", {}).items():
            for text in patterns:
                db.low_risk.append(RiskPattern(
                    text=text,
                    category=category,
                    risk_level=RiskLevel.LOW
                ))

        # Load thresholds
        db.thresholds = data.get("semantic_similarity_threshold", {
            "high_risk": 0.85,
            "moderate_risk": 0.80,
            "low_risk": 0.75
        })

        PatternLoader._patterns_cache = db
        logger.info(f"Loaded {len(db.get_all_patterns())} risk patterns")

        return db

    def load_responses(self, force_reload: bool = False) -> SafetyResponseDatabase:
        """
        Load safety responses from JSON file.

        Args:
            force_reload: Force reload even if cached

        Returns:
            SafetyResponseDatabase: Loaded responses
        """
        if PatternLoader._responses_cache is not None and not force_reload:
            return PatternLoader._responses_cache

        try:
            with open(self.responses_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Responses file not found: {self.responses_path}, using defaults")
            data = self._get_default_responses()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing responses file: {e}")
            data = self._get_default_responses()

        db = SafetyResponseDatabase()

        # Load crisis responses
        for key, response_data in data.get("crisis_responses", {}).items():
            db.crisis_responses[key] = CrisisResponse(
                message=response_data.get("message", ""),
                resources=response_data.get("resources", []),
                action=response_data.get("action", "")
            )

        # Load supportive responses
        db.supportive_responses = data.get("supportive_responses", {})

        # Load follow-up prompts
        db.follow_up_prompts = data.get("follow_up_prompts", {})

        PatternLoader._responses_cache = db
        logger.info(f"Loaded {len(db.crisis_responses)} crisis response templates")

        return db

    def _get_default_patterns(self) -> Dict[str, Any]:
        """Get default patterns if file not found."""
        return {
            "high_risk_patterns": {
                "self_harm": [
                    "I want to kill myself",
                    "I want to die",
                    "I'm going to hurt myself"
                ]
            },
            "moderate_risk_patterns": {
                "passive_suicidal_ideation": [
                    "I wish I didn't exist"
                ]
            },
            "low_risk_patterns": {
                "general_distress": [
                    "I'm feeling really down"
                ]
            },
            "semantic_similarity_threshold": {
                "high_risk": 0.85,
                "moderate_risk": 0.80,
                "low_risk": 0.75
            }
        }

    def _get_default_responses(self) -> Dict[str, Any]:
        """Get default responses if file not found."""
        return {
            "crisis_responses": {
                "immediate_danger": {
                    "message": "Please call 988 (Suicide & Crisis Lifeline) immediately.",
                    "resources": [{"name": "988 Lifeline", "phone": "988"}],
                    "action": "ESCALATE_IMMEDIATELY"
                }
            },
            "supportive_responses": {
                "moderate_risk": {
                    "prefix": "I hear that you're going through a difficult time.",
                    "suffix": "Would you like to talk more about what's happening?"
                }
            },
            "follow_up_prompts": {
                "safety_check": "Are you having any thoughts of hurting yourself?"
            }
        }

    @classmethod
    def clear_cache(cls):
        """Clear cached patterns and responses (for testing)."""
        cls._patterns_cache = None
        cls._responses_cache = None
