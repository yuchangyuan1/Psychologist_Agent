"""
Safety Gateway for mental health content screening.

This module provides the main SafetyGateway class that screens user input
for high-risk content using semantic similarity with BGE-small embeddings.
"""

import os
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.safety.embeddings import EmbeddingManager, EmbeddingConfig
from src.safety.patterns import (
    PatternLoader, RiskLevel, RiskPattern,
    RiskPatternDatabase, SafetyResponseDatabase, CrisisResponse
)
from src.utils.logging_config import setup_logging

logger = setup_logging("safety_gateway")


@dataclass
class SafetyResult:
    """Result from safety check."""
    is_safe: bool
    risk_level: RiskLevel
    matched_pattern: Optional[str] = None
    matched_category: Optional[str] = None
    similarity_score: float = 0.0
    response: Optional[str] = None
    resources: List[dict] = field(default_factory=list)
    action: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level.value,
            "matched_pattern": self.matched_pattern,
            "matched_category": self.matched_category,
            "similarity_score": self.similarity_score,
            "response": self.response,
            "resources": self.resources,
            "action": self.action
        }


class SafetyGateway:
    """
    Safety gateway for screening mental health conversations.

    Uses BGE-small embeddings to perform semantic similarity matching
    against known high-risk patterns to detect crisis situations.

    Example:
        gateway = SafetyGateway()
        result = await gateway.check("I'm feeling really sad today")
        if not result.is_safe:
            return result.response  # Crisis response
    """

    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        mock_mode: Optional[bool] = None,
        patterns_path: Optional[str] = None,
        responses_path: Optional[str] = None
    ):
        """
        Initialize the Safety Gateway.

        Args:
            embedding_config: Configuration for embedding model
            mock_mode: Whether to use mock mode (for testing)
            patterns_path: Path to risk patterns JSON
            responses_path: Path to safety responses JSON
        """
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            config=embedding_config,
            mock_mode=self.mock_mode
        )

        # Load patterns and responses
        self.pattern_loader = PatternLoader(patterns_path, responses_path)
        self.patterns_db: RiskPatternDatabase = self.pattern_loader.load_patterns()
        self.responses_db: SafetyResponseDatabase = self.pattern_loader.load_responses()

        # Pre-compute pattern embeddings
        self._pattern_embeddings: Optional[np.ndarray] = None
        self._pattern_list: List[RiskPattern] = []

        logger.info(f"SafetyGateway initialized (mock_mode={self.mock_mode})")

    def _ensure_embeddings_cached(self):
        """Ensure pattern embeddings are computed and cached."""
        if self._pattern_embeddings is not None:
            return

        self._pattern_list = self.patterns_db.get_all_patterns()
        if not self._pattern_list:
            logger.warning("No risk patterns loaded")
            return

        pattern_texts = [p.text for p in self._pattern_list]
        self._pattern_embeddings = self.embedding_manager.encode(pattern_texts)

        logger.info(f"Cached embeddings for {len(pattern_texts)} patterns")

    async def check(self, user_input: str) -> SafetyResult:
        """
        Check user input for safety concerns.

        Args:
            user_input: The user's message to check

        Returns:
            SafetyResult: Result containing safety assessment and any crisis response
        """
        if not user_input or not user_input.strip():
            return SafetyResult(is_safe=True, risk_level=RiskLevel.NONE)

        # Ensure pattern embeddings are cached
        self._ensure_embeddings_cached()

        if self._pattern_embeddings is None or len(self._pattern_list) == 0:
            logger.warning("No patterns available for safety check")
            return SafetyResult(is_safe=True, risk_level=RiskLevel.NONE)

        # Encode user input
        user_embedding = self.embedding_manager.encode(user_input)

        # Compute similarities
        similarities = self.embedding_manager.similarity(
            user_embedding,
            self._pattern_embeddings
        )

        # Find best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        best_pattern = self._pattern_list[best_idx]

        # Determine risk level based on thresholds
        thresholds = self.patterns_db.thresholds

        # Check against thresholds in order of severity
        if best_pattern.risk_level == RiskLevel.HIGH and best_score >= thresholds.get("high_risk", 0.85):
            return self._create_high_risk_result(best_pattern, best_score)
        elif best_pattern.risk_level == RiskLevel.MODERATE and best_score >= thresholds.get("moderate_risk", 0.80):
            return self._create_moderate_risk_result(best_pattern, best_score)
        elif best_pattern.risk_level == RiskLevel.LOW and best_score >= thresholds.get("low_risk", 0.75):
            return self._create_low_risk_result(best_pattern, best_score)

        # Also check if high-risk patterns have moderate similarity
        high_risk_patterns = [i for i, p in enumerate(self._pattern_list) if p.risk_level == RiskLevel.HIGH]
        if high_risk_patterns:
            high_risk_scores = similarities[high_risk_patterns]
            max_high_risk_idx = high_risk_patterns[np.argmax(high_risk_scores)]
            max_high_risk_score = float(np.max(high_risk_scores))

            # Lower threshold for high-risk patterns
            if max_high_risk_score >= thresholds.get("moderate_risk", 0.80):
                pattern = self._pattern_list[max_high_risk_idx]
                return self._create_moderate_risk_result(pattern, max_high_risk_score)

        return SafetyResult(
            is_safe=True,
            risk_level=RiskLevel.NONE,
            similarity_score=best_score
        )

    def _create_high_risk_result(
        self,
        pattern: RiskPattern,
        score: float
    ) -> SafetyResult:
        """Create result for high-risk match."""
        # Map category to crisis response
        category_map = {
            "self_harm": "self_harm_intent",
            "active_crisis": "immediate_danger",
            "harm_to_others": "harm_to_others",
            "abuse_disclosure": "abuse_disclosure"
        }

        response_key = category_map.get(pattern.category, "immediate_danger")
        crisis_response = self.responses_db.crisis_responses.get(
            response_key,
            self.responses_db.crisis_responses.get("immediate_danger")
        )

        if crisis_response:
            return SafetyResult(
                is_safe=False,
                risk_level=RiskLevel.CRITICAL if pattern.category == "active_crisis" else RiskLevel.HIGH,
                matched_pattern=pattern.text,
                matched_category=pattern.category,
                similarity_score=score,
                response=crisis_response.message,
                resources=crisis_response.resources,
                action=crisis_response.action
            )

        return SafetyResult(
            is_safe=False,
            risk_level=RiskLevel.HIGH,
            matched_pattern=pattern.text,
            matched_category=pattern.category,
            similarity_score=score,
            response="Please call 988 (Suicide & Crisis Lifeline) for immediate support.",
            action="ESCALATE_IMMEDIATELY"
        )

    def _create_moderate_risk_result(
        self,
        pattern: RiskPattern,
        score: float
    ) -> SafetyResult:
        """Create result for moderate-risk match."""
        supportive = self.responses_db.supportive_responses.get("moderate_risk", {})

        return SafetyResult(
            is_safe=True,  # Allow conversation but with caution
            risk_level=RiskLevel.MODERATE,
            matched_pattern=pattern.text,
            matched_category=pattern.category,
            similarity_score=score,
            response=supportive.get("prefix", ""),
            action="CONTINUE_WITH_CAUTION"
        )

    def _create_low_risk_result(
        self,
        pattern: RiskPattern,
        score: float
    ) -> SafetyResult:
        """Create result for low-risk match."""
        supportive = self.responses_db.supportive_responses.get("low_risk", {})

        return SafetyResult(
            is_safe=True,
            risk_level=RiskLevel.LOW,
            matched_pattern=pattern.text,
            matched_category=pattern.category,
            similarity_score=score,
            response=supportive.get("prefix", ""),
            action="NORMAL"
        )

    def get_follow_up_prompt(self, prompt_type: str) -> Optional[str]:
        """Get a follow-up prompt by type."""
        return self.responses_db.follow_up_prompts.get(prompt_type)

    def get_crisis_resources(self, category: str = "immediate_danger") -> List[dict]:
        """Get crisis resources for a category."""
        response = self.responses_db.crisis_responses.get(category)
        if response:
            return response.resources
        return []
