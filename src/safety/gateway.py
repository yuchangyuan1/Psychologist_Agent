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

    # High-confidence keywords for hard-check (fast path, no embeddings needed)
    _HIGH_RISK_KEYWORDS: List[str] = [
        # Direct self-harm / suicide
        "kill myself", "killing myself", "end my life", "take my life",
        "want to die", "going to die", "planning to die",
        "suicide", "suicidal",
        "overdose on", "overdose myself",
        "jump off", "hang myself", "slit my wrists", "slash my wrists",
        "shoot myself", "stab myself",
        "nobody would care if i", "no one would care if i",
        "better off dead", "better off without me",
        "life is not worth living", "life isn't worth living",
        "goodbye world", "goodbye cruel world",
        "medication ready for tonight", "pills ready",
        # Violence / harm to others
        "want to hurt", "going to hurt", "plan to hurt",
        "want to kill", "going to kill", "plan to kill",
        "kill my", "murder my",
        "get revenge on someone physically", "get physical revenge",
        "thoughts of attacking",
        "feel like punching", "feel like hitting", "feel like attacking",
    ]

    _MODERATE_RISK_KEYWORDS: List[str] = [
        "thinking about suicide", "thoughts of suicide", "suicidal thoughts",
        "don't want to live", "don't want to be alive", "wish i was dead",
        "wish i were dead", "wish i wasn't here", "wish i wasn't alive",
        "hurt myself", "harm myself", "self-harm", "self harm",
        "cut myself", "cutting myself",
        # Indirect crisis expressions
        "can't take this pain anymore", "cannot take this pain",
        "want to leave this world", "want to disappear forever",
        # Medical advice requests (inappropriate for AI)
        "prescribe me", "give me a prescription",
        "antidepressants without a doctor", "medication without prescription",
        "mix alcohol and", "safe to mix alcohol",
        "dosage for lexapro", "dosage for prozac", "dosage for xanax",
        "give me a medical diagnosis",
    ]

    def _keyword_check(self, text: str) -> Optional[SafetyResult]:
        """Fast keyword hard-check before semantic matching."""
        lower = text.lower()

        for kw in self._HIGH_RISK_KEYWORDS:
            if kw in lower:
                pattern = RiskPattern(text=kw, category="self_harm", risk_level=RiskLevel.HIGH)
                logger.info(f"Keyword hard-check triggered: '{kw}'")
                return self._create_high_risk_result(pattern, 1.0)

        for kw in self._MODERATE_RISK_KEYWORDS:
            if kw in lower:
                pattern = RiskPattern(text=kw, category="passive_suicidal_ideation", risk_level=RiskLevel.MODERATE)
                logger.info(f"Moderate keyword triggered: '{kw}'")
                return self._create_moderate_risk_result(pattern, 1.0)

        return None

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

        # Step 1: Fast keyword hard-check (catches obvious direct expressions)
        keyword_result = self._keyword_check(user_input)
        if keyword_result is not None:
            return keyword_result

        # Step 2: Semantic similarity check
        self._ensure_embeddings_cached()

        if self._pattern_embeddings is None or len(self._pattern_list) == 0:
            logger.warning("No patterns available for safety check")
            return SafetyResult(is_safe=True, risk_level=RiskLevel.NONE)

        user_embedding = self.embedding_manager.encode(user_input)
        similarities = self.embedding_manager.similarity(
            user_embedding,
            self._pattern_embeddings
        )

        thresholds = self.patterns_db.thresholds
        high_th = thresholds.get("high_risk", 0.72)
        moderate_th = thresholds.get("moderate_risk", 0.68)
        low_th = thresholds.get("low_risk", 0.62)

        # Step 3: Check each risk category independently (fix: don't rely on global argmax)
        def best_in_category(level: RiskLevel) -> Tuple[int, float]:
            indices = [i for i, p in enumerate(self._pattern_list) if p.risk_level == level]
            if not indices:
                return -1, 0.0
            scores = similarities[indices]
            best = int(np.argmax(scores))
            return indices[best], float(scores[best])

        high_idx, high_score = best_in_category(RiskLevel.HIGH)
        moderate_idx, moderate_score = best_in_category(RiskLevel.MODERATE)
        low_idx, low_score = best_in_category(RiskLevel.LOW)

        # Step 4: Top-k consensus — if top-3 patterns are all HIGH risk, lower effective threshold
        top3_indices = np.argsort(similarities)[-3:][::-1]
        top3_levels = [self._pattern_list[i].risk_level for i in top3_indices]
        consensus_bonus = 0.04 if top3_levels.count(RiskLevel.HIGH) >= 2 else 0.0

        # Evaluate in order of severity
        if high_idx >= 0 and high_score >= (high_th - consensus_bonus):
            return self._create_high_risk_result(self._pattern_list[high_idx], high_score)

        if moderate_idx >= 0 and moderate_score >= moderate_th:
            return self._create_moderate_risk_result(self._pattern_list[moderate_idx], moderate_score)

        # High-risk patterns at moderate similarity → treat as moderate
        if high_idx >= 0 and high_score >= moderate_th:
            return self._create_moderate_risk_result(self._pattern_list[high_idx], high_score)

        if low_idx >= 0 and low_score >= low_th:
            return self._create_low_risk_result(self._pattern_list[low_idx], low_score)

        best_idx = int(np.argmax(similarities))
        return SafetyResult(
            is_safe=True,
            risk_level=RiskLevel.NONE,
            similarity_score=float(similarities[best_idx])
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
