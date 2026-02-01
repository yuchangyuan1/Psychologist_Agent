"""
Risk Checker for validating cloud analysis responses.

This module provides the RiskChecker class for assessing risk levels
from cloud analysis and determining appropriate interventions.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.api.models import AnalysisResult, RiskLevel, TherapeuticApproach
from src.utils.logging_config import setup_logging

logger = setup_logging("risk_checker")


class InterventionLevel(Enum):
    """Level of intervention required."""
    NONE = "none"
    MONITOR = "monitor"
    SUPPORTIVE = "supportive"
    ACTIVE = "active"
    CRISIS = "crisis"
    EMERGENCY = "emergency"


@dataclass
class RiskAssessment:
    """Result from risk assessment."""
    risk_level: RiskLevel
    intervention_level: InterventionLevel
    requires_crisis_response: bool
    requires_escalation: bool
    confidence: float
    concerns: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "intervention_level": self.intervention_level.value,
            "requires_crisis_response": self.requires_crisis_response,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "concerns": self.concerns,
            "recommended_actions": self.recommended_actions,
            "flags": self.flags
        }


@dataclass
class RiskCheckerConfig:
    """Configuration for risk checker."""
    auto_escalate_critical: bool = True
    auto_escalate_high: bool = False
    confidence_threshold: float = 0.7
    enable_keyword_check: bool = True


class RiskChecker:
    """
    Risk checker for validating and enhancing cloud analysis.

    Validates the cloud analysis result against the original input
    and applies additional safety checks.

    Example:
        checker = RiskChecker()
        assessment = checker.assess(cloud_analysis, original_input)
        if assessment.requires_crisis_response:
            return crisis_handler.get_response(assessment)
    """

    # Risk level severity order (higher = more severe)
    RISK_SEVERITY = {
        RiskLevel.LOW: 1,
        RiskLevel.MODERATE: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4
    }

    # High-risk keywords that should trigger additional checks
    CRITICAL_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die",
        "kill someone", "hurt someone", "murder"
    ]

    HIGH_RISK_KEYWORDS = [
        "self-harm", "cutting", "hurt myself", "overdose",
        "hopeless", "no reason to live", "better off dead"
    ]

    CRISIS_INDICATORS = [
        "right now", "tonight", "today", "about to",
        "have a plan", "already started", "holding"
    ]

    def __init__(self, config: Optional[RiskCheckerConfig] = None):
        """
        Initialize risk checker.

        Args:
            config: Risk checker configuration
        """
        self.config = config or RiskCheckerConfig()
        logger.info("RiskChecker initialized")

    def assess(
        self,
        cloud_analysis: AnalysisResult,
        original_input: str
    ) -> RiskAssessment:
        """
        Assess risk from cloud analysis and original input.

        Args:
            cloud_analysis: Analysis result from cloud API
            original_input: Original user input

        Returns:
            RiskAssessment: Comprehensive risk assessment
        """
        # Start with cloud analysis risk level
        risk_level = cloud_analysis.risk_level
        concerns = [cloud_analysis.primary_concern]
        flags = {}

        # Apply keyword checks for additional safety
        if self.config.enable_keyword_check:
            keyword_risk, keyword_flags = self._check_keywords(original_input)
            flags.update(keyword_flags)

            # Upgrade risk level if keywords indicate higher risk
            if self.RISK_SEVERITY.get(keyword_risk, 0) > self.RISK_SEVERITY.get(risk_level, 0):
                risk_level = keyword_risk
                concerns.append("Keyword detection indicates elevated risk")

        # Check for immediacy indicators
        has_immediacy = self._check_immediacy(original_input)
        flags["immediate_concern"] = has_immediacy

        if has_immediacy and risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            flags["urgent"] = True
            concerns.append("Immediate action indicators detected")

        # Determine intervention level
        intervention_level = self._determine_intervention(risk_level, flags)

        # Determine if crisis response needed
        requires_crisis = intervention_level in [
            InterventionLevel.CRISIS,
            InterventionLevel.EMERGENCY
        ]

        # Determine if escalation needed
        requires_escalation = (
            (risk_level == RiskLevel.CRITICAL and self.config.auto_escalate_critical) or
            (risk_level == RiskLevel.HIGH and self.config.auto_escalate_high) or
            flags.get("urgent", False)
        )

        # Generate recommended actions
        recommended_actions = self._get_recommended_actions(
            risk_level, intervention_level, flags
        )

        return RiskAssessment(
            risk_level=risk_level,
            intervention_level=intervention_level,
            requires_crisis_response=requires_crisis,
            requires_escalation=requires_escalation,
            confidence=cloud_analysis.confidence,
            concerns=concerns,
            recommended_actions=recommended_actions,
            flags=flags
        )

    def _check_keywords(self, text: str) -> tuple:
        """Check for high-risk keywords."""
        text_lower = text.lower()
        flags = {
            "critical_keywords": False,
            "high_risk_keywords": False
        }

        # Check for critical keywords
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in text_lower:
                flags["critical_keywords"] = True
                return RiskLevel.CRITICAL, flags

        # Check for high-risk keywords
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in text_lower:
                flags["high_risk_keywords"] = True
                return RiskLevel.HIGH, flags

        return RiskLevel.LOW, flags

    def _check_immediacy(self, text: str) -> bool:
        """Check for immediacy indicators."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.CRISIS_INDICATORS)

    def _determine_intervention(
        self,
        risk_level: RiskLevel,
        flags: Dict[str, bool]
    ) -> InterventionLevel:
        """Determine appropriate intervention level."""
        if risk_level == RiskLevel.CRITICAL:
            if flags.get("urgent", False):
                return InterventionLevel.EMERGENCY
            return InterventionLevel.CRISIS

        if risk_level == RiskLevel.HIGH:
            if flags.get("immediate_concern", False):
                return InterventionLevel.CRISIS
            return InterventionLevel.ACTIVE

        if risk_level == RiskLevel.MODERATE:
            return InterventionLevel.SUPPORTIVE

        if risk_level == RiskLevel.LOW:
            return InterventionLevel.MONITOR

        return InterventionLevel.NONE

    def _get_recommended_actions(
        self,
        risk_level: RiskLevel,
        intervention_level: InterventionLevel,
        flags: Dict[str, bool]
    ) -> List[str]:
        """Get recommended actions based on assessment."""
        actions = []

        if intervention_level == InterventionLevel.EMERGENCY:
            actions.extend([
                "Provide immediate crisis resources (988, 911)",
                "Express urgent concern for safety",
                "Stay engaged until user contacts crisis line",
                "Log for clinical review"
            ])

        elif intervention_level == InterventionLevel.CRISIS:
            actions.extend([
                "Provide crisis resources (988)",
                "Express concern for safety",
                "Conduct safety check",
                "Encourage professional help"
            ])

        elif intervention_level == InterventionLevel.ACTIVE:
            actions.extend([
                "Validate feelings and concerns",
                "Explore current situation",
                "Mention crisis resources as available",
                "Suggest professional support"
            ])

        elif intervention_level == InterventionLevel.SUPPORTIVE:
            actions.extend([
                "Provide empathetic listening",
                "Validate emotions",
                "Explore coping strategies",
                "Monitor for escalation"
            ])

        else:
            actions.extend([
                "Continue supportive conversation",
                "Build rapport",
                "Monitor for changes"
            ])

        return actions

    def validate_analysis(
        self,
        cloud_analysis: AnalysisResult,
        original_input: str
    ) -> bool:
        """
        Validate that cloud analysis is consistent with input.

        Args:
            cloud_analysis: Analysis from cloud API
            original_input: Original user input

        Returns:
            bool: True if analysis appears valid
        """
        # Check if critical keywords are present but risk is low
        text_lower = original_input.lower()

        has_critical = any(kw in text_lower for kw in self.CRITICAL_KEYWORDS)
        has_high_risk = any(kw in text_lower for kw in self.HIGH_RISK_KEYWORDS)

        if has_critical and cloud_analysis.risk_level == RiskLevel.LOW:
            logger.warning("Validation failed: Critical keywords with LOW risk")
            return False

        if has_high_risk and cloud_analysis.risk_level == RiskLevel.LOW:
            logger.warning("Validation failed: High-risk keywords with LOW risk")
            return False

        return True
