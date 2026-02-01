"""
Tests for Risk Audit module.

All tests use MOCK mode.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.audit.risk_checker import (
    RiskChecker, RiskCheckerConfig, RiskAssessment, InterventionLevel
)
from src.audit.crisis_handler import CrisisHandler, CrisisResponse, CrisisResource
from src.audit.logger import AuditLogger, AuditLoggerConfig, AuditEvent, EventType
from src.api.models import AnalysisResult, RiskLevel, TherapeuticApproach


class TestRiskChecker:
    """Tests for RiskChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a risk checker."""
        return RiskChecker()

    @pytest.fixture
    def low_risk_analysis(self):
        """Create a low-risk analysis result."""
        return AnalysisResult(
            risk_level=RiskLevel.LOW,
            primary_concern="general stress",
            suggested_approach=TherapeuticApproach.SUPPORTIVE,
            key_points=["validate feelings"],
            confidence=0.9
        )

    @pytest.fixture
    def high_risk_analysis(self):
        """Create a high-risk analysis result."""
        return AnalysisResult(
            risk_level=RiskLevel.HIGH,
            primary_concern="self-harm risk",
            suggested_approach=TherapeuticApproach.DBT,
            key_points=["provide crisis resources"],
            confidence=0.85
        )

    def test_assess_low_risk(self, checker, low_risk_analysis):
        """Test assessment of low-risk input."""
        assessment = checker.assess(
            low_risk_analysis,
            "I'm feeling stressed about work"
        )

        assert assessment.risk_level == RiskLevel.LOW
        assert assessment.intervention_level == InterventionLevel.MONITOR
        assert not assessment.requires_crisis_response
        assert not assessment.requires_escalation

    def test_assess_high_risk(self, checker, high_risk_analysis):
        """Test assessment of high-risk input."""
        assessment = checker.assess(
            high_risk_analysis,
            "I want to hurt myself"
        )

        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.intervention_level in [
            InterventionLevel.ACTIVE,
            InterventionLevel.CRISIS
        ]

    def test_keyword_escalation(self, checker, low_risk_analysis):
        """Test that critical keywords escalate risk."""
        assessment = checker.assess(
            low_risk_analysis,
            "I want to kill myself"
        )

        # Should escalate despite low analysis
        assert assessment.risk_level == RiskLevel.CRITICAL
        assert assessment.requires_crisis_response

    def test_immediacy_detection(self, checker, high_risk_analysis):
        """Test detection of immediacy indicators."""
        assessment = checker.assess(
            high_risk_analysis,
            "I'm about to hurt myself right now"
        )

        assert assessment.flags.get("immediate_concern")
        assert assessment.flags.get("urgent")
        # HIGH + immediate_concern = CRISIS (EMERGENCY requires CRITICAL level)
        assert assessment.intervention_level == InterventionLevel.CRISIS

    def test_validate_analysis(self, checker, low_risk_analysis):
        """Test analysis validation."""
        # Should fail - critical keywords with low risk
        is_valid = checker.validate_analysis(
            low_risk_analysis,
            "I want to kill myself"
        )
        assert not is_valid

        # Should pass - low keywords with low risk
        is_valid = checker.validate_analysis(
            low_risk_analysis,
            "I'm stressed about work"
        )
        assert is_valid

    def test_assessment_to_dict(self, checker, low_risk_analysis):
        """Test RiskAssessment.to_dict()."""
        assessment = checker.assess(
            low_risk_analysis,
            "Just feeling down"
        )

        result_dict = assessment.to_dict()
        assert "risk_level" in result_dict
        assert "intervention_level" in result_dict
        assert "recommended_actions" in result_dict

    def test_recommended_actions(self, checker, high_risk_analysis):
        """Test that appropriate actions are recommended."""
        assessment = checker.assess(
            high_risk_analysis,
            "I've been thinking about hurting myself"
        )

        assert len(assessment.recommended_actions) > 0
        # Should include some form of crisis resource mention
        actions_text = " ".join(assessment.recommended_actions).lower()
        assert any(word in actions_text for word in ["crisis", "resource", "professional"])


class TestCrisisHandler:
    """Tests for CrisisHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a crisis handler."""
        return CrisisHandler()

    @pytest.fixture
    def crisis_assessment(self):
        """Create a crisis-level assessment."""
        return RiskAssessment(
            risk_level=RiskLevel.CRITICAL,
            intervention_level=InterventionLevel.CRISIS,
            requires_crisis_response=True,
            requires_escalation=True,
            confidence=0.9,
            concerns=["suicidal ideation"],
            recommended_actions=["Provide crisis resources"]
        )

    def test_get_response(self, handler, crisis_assessment):
        """Test getting crisis response."""
        response = handler.get_response(crisis_assessment)

        assert isinstance(response, CrisisResponse)
        assert len(response.message) > 0
        assert len(response.resources) > 0

    def test_response_includes_988(self, handler, crisis_assessment):
        """Test that response includes 988 hotline."""
        response = handler.get_response(crisis_assessment)

        # Should include 988 in message or resources
        message_has_988 = "988" in response.message
        resources_have_988 = any("988" in r.name or r.phone == "988" for r in response.resources)

        assert message_has_988 or resources_have_988

    def test_specific_crisis_type(self, handler, crisis_assessment):
        """Test response for specific crisis type."""
        response = handler.get_response(crisis_assessment, crisis_type="suicidal_ideation")

        assert response.response_type == "suicidal_ideation"
        assert response.requires_escalation

    def test_abuse_resources(self, handler, crisis_assessment):
        """Test that abuse disclosure includes specific resources."""
        crisis_assessment.concerns = ["abuse disclosure"]
        response = handler.get_response(crisis_assessment, crisis_type="abuse_disclosure")

        # Should include domestic violence or similar resources
        resource_names = [r.name for r in response.resources]
        assert len(response.resources) > 1

    def test_get_validation_message(self, handler):
        """Test getting validation message."""
        message = handler.get_validation_message()
        assert isinstance(message, str)
        assert len(message) > 0

    def test_get_safety_check_questions(self, handler):
        """Test getting safety check questions."""
        questions = handler.get_safety_check_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0

    def test_format_resources_text(self, handler, crisis_assessment):
        """Test formatting resources as text."""
        response = handler.get_response(crisis_assessment)
        text = handler.format_resources_text(response.resources)

        assert "988" in text or "Crisis" in text

    def test_response_to_dict(self, handler, crisis_assessment):
        """Test CrisisResponse.to_dict()."""
        response = handler.get_response(crisis_assessment)
        result_dict = response.to_dict()

        assert "message" in result_dict
        assert "resources" in result_dict
        assert "requires_escalation" in result_dict


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create an audit logger with temp directory."""
        config = AuditLoggerConfig(
            log_dir=str(tmp_path / "audit"),
            log_to_console=False
        )
        return AuditLogger(config)

    def test_log_api_call(self, logger):
        """Test logging API call."""
        logger.log_api_call(
            session_id="test-session",
            endpoint="/chat",
            status=200,
            latency_ms=150.5,
            tokens_used=100
        )

        events = logger.get_recent_events(session_id="test-session")
        assert len(events) == 1
        assert events[0].event_type == EventType.API_CALL

    def test_log_safety_check(self, logger):
        """Test logging safety check."""
        logger.log_safety_check(
            session_id="test-session",
            risk_level="low",
            is_safe=True,
            matched_pattern=None,
            action_taken="continue"
        )

        events = logger.get_recent_events(event_type=EventType.SAFETY_CHECK)
        assert len(events) == 1

    def test_log_crisis_intervention(self, logger):
        """Test logging crisis intervention."""
        logger.log_crisis_intervention(
            session_id="test-session",
            trigger="suicidal_ideation",
            resources_provided=["988", "Crisis Text Line"],
            escalated=True
        )

        events = logger.get_recent_events(event_type=EventType.CRISIS_INTERVENTION)
        assert len(events) == 1
        assert events[0].data["escalated"] is True

    def test_log_pii_redaction(self, logger):
        """Test logging PII redaction (counts only)."""
        logger.log_pii_redaction(
            session_id="test-session",
            entity_count=2,
            entity_types=["EMAIL", "PHONE"]
        )

        events = logger.get_recent_events(event_type=EventType.PII_REDACTION)
        assert len(events) == 1
        assert events[0].data["entity_count"] == 2

    def test_log_error(self, logger):
        """Test logging errors."""
        logger.log_error(
            session_id="test-session",
            error_type="APIError",
            error_message="Rate limit exceeded"
        )

        events = logger.get_recent_events(event_type=EventType.ERROR)
        assert len(events) == 1

    def test_session_summary(self, logger):
        """Test getting session summary."""
        session_id = "summary-test"

        logger.log_api_call(session_id, "/chat", 200, 100, 50)
        logger.log_safety_check(session_id, "low", True, None, "continue")
        logger.log_safety_check(session_id, "moderate", True, None, "continue")

        summary = logger.get_session_summary(session_id)

        assert summary["session_id"] == session_id
        assert summary["event_count"] == 3
        assert summary["api_calls"] == 1
        assert summary["safety_checks"] == 2

    def test_event_to_json(self, logger):
        """Test AuditEvent.to_json()."""
        event = AuditEvent(
            event_type=EventType.API_CALL,
            session_id="test",
            data={"status": 200}
        )

        json_str = event.to_json()
        assert "api_call" in json_str
        assert "test" in json_str

    def test_disabled_logger(self, tmp_path):
        """Test that disabled logger doesn't log."""
        config = AuditLoggerConfig(enabled=False)
        logger = AuditLogger(config)

        logger.log_api_call("session", "/chat", 200, 100, 50)
        events = logger.get_recent_events()

        assert len(events) == 0


class TestIntegration:
    """Integration tests for audit module."""

    def test_full_risk_assessment_flow(self):
        """Test complete risk assessment and crisis response flow."""
        checker = RiskChecker()
        handler = CrisisHandler()

        # Simulate cloud analysis
        analysis = AnalysisResult(
            risk_level=RiskLevel.HIGH,
            primary_concern="self-harm ideation",
            suggested_approach=TherapeuticApproach.DBT,
            key_points=["provide resources", "validate feelings"],
            confidence=0.85
        )

        # Assess risk
        assessment = checker.assess(
            analysis,
            "I've been thinking about hurting myself lately"
        )

        # Get crisis response if needed
        if assessment.requires_crisis_response:
            response = handler.get_response(assessment)

            assert "988" in response.message or any(
                "988" in r.phone for r in response.resources if r.phone
            )

    def test_audit_logging_integration(self, tmp_path):
        """Test audit logging with risk assessment."""
        config = AuditLoggerConfig(log_dir=str(tmp_path / "audit"))
        logger = AuditLogger(config)
        checker = RiskChecker()

        analysis = AnalysisResult(
            risk_level=RiskLevel.MODERATE,
            primary_concern="anxiety",
            suggested_approach=TherapeuticApproach.CBT,
            key_points=["explore triggers"],
            confidence=0.9
        )

        assessment = checker.assess(analysis, "I'm feeling anxious")

        # Log the assessment
        logger.log_risk_assessment(
            session_id="integration-test",
            risk_level=assessment.risk_level.value,
            primary_concern=analysis.primary_concern,
            approach=analysis.suggested_approach.value,
            key_points=analysis.key_points
        )

        events = logger.get_recent_events(event_type=EventType.RISK_ASSESSMENT)
        assert len(events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
