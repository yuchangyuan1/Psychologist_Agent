"""
Tests for PII Redactor module.

All tests use MOCK mode - no presidio or real NER.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.privacy.pii_redactor import PIIRedactor, RedactionResult, MockPIIRedactor
from src.privacy.patterns import PIIEntityType, PIIPatterns


class TestPIIPatterns:
    """Tests for PII pattern definitions."""

    def test_email_pattern(self):
        """Test email pattern detection."""
        import re
        pattern = re.compile(PIIPatterns.EMAIL.pattern, re.IGNORECASE)

        # Should match
        assert pattern.search("email@example.com")
        assert pattern.search("user.name@domain.co.uk")
        assert pattern.search("test+label@gmail.com")

        # Should not match
        assert not pattern.search("not an email")
        assert not pattern.search("@nodomain")

    def test_phone_pattern(self):
        """Test phone pattern detection."""
        import re
        pattern = re.compile(PIIPatterns.PHONE_US.pattern, re.IGNORECASE)

        # Should match
        assert pattern.search("555-123-4567")
        assert pattern.search("(555) 123-4567")
        assert pattern.search("5551234567")
        assert pattern.search("+1 555-123-4567")

    def test_ssn_pattern(self):
        """Test SSN pattern detection."""
        import re
        pattern = re.compile(PIIPatterns.SSN.pattern, re.IGNORECASE)

        # Should match
        assert pattern.search("123-45-6789")
        assert pattern.search("123 45 6789")

    def test_credit_card_pattern(self):
        """Test credit card pattern detection."""
        import re
        pattern = re.compile(PIIPatterns.CREDIT_CARD.pattern, re.IGNORECASE)

        # Should match
        assert pattern.search("1234-5678-9012-3456")
        assert pattern.search("1234 5678 9012 3456")

    def test_get_all_patterns(self):
        """Test getting all patterns."""
        patterns = PIIPatterns.get_all_patterns()
        assert len(patterns) > 0
        assert all(hasattr(p, 'pattern') for p in patterns)


class TestPIIRedactor:
    """Tests for PIIRedactor class."""

    @pytest.fixture
    def redactor(self):
        """Create a mock-mode redactor."""
        return PIIRedactor(mock_mode=True, use_presidio=False)

    def test_redact_email(self, redactor):
        """Test email redaction."""
        text = "My email is john.doe@example.com"
        result = redactor.redact(text)

        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text
        assert result.entity_count == 1

    def test_redact_phone(self, redactor):
        """Test phone number redaction."""
        text = "Call me at 555-123-4567"
        result = redactor.redact(text)

        assert "[PHONE_REDACTED]" in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

    def test_redact_ssn(self, redactor):
        """Test SSN redaction."""
        text = "My SSN is 123-45-6789"
        result = redactor.redact(text)

        assert "[SSN_REDACTED]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text

    def test_redact_multiple_entities(self, redactor):
        """Test multiple entity redaction."""
        text = "Email: test@example.com, Phone: 555-123-4567"
        result = redactor.redact(text)

        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "[PHONE_REDACTED]" in result.redacted_text
        assert result.entity_count == 2

    def test_redact_preserves_other_text(self, redactor):
        """Test that non-PII text is preserved."""
        text = "Hello, my email is test@example.com and I'm feeling anxious"
        result = redactor.redact(text)

        assert "Hello" in result.redacted_text
        assert "I'm feeling anxious" in result.redacted_text

    def test_empty_input(self, redactor):
        """Test empty input handling."""
        result = redactor.redact("")
        assert result.redacted_text == ""
        assert result.entity_count == 0

    def test_no_pii(self, redactor):
        """Test input with no PII."""
        text = "I'm feeling really stressed about work"
        result = redactor.redact(text)

        assert result.redacted_text == text
        assert result.entity_count == 0

    def test_restore(self, redactor):
        """Test restoration of original text."""
        original = "Contact me at john@example.com"
        result = redactor.redact(original)

        restored = redactor.restore(result.redacted_text, result.mapping)
        assert restored == original

    def test_result_to_dict(self, redactor):
        """Test RedactionResult.to_dict()."""
        text = "Email: test@example.com"
        result = redactor.redact(text)
        result_dict = result.to_dict()

        assert "redacted_text" in result_dict
        assert "entity_count" in result_dict
        assert "entities" in result_dict

    def test_multiple_same_type(self, redactor):
        """Test multiple entities of same type."""
        text = "Emails: a@b.com and c@d.com"
        result = redactor.redact(text)

        # Should have unique replacements
        assert result.entity_count == 2
        assert len(result.mapping) == 2


class TestMockPIIRedactor:
    """Tests for MockPIIRedactor class."""

    def test_mock_redactor_creation(self):
        """Test mock redactor initialization."""
        redactor = MockPIIRedactor()
        assert redactor.mock_mode

    def test_mock_redactor_redact(self):
        """Test mock redactor redaction."""
        redactor = MockPIIRedactor()
        result = redactor.redact("Email: test@example.com")

        assert isinstance(result, RedactionResult)
        assert "[EMAIL_REDACTED]" in result.redacted_text


class TestPIIRedactorIntegration:
    """Integration tests for PII redaction."""

    def test_realistic_therapy_message(self):
        """Test redaction of realistic therapy message."""
        redactor = PIIRedactor(mock_mode=True, use_presidio=False)

        message = """
        Hi, I'm John. I've been feeling really anxious lately.
        You can reach me at john.doe@email.com or call 555-123-4567.
        My therapist says I should track my feelings.
        """

        result = redactor.redact(message)

        # Should redact email and phone
        assert "john.doe@email.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text

        # Should preserve therapeutic content
        assert "anxious" in result.redacted_text
        assert "therapist" in result.redacted_text

    def test_supported_entities(self):
        """Test getting supported entity types."""
        redactor = PIIRedactor(mock_mode=True)
        entities = redactor.get_supported_entities()

        assert "EMAIL" in entities
        assert "PHONE" in entities
        assert "SSN" in entities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
