"""
Tests for Deepseek API integration.

All tests use MOCK mode - no real API calls.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.api.deepseek_client import DeepseekClient, MockDeepseekClient
from src.api.models import (
    APIConfig, AnalysisResult, AnalysisParser,
    RiskLevel, TherapeuticApproach,
    ChatMessage, ChatCompletionRequest
)
from src.api.exceptions import APIError, AuthenticationError


class TestAPIConfig:
    """Tests for APIConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = APIConfig()
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.model == "deepseek-chat"
        assert config.timeout == 30.0

    def test_from_env(self):
        """Test loading config from environment."""
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        os.environ["DEEPSEEK_MODEL"] = "test-model"

        config = APIConfig.from_env()
        assert config.api_key == "test-key"
        assert config.model == "test-model"

        # Cleanup
        del os.environ["DEEPSEEK_API_KEY"]
        del os.environ["DEEPSEEK_MODEL"]


class TestChatModels:
    """Tests for chat models."""

    def test_chat_message(self):
        """Test ChatMessage model."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_completion_request(self):
        """Test ChatCompletionRequest model."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.5
        )
        assert request.temperature == 0.5
        assert len(request.messages) == 1


class TestAnalysisParser:
    """Tests for AnalysisParser class."""

    def test_parse_valid_response(self):
        """Test parsing a valid analysis response."""
        response = """
        RISK_LEVEL: moderate
        PRIMARY_CONCERN: anxiety symptoms
        SUGGESTED_APPROACH: CBT
        KEY_POINTS:
        - Validate feelings
        - Explore triggers
        - Suggest techniques
        """

        result = AnalysisParser.parse(response)
        assert result.risk_level == RiskLevel.MODERATE
        assert result.primary_concern == "anxiety symptoms"
        assert result.suggested_approach == TherapeuticApproach.CBT
        assert len(result.key_points) == 3

    def test_parse_with_different_formats(self):
        """Test parsing with various format variations."""
        response = """
        risk_level: HIGH
        Primary_Concern: depression
        suggested_approach: supportive
        KEY_POINTS:
        - Point one
        - Point two
        """

        result = AnalysisParser.parse(response)
        assert result.risk_level == RiskLevel.HIGH
        assert "depression" in result.primary_concern

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = AnalysisParser.parse("")
        assert result.risk_level == RiskLevel.LOW  # Default


class TestAnalysisResult:
    """Tests for AnalysisResult class."""

    def test_to_dict(self):
        """Test AnalysisResult.to_dict()."""
        result = AnalysisResult(
            risk_level=RiskLevel.HIGH,
            primary_concern="test concern",
            suggested_approach=TherapeuticApproach.DBT,
            key_points=["point1", "point2"]
        )

        result_dict = result.to_dict()
        assert result_dict["risk_level"] == "high"
        assert result_dict["suggested_approach"] == "DBT"

    def test_to_string(self):
        """Test AnalysisResult.to_string()."""
        result = AnalysisResult(
            risk_level=RiskLevel.MODERATE,
            primary_concern="anxiety",
            suggested_approach=TherapeuticApproach.CBT,
            key_points=["validate", "explore"]
        )

        result_str = result.to_string()
        assert "RISK_LEVEL: moderate" in result_str
        assert "PRIMARY_CONCERN: anxiety" in result_str


class TestDeepseekClient:
    """Tests for DeepseekClient class."""

    @pytest.fixture
    def client(self):
        """Create a mock-mode client."""
        return DeepseekClient(mock_mode=True)

    @pytest.mark.asyncio
    async def test_analyze_anxiety(self, client):
        """Test analyzing anxiety-related message."""
        result = await client.analyze(
            system_message="Analyze this message",
            user_message="I'm feeling very anxious about work"
        )

        assert isinstance(result, AnalysisResult)
        assert result.risk_level == RiskLevel.LOW
        assert "anxiety" in result.primary_concern

    @pytest.mark.asyncio
    async def test_analyze_high_risk(self, client):
        """Test analyzing high-risk message."""
        result = await client.analyze(
            system_message="Analyze this message",
            user_message="I want to hurt myself"
        )

        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_analyze_depression(self, client):
        """Test analyzing depression-related message."""
        result = await client.analyze(
            system_message="Analyze this message",
            user_message="I feel so hopeless and sad"
        )

        assert result.risk_level == RiskLevel.MODERATE
        assert "depress" in result.primary_concern.lower()

    @pytest.mark.asyncio
    async def test_chat(self, client):
        """Test chat method."""
        response = await client.chat([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ])

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_client_close(self, client):
        """Test client close method."""
        await client.close()
        assert client._http_client is None


class TestMockDeepseekClient:
    """Tests for MockDeepseekClient class."""

    @pytest.mark.asyncio
    async def test_mock_analyze(self):
        """Test mock analysis."""
        client = MockDeepseekClient()
        result = await client.analyze(
            system_message="Test",
            user_message="I'm anxious"
        )

        assert isinstance(result, AnalysisResult)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_mock_chat(self):
        """Test mock chat."""
        client = MockDeepseekClient()
        response = await client.chat([
            {"role": "user", "content": "Hello"}
        ])

        assert "[MOCK]" in response


class TestExceptions:
    """Tests for API exceptions."""

    def test_api_error(self):
        """Test APIError exception."""
        error = APIError("Test error", status_code=400)
        assert str(error) == "[400] Test error"

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError()
        assert error.status_code == 401

    def test_error_without_status(self):
        """Test error without status code."""
        error = APIError("Generic error")
        assert str(error) == "Generic error"


class TestIntegration:
    """Integration tests for API module."""

    @pytest.mark.asyncio
    async def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        client = DeepseekClient(mock_mode=True)

        # Analyze message
        result = await client.analyze(
            system_message="You are a clinical psychologist...",
            user_message="I've been feeling really stressed lately"
        )

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.risk_level in list(RiskLevel)
        assert len(result.key_points) > 0

        # Convert to string for local model
        result_str = result.to_string()
        assert "RISK_LEVEL" in result_str
        assert "KEY_POINTS" in result_str

        await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
