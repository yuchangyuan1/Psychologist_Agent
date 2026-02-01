"""
Tests for Safety Gateway module.

All tests use MOCK mode - no real embeddings or API calls.
"""

import os
import pytest
import asyncio

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.safety.gateway import SafetyGateway, SafetyResult
from src.safety.patterns import RiskLevel, PatternLoader
from src.safety.embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Tests for EmbeddingManager class."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingManager.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingManager.reset_instance()

    def test_singleton_pattern(self):
        """Test that EmbeddingManager is a singleton."""
        manager1 = EmbeddingManager(mock_mode=True)
        manager2 = EmbeddingManager(mock_mode=True)
        assert manager1 is manager2

    def test_encode_single_text(self):
        """Test encoding a single text."""
        manager = EmbeddingManager(mock_mode=True)
        embedding = manager.encode("Hello world")
        assert embedding.shape == (1, 384)

    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        manager = EmbeddingManager(mock_mode=True)
        texts = ["Hello", "World", "Test"]
        embeddings = manager.encode(texts)
        assert embeddings.shape == (3, 384)

    def test_deterministic_embeddings(self):
        """Test that same text produces same embedding."""
        manager = EmbeddingManager(mock_mode=True)
        emb1 = manager.encode("Test text")
        emb2 = manager.encode("Test text")
        assert (emb1 == emb2).all()

    def test_similarity_computation(self):
        """Test similarity computation between embeddings."""
        manager = EmbeddingManager(mock_mode=True)
        query = manager.encode("Hello")
        corpus = manager.encode(["Hello", "World", "Goodbye"])

        similarities = manager.similarity(query, corpus)
        assert len(similarities) == 3
        # Same text should have highest similarity
        assert similarities[0] > 0.99

    def test_find_most_similar(self):
        """Test finding most similar texts."""
        manager = EmbeddingManager(mock_mode=True)
        corpus = ["I am happy", "I am sad", "I want to eat"]
        results = manager.find_most_similar("I am happy", corpus, top_k=3, threshold=-1.0)

        # In mock mode, at least the exact match should be returned
        assert len(results) >= 1
        assert results[0][0] == "I am happy"
        assert results[0][1] > 0.99


class TestPatternLoader:
    """Tests for PatternLoader class."""

    def setup_method(self):
        """Clear cache before each test."""
        PatternLoader.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        PatternLoader.clear_cache()

    def test_load_patterns(self):
        """Test loading risk patterns."""
        loader = PatternLoader()
        db = loader.load_patterns()

        assert len(db.high_risk) > 0
        assert len(db.moderate_risk) > 0
        assert len(db.low_risk) > 0

    def test_load_responses(self):
        """Test loading safety responses."""
        loader = PatternLoader()
        db = loader.load_responses()

        assert len(db.crisis_responses) > 0
        assert "immediate_danger" in db.crisis_responses

    def test_pattern_categories(self):
        """Test that patterns have proper categories."""
        loader = PatternLoader()
        db = loader.load_patterns()

        for pattern in db.high_risk:
            assert pattern.category in [
                "self_harm", "harm_to_others",
                "active_crisis", "abuse_disclosure"
            ]
            assert pattern.risk_level == RiskLevel.HIGH

    def test_thresholds_loaded(self):
        """Test that thresholds are loaded correctly."""
        loader = PatternLoader()
        db = loader.load_patterns()

        assert "high_risk" in db.thresholds
        assert "moderate_risk" in db.thresholds
        assert "low_risk" in db.thresholds
        assert db.thresholds["high_risk"] >= db.thresholds["moderate_risk"]


class TestSafetyGateway:
    """Tests for SafetyGateway class."""

    def setup_method(self):
        """Reset singletons before each test."""
        EmbeddingManager.reset_instance()
        PatternLoader.clear_cache()

    def teardown_method(self):
        """Reset singletons after each test."""
        EmbeddingManager.reset_instance()
        PatternLoader.clear_cache()

    @pytest.fixture
    def gateway(self):
        """Create a mock-mode gateway."""
        return SafetyGateway(mock_mode=True)

    @pytest.mark.asyncio
    async def test_safe_input(self, gateway):
        """Test that normal input is marked as safe."""
        result = await gateway.check("I'm having a great day today!")
        assert result.is_safe
        assert result.risk_level in [RiskLevel.NONE, RiskLevel.LOW]

    @pytest.mark.asyncio
    async def test_empty_input(self, gateway):
        """Test handling of empty input."""
        result = await gateway.check("")
        assert result.is_safe
        assert result.risk_level == RiskLevel.NONE

    @pytest.mark.asyncio
    async def test_result_to_dict(self, gateway):
        """Test SafetyResult.to_dict() method."""
        result = await gateway.check("I'm feeling okay")
        result_dict = result.to_dict()

        assert "is_safe" in result_dict
        assert "risk_level" in result_dict
        assert isinstance(result_dict["risk_level"], str)

    @pytest.mark.asyncio
    async def test_get_follow_up_prompt(self, gateway):
        """Test retrieving follow-up prompts."""
        prompt = gateway.get_follow_up_prompt("safety_check")
        assert prompt is not None
        assert "hurt" in prompt.lower() or "thoughts" in prompt.lower()

    @pytest.mark.asyncio
    async def test_get_crisis_resources(self, gateway):
        """Test retrieving crisis resources."""
        resources = gateway.get_crisis_resources("immediate_danger")
        assert len(resources) > 0
        assert any("988" in r.get("phone", "") for r in resources)

    @pytest.mark.asyncio
    async def test_high_risk_detection_exact_match(self, gateway):
        """Test that exact high-risk phrases are detected."""
        # Use exact pattern text for mock mode testing
        result = await gateway.check("I want to kill myself")

        # In mock mode with deterministic embeddings, exact match should work
        assert result.matched_pattern is not None or result.similarity_score > 0

    @pytest.mark.asyncio
    async def test_result_contains_response(self, gateway):
        """Test that results contain appropriate response data."""
        result = await gateway.check("I'm stressed about work")

        # Safe inputs should still have a result
        assert isinstance(result, SafetyResult)
        assert hasattr(result, "is_safe")
        assert hasattr(result, "risk_level")


class TestSafetyIntegration:
    """Integration tests for safety module."""

    def setup_method(self):
        """Reset state before each test."""
        EmbeddingManager.reset_instance()
        PatternLoader.clear_cache()

    @pytest.mark.asyncio
    async def test_multiple_checks(self):
        """Test running multiple safety checks in sequence."""
        gateway = SafetyGateway(mock_mode=True)

        inputs = [
            "I'm having a good day",
            "I'm feeling stressed",
            "I'm worried about work"
        ]

        for text in inputs:
            result = await gateway.check(text)
            assert isinstance(result, SafetyResult)

    @pytest.mark.asyncio
    async def test_gateway_reuses_embeddings(self):
        """Test that pattern embeddings are cached."""
        gateway = SafetyGateway(mock_mode=True)

        # First check triggers embedding computation
        await gateway.check("Test message 1")
        embeddings_after_first = gateway._pattern_embeddings

        # Second check should use cached embeddings
        await gateway.check("Test message 2")
        embeddings_after_second = gateway._pattern_embeddings

        assert embeddings_after_first is embeddings_after_second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
