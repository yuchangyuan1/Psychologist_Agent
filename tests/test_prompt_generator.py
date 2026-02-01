"""
Tests for Prompt Generator module.

All tests use MOCK mode.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.prompt.generator import (
    PromptGenerator, PromptConfig,
    CloudPrompt, LocalPrompt
)
from src.prompt.templates import TemplateLoader, PromptTemplate, DEFAULT_TEMPLATES


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_template_format(self):
        """Test template formatting."""
        template = PromptTemplate(
            name="test",
            system_message="You are {role}",
            user_template="User says: {message}",
            variables=["role", "message"]
        )

        formatted = template.format(role="assistant", message="hello")
        assert formatted["system"] == "You are assistant"
        assert formatted["user"] == "User says: hello"

    def test_template_format_no_vars(self):
        """Test template formatting without variables."""
        template = PromptTemplate(
            name="test",
            system_message="Static system",
            user_template="Static user"
        )

        formatted = template.format()
        assert formatted["system"] == "Static system"


class TestTemplateLoader:
    """Tests for TemplateLoader class."""

    def test_load_default_templates(self):
        """Test loading default templates."""
        loader = TemplateLoader()
        templates = loader.load()

        assert "cloud_analysis" in templates
        assert "local_generation" in templates
        assert "crisis_response" in templates

    def test_get_template(self):
        """Test getting a specific template."""
        loader = TemplateLoader()
        template = loader.get("cloud_analysis")

        assert template is not None
        assert template.name == "cloud_analysis"

    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist."""
        loader = TemplateLoader()
        template = loader.get("nonexistent_template")

        assert template is None

    def test_get_all_templates(self):
        """Test getting all templates."""
        loader = TemplateLoader()
        templates = loader.get_all()

        assert len(templates) >= 4  # At least the defaults


class TestPromptGenerator:
    """Tests for PromptGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a prompt generator."""
        return PromptGenerator()

    def test_gen_cloud_prompt(self, generator):
        """Test cloud prompt generation."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="I'm feeling anxious",
            rag_context="CBT techniques for anxiety",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        )

        assert isinstance(prompt, CloudPrompt)
        assert len(prompt.system_message) > 0
        assert "I'm feeling anxious" in prompt.user_message
        # RAG context is now in user_message for supervisor prompt
        assert "CBT" in prompt.user_message

    def test_gen_cloud_prompt_no_context(self, generator):
        """Test cloud prompt without context."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Hello"
        )

        assert isinstance(prompt, CloudPrompt)
        assert "Hello" in prompt.user_message

    def test_gen_local_prompt(self, generator):
        """Test local prompt generation."""
        # New format: cloud_analysis is a dict
        prompt = generator.gen_local_prompt(
            user_input="I'm feeling anxious",
            cloud_analysis={
                "risk_level": "low",
                "primary_concern": "anxiety",
                "guidance_for_local_model": "Validate feelings",
                "suggested_technique": "DBT grounding"
            },
            rag_context="DBT skills for anxiety",
            history=[]
        )

        assert isinstance(prompt, LocalPrompt)
        assert len(prompt.full_prompt) > 0
        assert "I'm feeling anxious" in prompt.user_message
        # Analysis info is now in system_message as Supervisor Guidance
        assert "anxiety" in prompt.system_message
        # Check messages list is populated
        assert len(prompt.messages) == 2
        assert prompt.messages[0]["role"] == "system"

    def test_gen_crisis_prompt(self, generator):
        """Test crisis prompt generation."""
        prompt = generator.gen_crisis_prompt(
            user_input="I want to hurt myself",
            risk_level="high",
            matched_pattern="self_harm"
        )

        assert isinstance(prompt, LocalPrompt)
        assert "988" in prompt.system_message  # Crisis line
        assert "high" in prompt.user_message

    def test_cloud_prompt_to_messages(self, generator):
        """Test CloudPrompt.to_messages()."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test message"
        )

        messages = prompt.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_prompt_metadata(self, generator):
        """Test prompt metadata."""
        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test",
            rag_context="Context",
            history=[{"role": "user", "content": "Hi"}]
        )

        assert "template" in prompt.metadata
        assert "input_length" in prompt.metadata
        assert prompt.metadata["has_rag_context"] is True

    def test_history_formatting(self, generator):
        """Test conversation history formatting."""
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"}
        ]

        prompt = generator.gen_cloud_prompt(
            sanitized_input="Message 3",
            history=history
        )

        assert "Message 1" in prompt.user_message
        assert "Response 1" in prompt.user_message

    def test_history_truncation(self, generator):
        """Test that long history is truncated for local prompt."""
        # Cloud prompt uses 10 turns, local uses 3 turns
        # Test local prompt truncation
        config = PromptConfig(max_history_turns=2)
        generator = PromptGenerator(config=config)

        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]

        prompt = generator.gen_local_prompt(
            user_input="Test",
            cloud_analysis={"primary_concern": "test"},
            history=history
        )

        # Should only include last few messages (3 turns = 6 messages for local)
        assert "Message 0" not in prompt.user_message

    def test_rag_context_truncation(self, generator):
        """Test that long RAG context is truncated."""
        config = PromptConfig(max_rag_context_length=100)
        generator = PromptGenerator(config=config)

        long_context = "A" * 500

        prompt = generator.gen_cloud_prompt(
            sanitized_input="Test",
            rag_context=long_context
        )

        # RAG context is now in user_message for supervisor prompt
        # Should be truncated with ellipsis
        assert "..." in prompt.user_message

    def test_available_templates(self, generator):
        """Test getting available templates."""
        templates = generator.get_available_templates()

        assert "cloud_analysis" in templates
        assert "local_generation" in templates
        assert len(templates) >= 4


class TestPromptConfig:
    """Tests for PromptConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()

        assert config.max_history_turns == 5
        assert config.max_rag_context_length == 1500
        assert config.therapeutic_approach == "integrative"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PromptConfig(
            max_history_turns=10,
            therapeutic_approach="cbt"
        )

        assert config.max_history_turns == 10
        assert config.therapeutic_approach == "cbt"


class TestPromptIntegration:
    """Integration tests for prompt generation."""

    def test_full_pipeline_prompts(self):
        """Test generating prompts for full pipeline."""
        generator = PromptGenerator()

        # Step 1: Cloud analysis prompt
        cloud_prompt = generator.gen_cloud_prompt(
            sanitized_input="I've been feeling really anxious about work",
            rag_context="CBT techniques: cognitive restructuring, exposure therapy",
            history=[]
        )

        # Simulate cloud analysis result (new dict format)
        cloud_analysis = {
            "risk_level": "low",
            "primary_concern": "work-related anxiety",
            "suggested_technique": "CBT cognitive restructuring",
            "guidance_for_local_model": "Validate feelings of work stress and explore triggers",
            "risk_reasoning": "No crisis indicators",
            "updated_user_profile": {}
        }

        # Step 2: Local generation prompt
        local_prompt = generator.gen_local_prompt(
            user_input="I've been feeling really anxious about work",
            cloud_analysis=cloud_analysis,
            rag_context="CBT techniques: cognitive restructuring",
            history=[]
        )

        assert "anxious" in cloud_prompt.user_message
        # Analysis info is in system_message as Supervisor Guidance
        assert "work-related anxiety" in local_prompt.system_message
        assert "CBT" in local_prompt.system_message
        # Verify messages list is correctly built
        assert len(local_prompt.to_messages()) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
