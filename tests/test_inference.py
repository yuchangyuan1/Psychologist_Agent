"""
Tests for Local Inference module.

All tests use MOCK mode - no real model loading.
"""

import os
import pytest

# Ensure MOCK mode for all tests
os.environ["LLM_TYPE"] = "MOCK"

from src.inference.config import (
    InferenceConfig, ModelConfig, GenerationConfig, ServerConfig
)
from src.inference.generator import (
    LocalGenerator, GenerationResult, MockLocalGenerator
)


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.n_ctx == 4096
        assert config.n_gpu_layers == -1
        assert config.verbose is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_path="custom/path.gguf",
            n_ctx=2048,
            n_gpu_layers=20
        )
        assert config.model_path == "custom/path.gguf"
        assert config.n_ctx == 2048


class TestGenerationConfig:
    """Tests for GenerationConfig class."""

    def test_default_values(self):
        """Test default generation config."""
        config = GenerationConfig()
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_stop_sequences(self):
        """Test stop sequences."""
        config = GenerationConfig()
        assert len(config.stop) > 0
        assert "<|user|>" in config.stop


class TestInferenceConfig:
    """Tests for InferenceConfig class."""

    def test_default_config(self):
        """Test default inference configuration."""
        config = InferenceConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.server, ServerConfig)

    def test_from_env(self):
        """Test loading from environment variables."""
        os.environ["LOCAL_MODEL_PATH"] = "test/model.gguf"
        os.environ["GENERATION_MAX_TOKENS"] = "256"

        config = InferenceConfig.from_env()
        assert config.model.model_path == "test/model.gguf"
        assert config.generation.max_tokens == 256

        # Cleanup
        del os.environ["LOCAL_MODEL_PATH"]
        del os.environ["GENERATION_MAX_TOKENS"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = InferenceConfig()
        result = config.to_dict()

        assert "model" in result
        assert "generation" in result
        assert "server" in result


class TestLocalGenerator:
    """Tests for LocalGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a mock-mode generator."""
        return LocalGenerator(mock_mode=True)

    @pytest.mark.asyncio
    async def test_initialize(self, generator):
        """Test generator initialization."""
        await generator.initialize()
        assert generator._initialized

    @pytest.mark.asyncio
    async def test_generate(self, generator):
        """Test text generation."""
        await generator.initialize()
        result = await generator.generate("Hello, how are you?")

        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.tokens_generated > 0
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_with_config(self, generator):
        """Test generation with custom config."""
        await generator.initialize()

        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5
        )

        result = await generator.generate("Test prompt", config)
        assert isinstance(result, GenerationResult)

    @pytest.mark.asyncio
    async def test_generate_stream(self, generator):
        """Test streaming generation."""
        await generator.initialize()

        tokens = []
        async for token in generator.generate_stream("Hello"):
            tokens.append(token)

        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_chat(self, generator):
        """Test chat completion."""
        await generator.initialize()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        result = await generator.chat(messages)
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0

    def test_format_chat_prompt(self, generator):
        """Test chat prompt formatting."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "Follow up"}
        ]

        prompt = generator._format_chat_prompt(messages)

        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "System message" in prompt

    @pytest.mark.asyncio
    async def test_get_model_info(self, generator):
        """Test getting model info."""
        await generator.initialize()
        info = generator.get_model_info()

        assert "initialized" in info
        assert "mock_mode" in info
        assert info["initialized"] is True
        assert info["mock_mode"] is True

    @pytest.mark.asyncio
    async def test_unload(self, generator):
        """Test model unloading."""
        await generator.initialize()
        assert generator._initialized

        await generator.unload()
        assert not generator._initialized

    @pytest.mark.asyncio
    async def test_result_to_dict(self, generator):
        """Test GenerationResult.to_dict()."""
        await generator.initialize()
        result = await generator.generate("Test")

        result_dict = result.to_dict()
        assert "text" in result_dict
        assert "tokens_generated" in result_dict
        assert "finish_reason" in result_dict


class TestMockLocalGenerator:
    """Tests for MockLocalGenerator class."""

    @pytest.mark.asyncio
    async def test_mock_generator(self):
        """Test mock generator."""
        generator = MockLocalGenerator()
        await generator.initialize()

        result = await generator.generate("Test prompt")

        assert "[MOCK]" in result.text
        assert result.metadata.get("mock") is True


class TestServerConfig:
    """Tests for ServerConfig class."""

    def test_default_values(self):
        """Test default server config."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.enable_cors is True

    def test_cors_origins(self):
        """Test CORS origins configuration."""
        config = ServerConfig(
            cors_origins=["http://localhost:3000", "http://example.com"]
        )
        assert len(config.cors_origins) == 2


class TestIntegration:
    """Integration tests for inference module."""

    @pytest.mark.asyncio
    async def test_full_generation_flow(self):
        """Test complete generation flow."""
        # Create generator
        generator = LocalGenerator(mock_mode=True)

        # Initialize
        await generator.initialize()
        assert generator._initialized

        # Generate response
        prompt = """<|system|>
You are a compassionate mental health assistant.
<|user|>
I'm feeling anxious today.
<|assistant|>
"""

        result = await generator.generate(prompt)

        assert len(result.text) > 0
        assert result.generation_time_ms > 0

        # Cleanup
        await generator.unload()

    @pytest.mark.asyncio
    async def test_chat_flow(self):
        """Test complete chat flow."""
        generator = LocalGenerator(mock_mode=True)
        await generator.initialize()

        # First turn
        messages = [
            {"role": "system", "content": "You are a mental health assistant."},
            {"role": "user", "content": "I'm stressed about work."}
        ]

        result = await generator.chat(messages)
        assert len(result.text) > 0

        # Second turn (with history)
        messages.append({"role": "assistant", "content": result.text})
        messages.append({"role": "user", "content": "What can I do about it?"})

        result2 = await generator.chat(messages)
        assert len(result2.text) > 0

        await generator.unload()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
