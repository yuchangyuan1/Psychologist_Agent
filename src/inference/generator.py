"""
Local Generator for GGUF model inference.

This module provides the LocalGenerator class for generating responses
using local GGUF models via llama-cpp-python.
"""

import os
import asyncio
from typing import Optional, AsyncIterator, Dict, Any, List
from dataclasses import dataclass, field

from src.inference.config import InferenceConfig, ModelConfig, GenerationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging("local_generator")


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    tokens_generated: int
    finish_reason: str
    generation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "tokens_generated": self.tokens_generated,
            "finish_reason": self.finish_reason,
            "generation_time_ms": self.generation_time_ms,
            "metadata": self.metadata
        }


class LocalGenerator:
    """
    Local generator for GGUF model inference.

    Provides methods for generating responses using local GGUF models
    with support for streaming and async generation.

    Example:
        generator = LocalGenerator()
        await generator.initialize()
        result = await generator.generate(prompt, config)
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize local generator.

        Args:
            config: Inference configuration
            mock_mode: Whether to use mock mode
        """
        self.config = config or InferenceConfig.from_env()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        self._model = None
        self._initialized = False

        logger.info(f"LocalGenerator created (mock_mode={self.mock_mode})")

    async def initialize(self) -> None:
        """Initialize the model (load into memory)."""
        if self._initialized:
            return

        if self.mock_mode:
            self._initialized = True
            logger.info("Mock mode - skipping model loading")
            return

        try:
            from llama_cpp import Llama

            model_path = self.config.model.model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            logger.info(f"Loading model from {model_path}...")

            self._model = Llama(
                model_path=model_path,
                n_ctx=self.config.model.n_ctx,
                n_gpu_layers=self.config.model.n_gpu_layers,
                n_batch=self.config.model.n_batch,
                n_threads=self.config.model.n_threads,
                use_mmap=self.config.model.use_mmap,
                use_mlock=self.config.model.use_mlock,
                chat_format="llama-3",  # Enable Llama-3 chat template
                verbose=self.config.model.verbose
            )

            self._initialized = True
            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning("llama-cpp-python not installed, using mock mode")
            self.mock_mode = True
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            config: Generation configuration (overrides defaults)

        Returns:
            GenerationResult: Generation result
        """
        if not self._initialized:
            await self.initialize()

        gen_config = config or self.config.generation

        if self.mock_mode:
            return self._mock_generate(prompt, gen_config)

        import time
        start_time = time.time()

        try:
            # Run generation in thread pool to not block async loop
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._model(
                    prompt,
                    max_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    repeat_penalty=gen_config.repeat_penalty,
                    presence_penalty=gen_config.presence_penalty,
                    frequency_penalty=gen_config.frequency_penalty,
                    stop=gen_config.stop,
                    echo=False
                )
            )

            generation_time = (time.time() - start_time) * 1000

            text = output["choices"][0]["text"]
            finish_reason = output["choices"][0].get("finish_reason", "unknown")
            tokens = output.get("usage", {}).get("completion_tokens", 0)

            return GenerationResult(
                text=text.strip(),
                tokens_generated=tokens,
                finish_reason=finish_reason,
                generation_time_ms=generation_time
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate response using chat completion API with structured messages.
        This is the PREFERRED method for Llama-3 models.

        Args:
            messages: List of message dictionaries with role and content
            config: Generation configuration (overrides defaults)

        Returns:
            GenerationResult: Generation result
        """
        if not self._initialized:
            await self.initialize()

        gen_config = config or self.config.generation

        if self.mock_mode:
            return self._mock_generate("", gen_config)

        import time
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    repeat_penalty=gen_config.repeat_penalty,
                    stop=gen_config.stop
                )
            )

            generation_time = (time.time() - start_time) * 1000
            text = output["choices"][0]["message"]["content"]
            finish_reason = output["choices"][0].get("finish_reason", "unknown")
            tokens = output.get("usage", {}).get("completion_tokens", 0)

            return GenerationResult(
                text=text.strip(),
                tokens_generated=tokens,
                finish_reason=finish_reason,
                generation_time_ms=generation_time
            )

        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming output.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            str: Generated text tokens
        """
        if not self._initialized:
            await self.initialize()

        gen_config = config or self.config.generation

        if self.mock_mode:
            # Mock streaming
            mock_response = "[MOCK] This is a streaming mock response from the local model."
            for word in mock_response.split():
                yield word + " "
                await asyncio.sleep(0.05)
            return

        try:
            # Create generator
            stream = self._model(
                prompt,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repeat_penalty=gen_config.repeat_penalty,
                stop=gen_config.stop,
                stream=True,
                echo=False
            )

            for output in stream:
                token = output["choices"][0].get("text", "")
                if token:
                    yield token

                # Allow other async tasks to run
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate response for chat messages.

        Args:
            messages: List of message dictionaries with role and content
            config: Generation configuration

        Returns:
            GenerationResult: Generation result
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)
        return await self.generate(prompt, config)

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into model prompt."""
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")

        # Add assistant prompt for response
        parts.append("<|assistant|>\n")

        return "\n".join(parts)

    def _mock_generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> GenerationResult:
        """Generate mock response for testing."""
        mock_responses = [
            "I understand that you're going through a difficult time. It's completely normal to feel this way, and I appreciate you sharing with me.",
            "Thank you for opening up about your feelings. What you're experiencing is valid, and I'm here to support you.",
            "I hear you. It sounds like you're dealing with a lot right now. Let's explore this together.",
            "Your feelings are completely understandable. Many people experience similar emotions in challenging situations."
        ]

        import random
        response = random.choice(mock_responses)

        return GenerationResult(
            text=f"[MOCK] {response}",
            tokens_generated=len(response.split()),
            finish_reason="stop",
            generation_time_ms=50.0,
            metadata={"mock": True}
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._initialized:
            return {"initialized": False, "mock_mode": self.mock_mode}

        if self.mock_mode:
            return {
                "initialized": True,
                "mock_mode": True,
                "model_path": "mock"
            }

        return {
            "initialized": True,
            "mock_mode": False,
            "model_path": self.config.model.model_path,
            "n_ctx": self.config.model.n_ctx,
            "n_gpu_layers": self.config.model.n_gpu_layers
        }

    async def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        self._initialized = False
        logger.info("Model unloaded")


class MockLocalGenerator(LocalGenerator):
    """Mock generator for testing."""

    def __init__(self):
        super().__init__(mock_mode=True)

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Return mock generation."""
        return self._mock_generate(prompt, config or GenerationConfig())
