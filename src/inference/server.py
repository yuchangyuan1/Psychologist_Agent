"""
Inference Server for local model serving.

This module provides a FastAPI-based server for serving the local
GGUF model via HTTP endpoints.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from src.inference.config import InferenceConfig, ServerConfig, GenerationConfig
from src.inference.generator import LocalGenerator, GenerationResult
from src.utils.logging_config import setup_logging

logger = setup_logging("inference_server")


# Pydantic models for API
try:
    from pydantic import BaseModel, Field

    class GenerateRequest(BaseModel):
        """Request for text generation."""
        prompt: str = Field(..., description="Input prompt")
        max_tokens: int = Field(default=512, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=40, ge=0)
        stream: bool = Field(default=False)

    class ChatMessage(BaseModel):
        """A chat message."""
        role: str = Field(..., description="Message role")
        content: str = Field(..., description="Message content")

    class ChatRequest(BaseModel):
        """Request for chat completion."""
        messages: List[ChatMessage]
        max_tokens: int = Field(default=512)
        temperature: float = Field(default=0.7)
        stream: bool = Field(default=False)

    class GenerateResponse(BaseModel):
        """Response from generation."""
        text: str
        tokens_generated: int
        finish_reason: str
        generation_time_ms: float

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        model_loaded: bool
        mock_mode: bool

except ImportError:
    # Fallback for when pydantic is not available
    GenerateRequest = dict
    ChatRequest = dict
    GenerateResponse = dict
    HealthResponse = dict


class InferenceServer:
    """
    FastAPI-based inference server.

    Serves the local GGUF model via HTTP endpoints for
    text generation and chat completion.

    Example:
        server = InferenceServer()
        await server.start()
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize inference server.

        Args:
            config: Inference configuration
            mock_mode: Whether to use mock mode
        """
        self.config = config or InferenceConfig.from_env()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        self.generator: Optional[LocalGenerator] = None
        self.app = None
        self._server = None

        logger.info(f"InferenceServer created (mock_mode={self.mock_mode})")

    def create_app(self):
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse
        except ImportError:
            logger.error("fastapi not installed")
            raise ImportError("fastapi is required for the inference server")

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.generator = LocalGenerator(self.config, self.mock_mode)
            await self.generator.initialize()
            logger.info("Inference server started")
            yield
            # Shutdown
            if self.generator:
                await self.generator.unload()
            logger.info("Inference server stopped")

        app = FastAPI(
            title="Psychologist Agent Inference Server",
            description="Local GGUF model inference API",
            version="1.0.0",
            lifespan=lifespan
        )

        # Add CORS middleware
        if self.config.server.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.server.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.generator is not None and self.generator._initialized,
                "mock_mode": self.mock_mode
            }

        @app.get("/model/info")
        async def model_info():
            """Get model information."""
            if self.generator is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return self.generator.get_model_info()

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generate text from prompt."""
            if self.generator is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            gen_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )

            if request.stream:
                async def stream_generator():
                    async for token in self.generator.generate_stream(
                        request.prompt, gen_config
                    ):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )

            result = await self.generator.generate(request.prompt, gen_config)
            return result.to_dict()

        @app.post("/chat")
        async def chat(request: ChatRequest):
            """Chat completion endpoint."""
            if self.generator is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]

            gen_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            if request.stream:
                # Format messages to prompt
                prompt = self.generator._format_chat_prompt(messages)

                async def stream_generator():
                    async for token in self.generator.generate_stream(
                        prompt, gen_config
                    ):
                        yield f"data: {token}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )

            result = await self.generator.chat(messages, gen_config)
            return {
                "message": {
                    "role": "assistant",
                    "content": result.text
                },
                "tokens_generated": result.tokens_generated,
                "generation_time_ms": result.generation_time_ms
            }

        self.app = app
        return app

    async def start(self) -> None:
        """Start the inference server."""
        if self.app is None:
            self.create_app()

        try:
            import uvicorn
            config = uvicorn.Config(
                self.app,
                host=self.config.server.host,
                port=self.config.server.port,
                workers=self.config.server.workers,
                timeout_keep_alive=self.config.server.timeout
            )
            self._server = uvicorn.Server(config)
            await self._server.serve()

        except ImportError:
            logger.error("uvicorn not installed")
            raise ImportError("uvicorn is required to run the server")

    async def stop(self) -> None:
        """Stop the inference server."""
        if self._server:
            self._server.should_exit = True


def create_app(mock_mode: bool = False) -> Any:
    """Create FastAPI app for deployment."""
    config = InferenceConfig.from_env()
    server = InferenceServer(config, mock_mode)
    return server.create_app()


# For running with uvicorn directly
app = None
try:
    mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"
    app = create_app(mock_mode)
except ImportError:
    pass
