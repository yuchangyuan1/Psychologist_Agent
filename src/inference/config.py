"""
Configuration for local inference service.

This module provides configuration dataclasses for the GGUF
model inference pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class ModelConfig:
    """Configuration for the local GGUF model."""
    model_path: str = "models/psychologist-8b-q4_k_m.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = offload all layers to GPU
    n_batch: int = 512
    n_threads: Optional[int] = None  # None = auto-detect
    verbose: bool = False
    use_mmap: bool = True
    use_mlock: bool = False


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: List[str] = field(default_factory=lambda: ["<|user|>", "<|system|>", "\n\n\n"])
    stream: bool = False


@dataclass
class ServerConfig:
    """Configuration for inference server."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    timeout: int = 120
    max_concurrent_requests: int = 4
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class InferenceConfig:
    """Complete inference configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        config = cls()

        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if "generation" in data:
            for key, value in data["generation"].items():
                if hasattr(config.generation, key):
                    setattr(config.generation, key, value)

        if "server" in data:
            for key, value in data["server"].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)

        return config

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Model config from env
        if os.getenv("LOCAL_MODEL_PATH"):
            config.model.model_path = os.getenv("LOCAL_MODEL_PATH")

        if os.getenv("MODEL_N_CTX"):
            config.model.n_ctx = int(os.getenv("MODEL_N_CTX"))

        if os.getenv("MODEL_N_GPU_LAYERS"):
            config.model.n_gpu_layers = int(os.getenv("MODEL_N_GPU_LAYERS"))

        # Generation config from env
        if os.getenv("GENERATION_MAX_TOKENS"):
            config.generation.max_tokens = int(os.getenv("GENERATION_MAX_TOKENS"))

        if os.getenv("GENERATION_TEMPERATURE"):
            config.generation.temperature = float(os.getenv("GENERATION_TEMPERATURE"))

        # Server config from env
        if os.getenv("SERVER_HOST"):
            config.server.host = os.getenv("SERVER_HOST")

        if os.getenv("SERVER_PORT"):
            config.server.port = int(os.getenv("SERVER_PORT"))

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": {
                "model_path": self.model.model_path,
                "n_ctx": self.model.n_ctx,
                "n_gpu_layers": self.model.n_gpu_layers,
                "n_batch": self.model.n_batch,
                "n_threads": self.model.n_threads,
                "verbose": self.model.verbose
            },
            "generation": {
                "max_tokens": self.generation.max_tokens,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "repeat_penalty": self.generation.repeat_penalty,
                "stop": self.generation.stop
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "timeout": self.server.timeout
            }
        }
