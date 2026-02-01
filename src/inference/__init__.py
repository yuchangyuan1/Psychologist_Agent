"""
Inference module for local GGUF model serving.

This module provides the local inference service using
llama-cpp-python for privacy-preserving response generation.
"""

from src.inference.config import (
    InferenceConfig, ModelConfig, GenerationConfig, ServerConfig
)
from src.inference.generator import (
    LocalGenerator, GenerationResult, MockLocalGenerator
)
from src.inference.server import InferenceServer, create_app

__all__ = [
    "InferenceConfig",
    "ModelConfig",
    "GenerationConfig",
    "ServerConfig",
    "LocalGenerator",
    "GenerationResult",
    "MockLocalGenerator",
    "InferenceServer",
    "create_app"
]
