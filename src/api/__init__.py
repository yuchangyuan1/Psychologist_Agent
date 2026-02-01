"""
API module for cloud integration.

This module provides the Deepseek API client for cloud-based
clinical analysis in the inference pipeline.
"""

from src.api.deepseek_client import DeepseekClient, MockDeepseekClient
from src.api.models import (
    APIConfig, AnalysisResult, AnalysisParser,
    RiskLevel, TherapeuticApproach,
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse
)
from src.api.exceptions import (
    APIError, AuthenticationError, RateLimitError,
    TimeoutError, ServerError, ConnectionError
)

__all__ = [
    "DeepseekClient",
    "MockDeepseekClient",
    "APIConfig",
    "AnalysisResult",
    "AnalysisParser",
    "RiskLevel",
    "TherapeuticApproach",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ServerError",
    "ConnectionError"
]
