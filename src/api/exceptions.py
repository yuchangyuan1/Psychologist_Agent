"""
Custom exceptions for API module.

This module defines exceptions for error handling in the
Deepseek API client and related components.
"""

from typing import Optional, Dict, Any


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401)


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class TimeoutError(APIError):
    """Raised when request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, status_code=408)


class InvalidRequestError(APIError):
    """Raised when request is invalid."""

    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, status_code=400)


class ServerError(APIError):
    """Raised when server returns 5xx error."""

    def __init__(
        self,
        message: str = "Server error",
        status_code: int = 500
    ):
        super().__init__(message, status_code=status_code)


class ModelNotAvailableError(APIError):
    """Raised when requested model is not available."""

    def __init__(self, model_name: str):
        super().__init__(f"Model not available: {model_name}", status_code=404)
        self.model_name = model_name


class ContentFilterError(APIError):
    """Raised when content is filtered by safety systems."""

    def __init__(self, message: str = "Content filtered"):
        super().__init__(message, status_code=400)


class ConnectionError(APIError):
    """Raised when connection to API fails."""

    def __init__(self, message: str = "Connection failed"):
        super().__init__(message)
