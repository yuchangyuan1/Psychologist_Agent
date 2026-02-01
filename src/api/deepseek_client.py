"""
Deepseek API Client for cloud-based analysis.

This module provides the DeepseekClient class for interacting with
the Deepseek-V3 API for clinical analysis in the inference pipeline.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
import json

from src.api.models import (
    APIConfig, ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
    AnalysisResult, AnalysisParser, RiskLevel, TherapeuticApproach,
    extract_json_from_text
)
from src.api.exceptions import (
    APIError, AuthenticationError, RateLimitError,
    TimeoutError, ServerError, ConnectionError
)
from src.utils.logging_config import setup_logging

logger = setup_logging("deepseek_client")


class DeepseekClient:
    """
    Async client for Deepseek-V3 API.

    Provides methods for clinical analysis of user messages with
    support for retries, timeouts, and error handling.

    Example:
        client = DeepseekClient()
        result = await client.analyze(
            system_message="Analyze this message...",
            user_message="I'm feeling anxious"
        )
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        mock_mode: Optional[bool] = None
    ):
        """
        Initialize Deepseek client.

        Args:
            config: API configuration
            mock_mode: Whether to use mock mode
        """
        self.config = config or APIConfig.from_env()
        self.mock_mode = mock_mode
        if self.mock_mode is None:
            self.mock_mode = os.getenv("LLM_TYPE", "MOCK").upper() == "MOCK"

        self._http_client = None
        self._initialized = False

        logger.info(f"DeepseekClient initialized (mock_mode={self.mock_mode})")

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._http_client is not None:
            return

        if self.mock_mode:
            return

        try:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
            )
            self._initialized = True
            logger.info("HTTP client initialized")
        except ImportError:
            logger.warning("httpx not installed, falling back to mock mode")
            self.mock_mode = True

    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self._initialized = False

    async def analyze(
        self,
        system_message: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AnalysisResult:
        """
        Analyze user message using cloud API.

        Args:
            system_message: System prompt for analysis
            user_message: User message to analyze
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            AnalysisResult: Parsed analysis result
        """
        if self.mock_mode:
            return self._mock_analyze(user_message)

        await self._ensure_client()

        messages = [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=user_message)
        ]

        request = ChatCompletionRequest(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )

        response = await self._chat_completion(request)

        if response.choices:
            response_text = response.choices[0].message.content

            # Try JSON extraction first
            json_str = extract_json_from_text(response_text)
            if json_str:
                try:
                    json_data = json.loads(json_str)
                    return AnalysisParser.parse_json(json_data)
                except json.JSONDecodeError:
                    logger.warning("JSON extraction failed, falling back to text parsing")

            # Fall back to text parsing
            return AnalysisParser.parse(response_text)

        return AnalysisResult(risk_level=RiskLevel.LOW)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Send chat messages and get response.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            str: Response text
        """
        if self.mock_mode:
            return self._mock_chat(messages)

        await self._ensure_client()

        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        request = ChatCompletionRequest(
            model=self.config.model,
            messages=chat_messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )

        response = await self._chat_completion(request)

        if response.choices:
            return response.choices[0].message.content

        return ""

    async def _chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Send chat completion request with retries.

        Args:
            request: Chat completion request

        Returns:
            ChatCompletionResponse: API response
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._http_client.post(
                    "/chat/completions",
                    json=request.model_dump()
                )

                if response.status_code == 200:
                    data = response.json()
                    return ChatCompletionResponse(**data)

                # Handle error responses
                self._handle_error_response(response)

            except asyncio.TimeoutError:
                last_error = TimeoutError("Request timed out")
                logger.warning(f"Request timeout (attempt {attempt + 1})")

            except Exception as e:
                if isinstance(e, APIError):
                    # Don't retry auth errors
                    if isinstance(e, AuthenticationError):
                        raise
                    last_error = e
                else:
                    last_error = ConnectionError(str(e))
                    logger.warning(f"Connection error: {e} (attempt {attempt + 1})")

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error or APIError("Max retries exceeded")

    def _handle_error_response(self, response) -> None:
        """Handle error response from API."""
        status = response.status_code
        try:
            data = response.json()
            message = data.get("error", {}).get("message", "Unknown error")
        except Exception:
            message = response.text or "Unknown error"

        if status == 401:
            raise AuthenticationError(message)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(message, int(retry_after) if retry_after else None)
        elif status >= 500:
            raise ServerError(message, status)
        else:
            raise APIError(message, status)

    def _mock_analyze(self, user_message: str) -> AnalysisResult:
        """Generate mock analysis for testing."""
        # Simple keyword-based mock analysis
        message_lower = user_message.lower()

        if any(word in message_lower for word in ["suicide", "kill myself", "end my life", "want to die"]):
            risk_level = RiskLevel.CRITICAL
            concern = "suicidal ideation"
            approach = TherapeuticApproach.SUPPORTIVE
            key_points = [
                "Express immediate concern for safety",
                "Provide crisis resources (988)",
                "Encourage professional help"
            ]
            guidance = "Prioritize safety assessment. Provide crisis resources immediately."
            technique = "Crisis Intervention with Safety Planning"
            risk_reasoning = "User expressing direct suicidal ideation requires immediate crisis response."
        elif any(word in message_lower for word in ["hurt myself", "self-harm", "cutting"]):
            risk_level = RiskLevel.HIGH
            concern = "self-harm risk"
            approach = TherapeuticApproach.DBT
            key_points = [
                "Validate distress without reinforcing behavior",
                "Explore DBT distress tolerance skills",
                "Assess safety and support system"
            ]
            guidance = "Use DBT distress tolerance skills. Validate emotions without reinforcing self-harm."
            technique = "DBT TIPP Skills (Temperature, Intense Exercise, Paced Breathing, Progressive Relaxation)"
            risk_reasoning = "Self-harm mention indicates elevated risk requiring careful intervention."
        elif any(word in message_lower for word in ["anxious", "anxiety", "panic", "worried"]):
            risk_level = RiskLevel.LOW
            concern = "anxiety symptoms"
            approach = TherapeuticApproach.CBT
            key_points = [
                "Validate feelings of anxiety",
                "Explore specific triggers",
                "Suggest grounding techniques"
            ]
            guidance = "Explore anxiety triggers and introduce grounding techniques."
            technique = "CBT Cognitive Restructuring with Grounding Exercises"
            risk_reasoning = "Anxiety symptoms without crisis indicators. Standard therapeutic approach."
        elif any(word in message_lower for word in ["depressed", "hopeless", "sad", "empty"]):
            risk_level = RiskLevel.MODERATE
            concern = "depressive symptoms"
            approach = TherapeuticApproach.CBT
            key_points = [
                "Acknowledge and validate low mood",
                "Explore behavioral activation",
                "Check for suicidal ideation"
            ]
            guidance = "Validate low mood. Gently explore behavioral activation opportunities."
            technique = "CBT Behavioral Activation with Validation"
            risk_reasoning = "Depressive symptoms require monitoring for suicidal ideation."
        else:
            risk_level = RiskLevel.LOW
            concern = "general emotional support"
            approach = TherapeuticApproach.SUPPORTIVE
            key_points = [
                "Provide empathetic listening",
                "Explore current feelings",
                "Build rapport"
            ]
            guidance = "Focus on building rapport and exploring the user's current emotional state."
            technique = "Person-Centered Active Listening"
            risk_reasoning = "No crisis indicators detected. Standard supportive approach."

        return AnalysisResult(
            risk_level=risk_level,
            primary_concern=concern,
            suggested_approach=approach,
            key_points=key_points,
            raw_response=f"[MOCK] Analysis for: {user_message[:50]}...",
            confidence=0.85,
            risk_reasoning=risk_reasoning,
            guidance_for_local_model=guidance,
            suggested_technique=technique,
            updated_user_profile={}
        )

    def _mock_chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock chat response for testing."""
        return "[MOCK] This is a mock response from the Deepseek API."


class MockDeepseekClient(DeepseekClient):
    """Mock client for testing."""

    def __init__(self):
        super().__init__(mock_mode=True)

    async def analyze(
        self,
        system_message: str,
        user_message: str,
        **kwargs
    ) -> AnalysisResult:
        """Return mock analysis."""
        return self._mock_analyze(user_message)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Return mock chat response."""
        return self._mock_chat(messages)
