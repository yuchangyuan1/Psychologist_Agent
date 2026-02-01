"""
Pydantic models for API request/response data.

This module defines data models for the Deepseek API integration,
including request parameters and response parsing.
"""

import re
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract first JSON object from text using regex.

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string or None if not found
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group()
    return None


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TherapeuticApproach(str, Enum):
    """Therapeutic approach enumeration."""
    CBT = "CBT"
    DBT = "DBT"
    SUPPORTIVE = "supportive"
    PSYCHODYNAMIC = "psychodynamic"
    INTEGRATIVE = "integrative"


class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion."""
    model: str = Field(default="deepseek-chat", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = Field(default=None)


class ChatCompletionChoice(BaseModel):
    """A completion choice in the response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: Optional[Usage] = None


@dataclass
class AnalysisResult:
    """Parsed result from cloud analysis."""
    risk_level: RiskLevel = RiskLevel.LOW
    primary_concern: str = ""
    suggested_approach: TherapeuticApproach = TherapeuticApproach.SUPPORTIVE
    key_points: List[str] = field(default_factory=list)
    raw_response: str = ""
    confidence: float = 1.0
    # New fields from review.md
    risk_reasoning: str = ""
    guidance_for_local_model: str = ""
    suggested_technique: str = ""
    updated_user_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "primary_concern": self.primary_concern,
            "suggested_approach": self.suggested_approach.value,
            "key_points": self.key_points,
            "confidence": self.confidence,
            "risk_reasoning": self.risk_reasoning,
            "guidance_for_local_model": self.guidance_for_local_model,
            "suggested_technique": self.suggested_technique,
            "updated_user_profile": self.updated_user_profile
        }

    def to_string(self) -> str:
        """Convert to formatted string for prompt."""
        key_points_str = "\n".join(f"- {p}" for p in self.key_points)
        return f"""RISK_LEVEL: {self.risk_level.value}
PRIMARY_CONCERN: {self.primary_concern}
SUGGESTED_APPROACH: {self.suggested_approach.value}
KEY_POINTS:
{key_points_str}"""


class AnalysisParser:
    """Parser for cloud analysis responses."""

    @staticmethod
    def parse_json(json_data: Dict[str, Any]) -> AnalysisResult:
        """
        Parse analysis from JSON structure (new format).

        Args:
            json_data: Parsed JSON dictionary from cloud API

        Returns:
            AnalysisResult: Parsed analysis
        """
        risk_str = json_data.get("risk_level", "low").lower()
        # Map "medium" to "moderate" for compatibility
        if risk_str == "medium":
            risk_str = "moderate"

        try:
            risk_level = RiskLevel(risk_str)
        except ValueError:
            if "high" in risk_str or "critical" in risk_str:
                risk_level = RiskLevel.HIGH
            elif "moderate" in risk_str or "medium" in risk_str:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.LOW

        # Map suggested_technique to therapeutic approach
        technique = json_data.get("suggested_technique", "").upper()
        if "CBT" in technique:
            approach = TherapeuticApproach.CBT
        elif "DBT" in technique:
            approach = TherapeuticApproach.DBT
        else:
            approach = TherapeuticApproach.SUPPORTIVE

        # Build key_points from guidance and technique
        key_points = []
        if json_data.get("guidance_for_local_model"):
            key_points.append(json_data["guidance_for_local_model"])
        if json_data.get("suggested_technique"):
            key_points.append(json_data["suggested_technique"])

        return AnalysisResult(
            risk_level=risk_level,
            primary_concern=json_data.get("primary_concern", ""),
            suggested_approach=approach,
            key_points=key_points,
            raw_response=json.dumps(json_data),
            confidence=1.0,
            risk_reasoning=json_data.get("risk_reasoning", ""),
            guidance_for_local_model=json_data.get("guidance_for_local_model", ""),
            suggested_technique=json_data.get("suggested_technique", ""),
            updated_user_profile=json_data.get("updated_user_profile", {})
        )

    @staticmethod
    def parse(response_text: str) -> AnalysisResult:
        """
        Parse analysis response into structured result.

        Args:
            response_text: Raw response from cloud API

        Returns:
            AnalysisResult: Parsed analysis
        """
        result = AnalysisResult(raw_response=response_text)

        lines = response_text.strip().split("\n")
        current_section = None
        key_points = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse risk level
            if line.upper().startswith("RISK_LEVEL:"):
                level_str = line.split(":", 1)[1].strip().lower()
                try:
                    result.risk_level = RiskLevel(level_str)
                except ValueError:
                    # Default to low if parsing fails
                    if "high" in level_str or "critical" in level_str:
                        result.risk_level = RiskLevel.HIGH
                    elif "moderate" in level_str:
                        result.risk_level = RiskLevel.MODERATE
                    else:
                        result.risk_level = RiskLevel.LOW

            # Parse primary concern
            elif line.upper().startswith("PRIMARY_CONCERN:"):
                result.primary_concern = line.split(":", 1)[1].strip()

            # Parse suggested approach
            elif line.upper().startswith("SUGGESTED_APPROACH:"):
                approach_str = line.split(":", 1)[1].strip().upper()
                try:
                    result.suggested_approach = TherapeuticApproach(approach_str)
                except ValueError:
                    if "CBT" in approach_str:
                        result.suggested_approach = TherapeuticApproach.CBT
                    elif "DBT" in approach_str:
                        result.suggested_approach = TherapeuticApproach.DBT
                    else:
                        result.suggested_approach = TherapeuticApproach.SUPPORTIVE

            # Parse key points
            elif line.upper().startswith("KEY_POINTS:"):
                current_section = "key_points"

            elif current_section == "key_points" and line.startswith("-"):
                point = line.lstrip("- ").strip()
                if point:
                    key_points.append(point)

        result.key_points = key_points
        return result


@dataclass
class APIConfig:
    """Configuration for Deepseek API."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 1000

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create config from environment variables."""
        import os
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "30")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
        )
