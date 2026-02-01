"""
Prompt template loading and management.

This module provides functions for loading and managing prompt templates
for cloud and local model inference.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml

from src.utils.logging_config import setup_logging

logger = setup_logging("prompt_templates")


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    name: str
    system_message: str
    user_template: str
    description: str = ""
    variables: list = field(default_factory=list)

    def format(self, **kwargs) -> Dict[str, str]:
        """Format the template with provided variables."""
        return {
            "system": self.system_message.format(**kwargs) if kwargs else self.system_message,
            "user": self.user_template.format(**kwargs) if kwargs else self.user_template
        }


# Default templates
DEFAULT_TEMPLATES = {
    "cloud_analysis": PromptTemplate(
        name="cloud_analysis",
        description="Template for Deepseek-V3 cloud analysis",
        system_message="""You are an expert clinical psychologist assistant. Your role is to analyze user messages and provide clinical insights.

Analyze the following user message and provide:
1. Risk Level Assessment (low/moderate/high/critical)
2. Primary Emotional Concern
3. Suggested Therapeutic Approach (CBT, DBT, supportive, etc.)
4. Key Points for Response

Be concise and clinical in your analysis. Focus on actionable insights.

{rag_context}""",
        user_template="""User Message: {user_input}

Conversation History:
{conversation_history}

Provide your clinical analysis in the following format:
RISK_LEVEL: [low/moderate/high/critical]
PRIMARY_CONCERN: [brief description]
SUGGESTED_APPROACH: [therapeutic approach]
KEY_POINTS:
- [point 1]
- [point 2]
- [point 3]""",
        variables=["user_input", "conversation_history", "rag_context"]
    ),

    "local_generation": PromptTemplate(
        name="local_generation",
        description="Template for local GGUF model response generation",
        system_message="""You are a compassionate and professional mental health support assistant.
Your role is to provide empathetic, supportive responses while maintaining appropriate boundaries.

Guidelines:
- Be warm, understanding, and non-judgmental
- Use active listening techniques
- Validate emotions before offering suggestions
- Encourage professional help when appropriate
- Never provide medical diagnoses or prescribe treatment
- Maintain hope while being realistic

{therapeutic_guidance}""",
        user_template="""Based on the following clinical analysis, generate a supportive response:

Clinical Analysis:
{cloud_analysis}

Relevant Context:
{rag_context}

User's Message: {user_input}

Recent Conversation:
{conversation_history}

Respond with empathy and support, incorporating the suggested therapeutic approach.""",
        variables=["user_input", "cloud_analysis", "rag_context", "conversation_history", "therapeutic_guidance"]
    ),

    "crisis_response": PromptTemplate(
        name="crisis_response",
        description="Template for crisis situations",
        system_message="""You are a crisis support specialist. Your primary goal is to ensure user safety.

CRITICAL: Always prioritize user safety. Provide crisis resources immediately.
- 988 Suicide & Crisis Lifeline
- Crisis Text Line: Text HOME to 741741
- Emergency: 911

Be calm, supportive, and direct about the need for professional help.""",
        user_template="""The user is experiencing a crisis. Provide immediate support and resources.

Risk Level: {risk_level}
Matched Concern: {matched_pattern}

User's Message: {user_input}

Respond with immediate crisis support, validation, and clear resource information.""",
        variables=["user_input", "risk_level", "matched_pattern"]
    ),

    "safety_followup": PromptTemplate(
        name="safety_followup",
        description="Template for safety check follow-ups",
        system_message="""You are a mental health support assistant conducting a gentle safety check.
Be caring and non-judgmental while assessing the user's wellbeing.""",
        user_template="""Previous conversation indicated potential risk. Conduct a gentle safety check.

Previous Context:
{conversation_history}

User's Current Message: {user_input}

Respond with care, checking in on their wellbeing without being intrusive.""",
        variables=["user_input", "conversation_history"]
    )
}


class TemplateLoader:
    """
    Loader for prompt templates from YAML files.

    Supports loading from file and falling back to defaults.
    """

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize template loader.

        Args:
            template_path: Path to YAML template file
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.template_path = template_path or os.path.join(
            base_dir, "prompts", "prompt_templates.yaml"
        )
        self._templates: Dict[str, PromptTemplate] = {}
        self._loaded = False

    def load(self, force_reload: bool = False) -> Dict[str, PromptTemplate]:
        """
        Load templates from YAML file.

        Args:
            force_reload: Force reload even if already loaded

        Returns:
            Dictionary of template name to PromptTemplate
        """
        if self._loaded and not force_reload:
            return self._templates

        # Start with defaults
        self._templates = dict(DEFAULT_TEMPLATES)

        # Try to load from file
        if os.path.exists(self.template_path):
            try:
                with open(self.template_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if data and "templates" in data:
                    for name, template_data in data["templates"].items():
                        self._templates[name] = PromptTemplate(
                            name=name,
                            system_message=template_data.get("system_message", ""),
                            user_template=template_data.get("user_template", ""),
                            description=template_data.get("description", ""),
                            variables=template_data.get("variables", [])
                        )
                    logger.info(f"Loaded {len(data['templates'])} templates from {self.template_path}")

            except Exception as e:
                logger.warning(f"Error loading templates from {self.template_path}: {e}")
        else:
            logger.info("No template file found, using defaults")

        self._loaded = True
        return self._templates

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        if not self._loaded:
            self.load()
        return self._templates.get(name)

    def get_all(self) -> Dict[str, PromptTemplate]:
        """Get all templates."""
        if not self._loaded:
            self.load()
        return self._templates
