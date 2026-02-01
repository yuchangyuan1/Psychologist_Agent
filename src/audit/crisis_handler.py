"""
Crisis Handler for generating crisis intervention responses.

This module provides the CrisisHandler class for generating appropriate
crisis responses based on risk assessments.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from src.audit.risk_checker import RiskAssessment, InterventionLevel
from src.api.models import RiskLevel
from src.utils.logging_config import setup_logging

logger = setup_logging("crisis_handler")


@dataclass
class CrisisResource:
    """A crisis support resource."""
    name: str
    phone: Optional[str] = None
    text: Optional[str] = None
    website: Optional[str] = None
    description: str = ""


@dataclass
class CrisisResponse:
    """A crisis response to send to user."""
    message: str
    follow_up: str
    resources: List[CrisisResource]
    priority: int
    requires_escalation: bool
    response_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "follow_up": self.follow_up,
            "resources": [
                {
                    "name": r.name,
                    "phone": r.phone,
                    "text": r.text,
                    "website": r.website,
                    "description": r.description
                }
                for r in self.resources
            ],
            "priority": self.priority,
            "requires_escalation": self.requires_escalation,
            "response_type": self.response_type
        }


class CrisisHandler:
    """
    Handler for generating crisis intervention responses.

    Loads crisis response templates and resources from JSON files
    and generates appropriate responses based on risk assessments.

    Example:
        handler = CrisisHandler()
        if assessment.requires_crisis_response:
            response = handler.get_response(assessment)
            return response.message
    """

    def __init__(
        self,
        resources_path: Optional[str] = None,
        responses_path: Optional[str] = None
    ):
        """
        Initialize crisis handler.

        Args:
            resources_path: Path to crisis resources JSON
            responses_path: Path to crisis responses JSON
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.resources_path = resources_path or os.path.join(
            base_dir, "data", "crisis", "resources.json"
        )
        self.responses_path = responses_path or os.path.join(
            base_dir, "data", "crisis", "responses.json"
        )

        self._resources: Dict[str, Any] = {}
        self._templates: Dict[str, Any] = {}
        self._loaded = False

        self._load_data()
        logger.info("CrisisHandler initialized")

    def _load_data(self) -> None:
        """Load crisis resources and response templates."""
        # Load resources
        try:
            with open(self.resources_path, 'r', encoding='utf-8') as f:
                self._resources = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Resources file not found: {self.resources_path}")
            self._resources = self._get_default_resources()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing resources: {e}")
            self._resources = self._get_default_resources()

        # Load templates
        try:
            with open(self.responses_path, 'r', encoding='utf-8') as f:
                self._templates = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Templates file not found: {self.responses_path}")
            self._templates = self._get_default_templates()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing templates: {e}")
            self._templates = self._get_default_templates()

        self._loaded = True

    def get_response(
        self,
        assessment: RiskAssessment,
        crisis_type: Optional[str] = None
    ) -> CrisisResponse:
        """
        Get crisis response based on assessment.

        Args:
            assessment: Risk assessment result
            crisis_type: Specific crisis type (if known)

        Returns:
            CrisisResponse: Appropriate crisis response
        """
        # Determine crisis type from assessment if not provided
        if not crisis_type:
            crisis_type = self._determine_crisis_type(assessment)

        # Get template
        templates = self._templates.get("crisis_templates", {})
        template = templates.get(crisis_type, templates.get("severe_distress", {}))

        # Get resources
        resources = self._get_relevant_resources(crisis_type)

        return CrisisResponse(
            message=template.get("message", self._get_default_message()),
            follow_up=template.get("follow_up", "I'm here with you."),
            resources=resources,
            priority=template.get("priority", 3),
            requires_escalation=template.get("requires_escalation", False),
            response_type=crisis_type
        )

    def _determine_crisis_type(self, assessment: RiskAssessment) -> str:
        """Determine crisis type from assessment."""
        flags = assessment.flags
        concerns = " ".join(assessment.concerns).lower()

        if flags.get("urgent", False) or assessment.intervention_level == InterventionLevel.EMERGENCY:
            return "immediate_danger"

        if "suicid" in concerns or "kill myself" in concerns:
            return "suicidal_ideation"

        if "self-harm" in concerns or "cutting" in concerns or "hurt myself" in concerns:
            return "self_harm"

        if "harm" in concerns and ("other" in concerns or "someone" in concerns):
            return "harm_to_others"

        if "abuse" in concerns:
            return "abuse_disclosure"

        if "substance" in concerns or "overdose" in concerns:
            return "substance_crisis"

        if "panic" in concerns or "anxiety" in concerns:
            return "panic_attack"

        return "severe_distress"

    def _get_relevant_resources(self, crisis_type: str) -> List[CrisisResource]:
        """Get relevant resources for crisis type."""
        resources = []

        hotlines = self._resources.get("crisis_hotlines", {}).get("us", [])

        # Always include 988
        for hotline in hotlines:
            if "988" in hotline.get("name", ""):
                resources.append(CrisisResource(
                    name=hotline["name"],
                    phone=hotline.get("phone"),
                    text=hotline.get("text"),
                    description=hotline.get("description", "")
                ))
                break

        # Add crisis text line
        for hotline in hotlines:
            if "Crisis Text Line" in hotline.get("name", ""):
                resources.append(CrisisResource(
                    name=hotline["name"],
                    text=hotline.get("text"),
                    description=hotline.get("description", "")
                ))
                break

        # Add type-specific resources
        if crisis_type == "abuse_disclosure":
            for hotline in hotlines:
                if any(kw in hotline.get("name", "") for kw in ["Domestic Violence", "RAINN", "Childhelp"]):
                    resources.append(CrisisResource(
                        name=hotline["name"],
                        phone=hotline.get("phone"),
                        text=hotline.get("text"),
                        description=hotline.get("description", "")
                    ))

        elif crisis_type == "substance_crisis":
            for hotline in hotlines:
                if "SAMHSA" in hotline.get("name", ""):
                    resources.append(CrisisResource(
                        name=hotline["name"],
                        phone=hotline.get("phone"),
                        description=hotline.get("description", "")
                    ))
                    break

        return resources

    def get_validation_message(self) -> str:
        """Get a random validation phrase."""
        phrases = self._templates.get("validation_phrases", [
            "I hear you, and what you're feeling is valid."
        ])
        import random
        return random.choice(phrases)

    def get_safety_check_questions(self) -> List[str]:
        """Get safety check questions."""
        return self._templates.get("safety_check_questions", [
            "Are you in a safe place right now?"
        ])

    def format_resources_text(
        self,
        resources: List[CrisisResource],
        include_descriptions: bool = True
    ) -> str:
        """Format resources as text for response."""
        lines = []
        for r in resources:
            line = f"• **{r.name}**"
            if r.phone:
                line += f": {r.phone}"
            if r.text:
                line += f" (Text: {r.text})"
            if include_descriptions and r.description:
                line += f"\n  {r.description}"
            lines.append(line)
        return "\n".join(lines)

    def _get_default_message(self) -> str:
        """Get default crisis message."""
        return (
            "I'm concerned about what you're sharing. Please reach out to:\n\n"
            "**988 Suicide & Crisis Lifeline**: Call or text 988\n"
            "**Crisis Text Line**: Text HOME to 741741\n\n"
            "You don't have to face this alone."
        )

    def _get_default_resources(self) -> Dict[str, Any]:
        """Get default resources if file not found."""
        return {
            "crisis_hotlines": {
                "us": [
                    {
                        "name": "988 Suicide & Crisis Lifeline",
                        "phone": "988",
                        "text": "988",
                        "description": "24/7 crisis support"
                    }
                ]
            }
        }

    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default templates if file not found."""
        return {
            "crisis_templates": {
                "immediate_danger": {
                    "message": self._get_default_message(),
                    "follow_up": "I'm here with you.",
                    "priority": 1,
                    "requires_escalation": True
                }
            },
            "validation_phrases": [
                "I hear you, and what you're feeling is valid."
            ]
        }
