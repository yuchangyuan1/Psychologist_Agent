"""
Prompt module for cloud and local model prompts.

This module provides prompt generation and template management
for the two-stage inference pipeline.
"""

from src.prompt.generator import (
    PromptGenerator, PromptConfig,
    CloudPrompt, LocalPrompt
)
from src.prompt.templates import TemplateLoader, PromptTemplate

__all__ = [
    "PromptGenerator",
    "PromptConfig",
    "CloudPrompt",
    "LocalPrompt",
    "TemplateLoader",
    "PromptTemplate"
]
