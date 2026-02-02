"""
Utilities package for LegalDoc AI

This package provides:
- LLMProviderManager: Unified LLM interface with OpenAI/Gemini fallback
- PromptManager: Provider-specific prompt loading
- Schemas: Pydantic models for structured LLM outputs
- Configuration utilities
"""

from .prompt_manager import PromptManager, detect_provider_from_llm
from .llm_provider_manager import LLMProviderManager, LLMProviderError, RateLimitError, APIError
from .load_env import get_config
from .schemas import (
    Clause,
    SplitterResponse,
    ClassificationResult,
    IdentifiedRisk,
    RiskAssessmentResult,
)

__all__ = [
    'PromptManager',
    'detect_provider_from_llm',
    'LLMProviderManager',
    'LLMProviderError',
    'RateLimitError',
    'APIError',
    'get_config',
    # Schemas
    'Clause',
    'SplitterResponse',
    'ClassificationResult',
    'IdentifiedRisk',
    'RiskAssessmentResult',
]
