"""
Utilities package for LegalDoc AI

This package provides:
- LLMClient: OpenAI client with structured outputs and retries
- Schemas: Pydantic models for structured LLM outputs
- Configuration utilities
"""

from .llm_client import (
    LLMClient,
    LLMError,
    RateLimitError,
    APIError,
)

from .schemas import (
    Clause,
    SplitterResponse,
    ClassificationResult,
    IdentifiedRisk,
    RiskAssessmentResult,
)

__all__ = [
    # LLM Client
    "LLMClient",
    "LLMError",
    "RateLimitError",
    "APIError",
    # Schemas
    "Clause",
    "SplitterResponse",
    "ClassificationResult",
    "IdentifiedRisk",
    "RiskAssessmentResult",
]
