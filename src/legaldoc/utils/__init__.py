"""
Utilities package for LegalDoc AI

This package provides:
- LLMProviderManager: Unified LLM interface with OpenAI/Gemini fallback
- PDFReportGenerator: PDF report generation for analysis results
- Schemas: Pydantic models for structured LLM outputs
- Configuration utilities
"""

from .llm_provider_manager import (
    LLMProviderManager,
    LLMProviderError,
    RateLimitError,
    APIError,
    AllProvidersFailedError,
)
from .load_env import get_config
from .pdf_generator import PDFReportGenerator
from .schemas import (
    Clause,
    SplitterResponse,
    ClassificationResult,
    IdentifiedRisk,
    RiskAssessmentResult,
)

__all__ = [
    # LLM Management
    "LLMProviderManager",
    "LLMProviderError",
    "RateLimitError",
    "APIError",
    "AllProvidersFailedError",
    # PDF Generation
    "PDFReportGenerator",
    # Configuration
    "get_config",
    # Schemas
    "Clause",
    "SplitterResponse",
    "ClassificationResult",
    "IdentifiedRisk",
    "RiskAssessmentResult",
]
