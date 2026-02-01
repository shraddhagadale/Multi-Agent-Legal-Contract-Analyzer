"""
Utilities package for LegalDoc AI

This package provides:
- LLMProviderManager: Unified LLM interface with OpenAI/Gemini fallback
- PromptManager: Provider-specific prompt loading
- Configuration utilities
"""

from .prompt_manager import PromptManager, detect_provider_from_llm
from .llm_provider_manager import LLMProviderManager, LLMProviderError, RateLimitError, APIError
from .load_env import get_config

__all__ = [
    'PromptManager',
    'detect_provider_from_llm',
    'LLMProviderManager',
    'LLMProviderError',
    'RateLimitError',
    'APIError',
    'get_config'
]
