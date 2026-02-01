"""
Utilities package for LegalDoc AI
"""

from .prompt_manager import PromptManager, detect_provider_from_llm
from .llm_provider_manager import LLMProviderManager

__all__ = ['PromptManager', 'detect_provider_from_llm', 'LLMProviderManager']
