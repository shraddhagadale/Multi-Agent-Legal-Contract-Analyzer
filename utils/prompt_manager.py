"""
Prompt Manager for LegalDoc AI

This module handles loading and managing prompts for different LLM providers.
It automatically selects the appropriate prompt templates based on the active provider.

Prompt files are stored in:
- prompts/openai/   - OpenAI-specific prompts
- prompts/gemini/   - Gemini-specific prompts
"""

import os
from typing import Optional
from pathlib import Path


class PromptManager:
    """
    Manages prompt templates for different LLM providers.
    
    Automatically loads prompts from the appropriate provider directory
    based on the active LLM provider.
    """
    
    # Supported providers and their prompt directories
    PROVIDER_DIRS = {
        "openai": "openai",
        "gemini": "gemini",
    }
    
    # Default provider if detection fails
    DEFAULT_PROVIDER = "openai"
    
    def __init__(self, llm_manager):
        """
        Initialize the PromptManager.
        
        Args:
            llm_manager: An LLMProviderManager instance
        """
        # Determine the provider from the LLM manager
        self.provider = detect_provider_from_llm(llm_manager)
        
        # Validate provider
        if self.provider not in self.PROVIDER_DIRS:
            print(f"[PromptManager] ⚠️ Unknown provider '{self.provider}', defaulting to '{self.DEFAULT_PROVIDER}'")
            self.provider = self.DEFAULT_PROVIDER
        
        # Set up paths
        self.base_path = Path(__file__).parent.parent / "prompts"
        self.prompt_dir = self.base_path / self.PROVIDER_DIRS[self.provider]
        
        # Cache for loaded prompts
        self._prompt_cache = {}
        
        print(f"[PromptManager] 📁 Using prompts from: {self.prompt_dir}")
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template by name.
        
        Args:
            prompt_name: Name of the prompt (without .txt extension)
        
        Returns:
            str: The prompt template content
        
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        # Check cache first
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        
        # Build file path
        prompt_file = self.prompt_dir / f"{prompt_name}.txt"
        
        # Try to load the prompt
        if not prompt_file.exists():
            # Try fallback to default provider
            fallback_file = self.base_path / self.PROVIDER_DIRS[self.DEFAULT_PROVIDER] / f"{prompt_name}.txt"
            if fallback_file.exists():
                print(f"[PromptManager] ⚠️ Prompt '{prompt_name}' not found for {self.provider}, using {self.DEFAULT_PROVIDER} version")
                prompt_file = fallback_file
            else:
                raise FileNotFoundError(
                    f"Prompt '{prompt_name}' not found.\n"
                    f"Looked in: {prompt_file}\n"
                    f"Fallback: {fallback_file}"
                )
        
        # Read and cache the prompt
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        
        self._prompt_cache[prompt_name] = prompt_content
        return prompt_content
    
    def get_provider(self) -> str:
        """Return the current provider name."""
        return self.provider
    
    def list_available_prompts(self) -> list:
        """List all available prompt files for the current provider."""
        if not self.prompt_dir.exists():
            return []
        
        return [
            f.stem for f in self.prompt_dir.glob("*.txt")
        ]
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._prompt_cache.clear()


def detect_provider_from_llm(llm_manager) -> str:
    """
    Detect the provider name from an LLM manager or model instance.
    
    Args:
        llm_manager: An LLM manager or model instance
    
    Returns:
        str: Provider name ('openai', 'gemini', etc.)
    """
    # If it has get_provider_name method (our LLMProviderManager)
    if hasattr(llm_manager, 'get_provider_name'):
        return llm_manager.get_provider_name()
    
    # If it has active_provider attribute
    if hasattr(llm_manager, 'active_provider'):
        return llm_manager.active_provider or "openai"
    
    # Try to detect from class name
    class_name = type(llm_manager).__name__.lower()
    
    if "openai" in class_name or "gpt" in class_name:
        return "openai"
    elif "gemini" in class_name or "google" in class_name:
        return "gemini"
    
    # Default fallback
    print(f"[PromptManager] ⚠️ Could not detect provider from {type(llm_manager)}, defaulting to 'openai'")
    return "openai"
