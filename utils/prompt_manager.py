"""
Prompt Manager for LegalDoc AI

This module handles loading and managing prompts for LLM agents.
Prompts are stored in the prompts/ directory as .txt files.

With structured outputs (Pydantic schemas), prompts no longer need
to specify output format - the schema handles that automatically.
"""

import os
from pathlib import Path


class PromptManager:
    """
    Manages prompt templates for LLM agents.
    
    Loads prompts from the prompts/ directory. Since we use Pydantic
    schemas for structured outputs, prompts focus on task instructions
    rather than output format specifications.
    """
    
    def __init__(self, llm_manager=None):
        """
        Initialize the PromptManager.
        
        Args:
            llm_manager: Optional LLMProviderManager instance (kept for
                        backwards compatibility, but no longer used for
                        prompt selection since prompts are now unified)
        """
        # Set up paths
        self.base_path = Path(__file__).parent.parent / "prompts"
        
        # Cache for loaded prompts
        self._prompt_cache = {}
        
        # Store provider name for logging (optional)
        self._provider = None
        if llm_manager and hasattr(llm_manager, 'get_provider_name'):
            self._provider = llm_manager.get_provider_name()
        
        print(f"[PromptManager] 📁 Using prompts from: {self.base_path}")
    
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
        prompt_file = self.base_path / f"{prompt_name}.txt"
        
        # Try to load the prompt
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found.\n"
                f"Looked in: {prompt_file}"
            )
        
        # Read and cache the prompt
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        
        self._prompt_cache[prompt_name] = prompt_content
        return prompt_content
    
    def get_provider(self) -> str:
        """Return the current provider name (for logging purposes)."""
        return self._provider or "unknown"
    
    def list_available_prompts(self) -> list:
        """List all available prompt files."""
        if not self.base_path.exists():
            return []
        
        return [
            f.stem for f in self.base_path.glob("*.txt")
        ]
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._prompt_cache.clear()
