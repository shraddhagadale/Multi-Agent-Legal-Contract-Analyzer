"""
Base Agent for LegalDoc AI

This module provides an abstract base class for all LegalDoc agents,
implementing common functionality and enforcing a consistent interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Type, TypeVar, ClassVar

from pydantic import BaseModel

from legaldoc.utils.llm_provider_manager import LLMProviderManager


# Generic type for Pydantic response models
T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all LegalDoc agents.
    
    Provides common functionality for:
    - LLM initialization
    - Prompt template loading (with caching)
    - Message building for LLM calls
    - Structured LLM interactions
    
    Subclasses must implement:
    - role: The agent's role description
    - goal: The agent's primary goal
    - prompt_name: Name of the prompt template file (without .txt)
    - expertise: Detailed expertise description for system prompt
    """
    
    # Class-level prompt cache shared across all agent instances
    _prompt_cache: ClassVar[Dict[str, str]] = {}
    
    # Path to prompts directory (resolved once)
    _prompts_dir: ClassVar[Path] = Path(__file__).parent.parent / "prompts"
    
    def __init__(self, llm_manager: LLMProviderManager) -> None:
        """
        Initialize the base agent.
        
        Args:
            llm_manager: LLMProviderManager instance for making LLM calls
        """
        self.llm = llm_manager
    
    @property
    @abstractmethod
    def role(self) -> str:
        """The agent's role (e.g., 'Legal Document Analyst')."""
        ...
    
    @property
    @abstractmethod
    def goal(self) -> str:
        """The agent's primary goal."""
        ...
    
    @property
    @abstractmethod
    def prompt_name(self) -> str:
        """Name of the prompt template file (without .txt extension)."""
        ...
    
    @property
    @abstractmethod
    def expertise(self) -> str:
        """Detailed expertise description for the system prompt."""
        ...
    
    def _load_prompt_template(self) -> str:
        """
        Load the prompt template for this agent.
        
        Uses class-level caching to avoid repeated file reads.
        
        Returns:
            str: The prompt template content
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        if self.prompt_name not in self._prompt_cache:
            prompt_path = self._prompts_dir / f"{self.prompt_name}.txt"
            
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"Prompt '{self.prompt_name}' not found at {prompt_path}"
                )
            
            self._prompt_cache[self.prompt_name] = prompt_path.read_text(encoding="utf-8")
        
        return self._prompt_cache[self.prompt_name]
    
    def _get_system_prompt(self) -> str:
        """
        Build the system prompt for this agent.
        
        Combines role, goal, and expertise into a cohesive system message.
        
        Returns:
            str: The complete system prompt
        """
        return f"You are a {self.role}. Your goal is to {self.goal}. {self.expertise}"
    
    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """
        Build the messages list for an LLM call.
        
        Args:
            user_prompt: The formatted user prompt
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    
    def _call_llm_structured(
        self,
        user_prompt: str,
        response_model: Type[T]
    ) -> T:
        """
        Make a structured LLM call with the agent's system prompt.
        
        Args:
            user_prompt: The formatted user prompt
            response_model: Pydantic model for response validation
        
        Returns:
            Validated Pydantic model instance
        """
        messages = self._build_messages(user_prompt)
        return self.llm.structured_chat(
            messages=messages,
            response_model=response_model
        )
