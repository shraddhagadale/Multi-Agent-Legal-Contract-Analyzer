"""
LLM Client for LegalDoc AI

A clean, focused client for OpenAI API interactions with:
- Structured outputs via Instructor + Pydantic
- Automatic retries with exponential backoff
- Centralized error handling

Usage:
    from legaldoc.utils import LLMClient
    
    client = LLMClient()
    response = client.structured_chat(messages, ResponseModel)
"""

import os
import logging
from typing import List, Dict, Type, TypeVar

import openai
import instructor
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Module logger
logger = logging.getLogger(__name__)

# Generic type for Pydantic response models
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded."""
    pass


class APIError(LLMError):
    """Raised for general API errors."""
    pass


class LLMClient:
    """
    OpenAI LLM client with structured outputs and automatic retries.
    
    Features:
    - Structured outputs via Instructor + Pydantic for type-safe responses
    - Exponential backoff retry on rate limits
    - Centralized configuration and error handling
    """
    
    def __init__(self):
        """Initialize the LLM client with OpenAI."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        
        if not self.api_key:
            raise LLMError(
                "OpenAI API key not found.\n"
                "Please set OPENAI_API_KEY in your .env file."
            )
        
        # Initialize OpenAI client with Instructor for structured outputs
        self._client = instructor.from_openai(
            openai.OpenAI(api_key=self.api_key)
        )
        
        logger.info(f"LLM Client initialized with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def structured_chat(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> T:
        """
        Send a chat request with structured output.
        
        Uses Pydantic models to guarantee the response matches
        the expected schema. Instructor handles validation retries.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class defining expected response
            temperature: Sampling temperature (0-1), default 0 for consistency
            max_retries: Retries for validation failures (handled by Instructor)
        
        Returns:
            Validated Pydantic model instance
        
        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: For other API errors
        """
        try:
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_retries=max_retries,
            )
        except openai.RateLimitError as e:
            raise RateLimitError(f"Rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            raise APIError(f"Authentication failed - check your API key: {e}")
        except openai.APIError as e:
            raise APIError(f"OpenAI API error: {e}")
        except Exception as e:
            raise APIError(f"Unexpected error: {e}")
    
    def get_model_name(self) -> str:
        """Return the model name being used."""
        return self.model


# Backward compatibility alias
LLMProviderManager = LLMClient
