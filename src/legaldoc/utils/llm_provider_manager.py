"""
LLM Provider Manager with Runtime Fallback, Rate Limiting, and Structured Outputs

This module manages LLM provider initialization with automatic runtime fallback.
Primary: OpenAI (gpt-4o-mini)
Fallback: Google Gemini (gemini-2.0-flash)

Features:
- Automatic fallback on API errors (rate limits, timeouts, etc.)
- Built-in rate limiting with exponential backoff
- Unified interface for both providers
- Structured outputs via Instructor + Pydantic for type-safe responses
"""

import logging
from typing import Optional, List, Dict, Type, TypeVar

import openai
import instructor
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .load_env import get_config


# Module logger
logger = logging.getLogger(__name__)

# Generic type for Pydantic response models
T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    pass


class APIError(LLMProviderError):
    """Raised for general API errors."""
    pass


class AllProvidersFailedError(LLMProviderError):
    """Raised when all providers fail."""
    pass


class LLMProviderManager:
    """
    Unified LLM interface with automatic runtime fallback and rate limiting.
    
    Priority:
    1. OpenAI (gpt-4o-mini) - Primary
    2. Google Gemini (gemini-2.0-flash) - Fallback
    
    Features:
    - Runtime fallback: Automatically switches to Gemini if OpenAI fails
    - Rate limiting: Exponential backoff with configurable retries
    - Lazy validation: Providers validated on first use, not at init
    """
    
    def __init__(self):
        """Initialize the LLM Provider Manager with available clients."""
        self.config = get_config()
        
        # API Keys
        self.openai_key = self.config.get("OPENAI_API_KEY")
        self.google_key = self.config.get("GEMINI_API_KEY") or self.config.get("GOOGLE_API_KEY")
        
        # Model names
        self.openai_model = self.config.get("OPENAI_MODEL_NAME", "gpt-4o")
        self.google_model = self.config.get("GOOGLE_MODEL_NAME", "gemini-2.0-flash")
        
        # Initialize clients based on available keys (no validation/ping)
        self._openai_client = None
        self._gemini_client = None
        
        # Track available providers based on API key presence
        self._providers_available = {"openai": False, "gemini": False}
        
        # Determine active provider (primary)
        self.active_provider = None
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on API key presence."""
        # Initialize OpenAI if key exists
        if self.openai_key:
            self._openai_client = openai.OpenAI(api_key=self.openai_key)
            self._providers_available["openai"] = True
            self.active_provider = "openai"
            logger.debug("OpenAI provider initialized")
        
        # Initialize Gemini if key exists
        if self.google_key:
            self._gemini_client = genai.Client(api_key=self.google_key)
            self._providers_available["gemini"] = True
            if not self.active_provider:
                self.active_provider = "gemini"
            logger.debug("Gemini provider initialized")
        
        # Check if at least one provider is available
        if not any(self._providers_available.values()):
            raise AllProvidersFailedError(
                "CRITICAL: No LLM providers available.\n"
                "Please check your API keys in the .env file:\n"
                "- OPENAI_API_KEY\n"
                "- GEMINI_API_KEY (or GOOGLE_API_KEY)"
            )
        
        logger.info(f"LLM Provider initialized with: {self.active_provider}")
    
    def _log_error(self, provider: str, error_msg: str):
        """Log provider-specific error messages."""
        error_lower = error_msg.lower()
        
        if "authentication" in error_lower or "api key" in error_lower or "401" in error_lower:
            logger.error(f"{provider}: Invalid API Key or authentication failed")
        elif "rate limit" in error_lower or "quota" in error_lower or "429" in error_lower:
            logger.warning(f"{provider}: Rate limit exceeded or quota depleted")
        elif "not found" in error_lower or "404" in error_lower:
            logger.error(f"{provider}: Model not found or unavailable")
        elif "connection" in error_lower or "timeout" in error_lower or "503" in error_lower:
            logger.error(f"{provider}: Network connection issue or server unavailable")
        else:
            logger.error(f"{provider}: {error_msg[:200]}")
    

    
    def structured_chat(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> T:
        """
        Send a chat request with structured output using Instructor.
        
        This method uses Pydantic models to guarantee the response matches
        the expected schema. Instructor handles retries for validation errors.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class defining expected response structure
            temperature: Sampling temperature (0-1)
            max_retries: Number of retries for validation failures
        
        Returns:
            T: Validated Pydantic model instance
        
        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors = []
        
        # Try OpenAI first if available
        if self._providers_available["openai"]:
            try:
                return self._call_openai_structured(
                    messages, response_model, temperature, max_retries
                )
            except Exception as e:
                error_msg = f"OpenAI structured call failed: {str(e)}"
                errors.append(error_msg)
                self._log_error("OpenAI", str(e))
                if self._providers_available["gemini"]:
                    logger.info("Falling back to Gemini for structured output...")
        
        # Fallback to Gemini
        if self._providers_available["gemini"]:
            try:
                return self._call_gemini_structured(
                    messages, response_model, temperature, max_retries
                )
            except Exception as e:
                error_msg = f"Gemini structured call failed: {str(e)}"
                errors.append(error_msg)
                self._log_error("Gemini", str(e))
        
        # All providers failed
        raise AllProvidersFailedError(
            f"All LLM providers failed for structured output:\n" + "\n".join(errors)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_openai_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float,
        max_retries: int,
    ) -> T:
        """Call OpenAI with Instructor for structured output."""
        try:
            # Patch client with Instructor
            client = instructor.from_openai(self._openai_client)
            
            return client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_retries=max_retries,
            )
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit: {e}")
        except openai.APIError as e:
            raise APIError(f"OpenAI API error: {e}")
        except Exception as e:
            raise APIError(f"OpenAI unexpected error: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_gemini_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float,
        max_retries: int,
    ) -> T:
        """Call Gemini with Instructor for structured output."""
        try:
            # Patch client with Instructor
            client = instructor.from_gemini(
                client=self._gemini_client,
                mode=instructor.Mode.GEMINI_JSON,
            )
            
            # Convert messages to Gemini format
            contents = self._convert_messages_to_gemini(messages)
            
            return client.chat.completions.create(
                model=self.google_model,
                messages=[{"role": "user", "content": contents}],
                response_model=response_model,
                max_retries=max_retries,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                raise RateLimitError(f"Gemini rate limit: {e}")
            raise APIError(f"Gemini API error: {e}")
    

    
    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt for Gemini.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System Instructions:\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant:\n{content}\n")
            else:  # user
                prompt_parts.append(f"User:\n{content}\n")
        
        return "\n".join(prompt_parts)
    
    def get_provider_name(self) -> str:
        """Return the name of the currently active provider."""
        return self.active_provider or "unknown"
    

