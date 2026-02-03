"""
Environment configuration loader for LegalDoc AI.

Loads environment variables from .env file and provides
centralized configuration access.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables once at import time
load_dotenv()


def get_config() -> Dict[str, Optional[str]]:
    """
    Centralized configuration loader.
    
    Returns a dictionary containing all API keys and Model settings.
    
    Returns:
        Dict with configuration values:
        - OPENAI_API_KEY: OpenAI API key
        - GOOGLE_API_KEY: Google AI API key
        - GEMINI_API_KEY: Alias for Google AI API key (for compatibility)
        - OPENAI_MODEL_NAME: OpenAI model to use (default: gpt-4o-mini)
        - GOOGLE_MODEL_NAME: Google model to use (default: gemini-2.0-flash)
    """
    return {
        # API Keys
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        
        # Model Configurations (with defaults)
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        "GOOGLE_MODEL_NAME": os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")
    }
