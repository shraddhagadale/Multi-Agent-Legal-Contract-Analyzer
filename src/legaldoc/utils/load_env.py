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
    
    Returns a dictionary containing API keys and model settings.
    
    Returns:
        Dict with configuration values:
        - OPENAI_API_KEY: OpenAI API key
        - OPENAI_MODEL_NAME: OpenAI model to use (default: gpt-4o-mini)
    """
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    }
