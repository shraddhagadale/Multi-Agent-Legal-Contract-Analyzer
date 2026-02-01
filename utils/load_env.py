import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables once at import time
load_dotenv()

def get_config() -> Dict[str, Optional[str]]:
    """
    Centralized configuration loader.
    Returns a dictionary containing all API keys and Model settings.
    """
    return {
        # API Keys
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),  # LiteLLM requires this
        
        # Model Configurations (with defaults)
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        "GOOGLE_MODEL_NAME": os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro")
    }

