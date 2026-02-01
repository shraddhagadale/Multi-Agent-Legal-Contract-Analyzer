"""
LLM Provider Manager with Auto-Fallback

This module manages LLM provider initialization with automatic fallback capabilities.
It prioritizes OpenAI, validates the connection, and falls back to Gemini if
issues are detected.
"""

import os
from typing import Optional, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .load_env import get_config

class LLMProviderManager:
    """
    Manages LLM initialization with built-in health checks and fallback logic.
    
    Priority:
    1. OpenAI (gpt-4o-mini)
    2. Google Gemini (gemini-1.5-pro)
    """
    
    def __init__(self):
        # Load centralized configuration
        self.config = get_config()
        
        # Strictly use config
        self.openai_key = self.config["OPENAI_API_KEY"]
        self.google_key = self.config.get("GEMINI_API_KEY") or self.config.get("GOOGLE_API_KEY")  # Support both
        
        self.active_llm = None
        self.active_provider = None
        
        # Initialize immediately
        self._initialize_best_provider()

    def _initialize_best_provider(self):
        """Try providers in priority order."""
        
        # Get model names from config
        openai_model = self.config["OPENAI_MODEL_NAME"]
        google_model = self.config["GOOGLE_MODEL_NAME"]
        
        # 1. Try OpenAI
        print(f"\n[LLM Manager] 🔄 Attempting to initialize OpenAI ({openai_model})...")
        if self.openai_key:
            try:
                llm = ChatOpenAI(
                    model=openai_model,
                    temperature=0.1,
                    api_key=self.openai_key,
                    max_retries=1 # Don't retry too much on init
                )
                
                if self._validate_connection(llm, "OpenAI"):
                    self.active_llm = llm
                    self.active_provider = "openai"
                    print("[LLM Manager] ✅ OpenAI successfully initialized and verified.")
                    return
            except Exception as e:
                print(f"[LLM Manager] ⚠️ OpenAI Initialization Failed: {str(e)}")
        else:
            print("[LLM Manager] ⚠️ Skipping OpenAI: No API Key found.")

        # 2. Fallback to Gemini
        print(f"\n[LLM Manager] 🔄 Attempting to initialize Gemini ({google_model})...")
        if self.google_key:
            try:
                # Use CrewAI's native LLM class (avoids CrewAI bug #2645)
                from crewai import LLM
                
                llm = LLM(
                    model=f"gemini/{google_model}",  # gemini/ prefix required for LiteLLM
                    api_key=self.google_key,
                    temperature=0.1
                )
                
                if self._validate_connection(llm, "Gemini"):
                    self.active_llm = llm
                    self.active_provider = "gemini"
                    print("[LLM Manager] ✅ Gemini successfully initialized and verified.")
                    return
            except Exception as e:
                print(f"[LLM Manager] ⚠️ Gemini Initialization Failed: {str(e)}")
        else:
            print("[LLM Manager] ⚠️ Skipping Gemini: No API Key found.")

        # 3. Critical Failure
        raise RuntimeError(
            "CRITICAL: All LLM providers failed to initialize.\n"
            "Please check your API keys in the .env file.\n"
            "- OPENAI_API_KEY\n"
            "- GEMINI_API_KEY (or GOOGLE_API_KEY)"
        )

    def _validate_connection(self, llm: Any, provider_name: str) -> bool:
        """
        Level 2 Error Detection: Proactive 'Ping' test.
        Returns True if healthy, False if failed.
        """
        try:
            # Send a tiny "ping" message
            print(f"[LLM Manager] 🧪 Running health check on {provider_name}...")
            
            # Check if it's a CrewAI LLM or LangChain LLM
            if hasattr(llm, 'call'):  # CrewAI LLM
                llm.call("Hello")
            else:  # LangChain LLM (OpenAI)
                llm.invoke([HumanMessage(content="Hello")])
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            self._classify_error(provider_name, error_msg)
            return False

    def _classify_error(self, provider: str, error_msg: str):
        """Log specific understandable error messages."""
        prefix = f"[LLM Manager] ❌ {provider} Error:"
        
        if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
            print(f"{prefix} Invalid API Key or Authentication failed.")
        elif "rate limit" in error_msg or "quota" in error_msg or "429" in error_msg:
            print(f"{prefix} Rate limit exceeded or quota depleted.")
        elif "not found" in error_msg or "404" in error_msg:
            print(f"{prefix} Model not found or not available.")
        elif "connection" in error_msg or "timeout" in error_msg or "503" in error_msg:
            print(f"{prefix} Network connection issue or server down.")
        else:
            print(f"{prefix} Unexpected error: {error_msg}")

    def get_llm(self):
        """Return the active, validated LLM instance."""
        if not self.active_llm:
            raise RuntimeError("No LLM initialized. Manager failed.")
        return self.active_llm

    def get_provider_name(self) -> str:
        """Return the name of the active provider."""
        return self.active_provider or "unknown"
