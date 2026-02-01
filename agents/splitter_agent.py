"""
Clause Splitter Agent

This agent analyzes legal documents (particularly NDAs) and splits them
into logical, meaningful clauses for further analysis.
"""

import json
from typing import List, Dict, Any


class ClauseSplitterAgent:
    """
    Agent responsible for splitting legal documents into individual clauses.
    
    Uses the LLM to identify and extract distinct clauses from NDA documents,
    preserving structure and legal meaning.
    """

    def __init__(self, llm_manager, prompt_manager):
        """
        Initialize the Clause Splitter Agent.
        
        Args:
            llm_manager: LLMProviderManager instance for making LLM calls
            prompt_manager: PromptManager instance for loading prompts
        """
        self.llm = llm_manager
        self.prompt_manager = prompt_manager
        
        # Agent configuration
        self.role = "Legal Document Analyst"
        self.goal = "Break down NDA documents into logical, meaningful clauses"

    def split_document(self, document_text: str) -> List[Dict[str, Any]]:
        """
        Split a document into individual clauses.
        
        Args:
            document_text: The full text of the legal document
        
        Returns:
            List of clause dictionaries with structure:
            {
                "clause_id": str,
                "clause_number": str,
                "clause_title": str,
                "clause_text": str,
                "clause_type": str
            }
        """
        try:
            # Load and format the prompt
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(document_text=document_text)
            
            # Build messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {self.role}. Your goal is to {self.goal}. "
                        "You have years of experience in contract law and document analysis. "
                        "You specialize in understanding the structure and components of legal agreements, "
                        "particularly Non-Disclosure Agreements. Always respond with valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Call the LLM
            print(f"[Splitter Agent] 📄 Analyzing document ({len(document_text)} characters)...")
            response = self.llm.chat(messages)
            
            # Parse the response
            clauses = self._parse_response(response)
            print(f"[Splitter Agent] ✅ Extracted {len(clauses)} clauses")
            
            return clauses
        
        except Exception as e:
            print(f"[Splitter Agent] ❌ Error in clause splitting: {str(e)}")
            return []

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("splitter_prompt")

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into structured clause data.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            List of parsed clause dictionaries
        """
        try:
            # Try to find JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                clauses = json.loads(json_str)
                return clauses
            
            # Try to find JSON object with clauses array
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if 'clauses' in result and isinstance(result['clauses'], list):
                    return result['clauses']
            
            print("[Splitter Agent] ⚠️ No valid JSON found in response")
            return []
        
        except json.JSONDecodeError as e:
            print(f"[Splitter Agent] ⚠️ JSON parsing error: {str(e)}")
            return []
        except Exception as e:
            print(f"[Splitter Agent] ⚠️ Error parsing response: {str(e)}")
            return []