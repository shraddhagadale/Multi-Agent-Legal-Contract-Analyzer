"""
Clause Splitter Agent

This agent analyzes legal documents (particularly NDAs) and splits them
into logical, meaningful clauses for further analysis.
"""

from typing import List, Dict, Any

from utils.schemas import SplitterResponse, Clause


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
                        "particularly Non-Disclosure Agreements."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Call the LLM with structured output
            response = self.llm.structured_chat(
                messages=messages,
                response_model=SplitterResponse
            )
            
            # Convert Pydantic models to dictionaries for downstream compatibility
            clauses = [clause.model_dump() for clause in response.clauses]
            
            return clauses
        
        except Exception as e:
            raise Exception(f"Clause splitting failed: {str(e)}")

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("splitter_prompt")