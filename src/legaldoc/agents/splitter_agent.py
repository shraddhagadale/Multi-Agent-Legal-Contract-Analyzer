"""
Clause Splitter Agent

This agent analyzes legal documents (particularly NDAs) and splits them
into logical, meaningful clauses for further analysis.
"""

import logging
from typing import List, Dict, Any

from legaldoc.utils.schemas import SplitterResponse
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class ClauseSplitterAgent(BaseAgent):
    """
    Agent responsible for splitting legal documents into individual clauses.
    
    Uses the LLM to identify and extract distinct clauses from NDA documents,
    preserving structure and legal meaning.
    """
    
    @property
    def role(self) -> str:
        return "Legal Document Analyst"
    
    @property
    def goal(self) -> str:
        return "Break down NDA documents into logical, meaningful clauses"
    
    @property
    def prompt_name(self) -> str:
        return "splitter_prompt"
    
    @property
    def expertise(self) -> str:
        return (
            "You have years of experience in contract law and document analysis. "
            "You specialize in understanding the structure and components of legal agreements, "
            "particularly Non-Disclosure Agreements."
        )

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
                "clause_text": str
            }
        
        Raises:
            Exception: If clause splitting fails
        """
        try:
            # Load and format the prompt
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(document_text=document_text)
            
            # Call the LLM with structured output
            response = self._call_llm_structured(user_prompt, SplitterResponse)
            
            # Convert Pydantic models to dictionaries for downstream compatibility
            return [clause.model_dump() for clause in response.clauses]
        
        except Exception as e:
            logger.error(f"Clause splitting failed: {e}")
            raise Exception(f"Clause splitting failed: {str(e)}")
