"""
Clause Classifier Agent

This agent classifies legal clauses into specific categories based on
their content and purpose.
"""

import logging
from typing import Dict, Any, List

from legaldoc.utils.schemas import ClassificationResult
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class ClauseClassifierAgent(BaseAgent):
    """
    Agent responsible for classifying legal clauses into categories.
    
    Uses the LLM to analyze clause content and assign appropriate
    legal categories with confidence scores.
    """
    
    @property
    def role(self) -> str:
        return "Legal Classification Expert"
    
    @property
    def goal(self) -> str:
        return "Accurately classify NDA clauses into specific legal categories"
    
    @property
    def prompt_name(self) -> str:
        return "classifier_prompt"
    
    @property
    def expertise(self) -> str:
        return (
            "You are a senior legal expert with extensive experience in contract law "
            "and legal document analysis. You have deep knowledge of NDA structures and can "
            "quickly identify the purpose and category of any legal clause."
        )

    def classify_clause(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single clause.
        
        Args:
            clause: Clause dictionary with 'clause_id' and 'clause_text'
        
        Returns:
            Classification dictionary with category, confidence, and reasoning
        """
        try:
            # Load and format the prompt
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(
                clause_text=clause['clause_text'],
                clause_id=clause['clause_id']
            )
            
            # Call the LLM with structured output
            classification = self._call_llm_structured(user_prompt, ClassificationResult)
            
            # Convert to dictionary and add original clause
            result = classification.model_dump()
            result['original_clause'] = clause
            
            return result

        except Exception as e:
            logger.error(f"Error classifying clause {clause.get('clause_id', 'unknown')}: {e}")
            return self._create_fallback_classification(clause)

    def classify_multiple_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple clauses.
        
        Args:
            clauses: List of clause dictionaries
        
        Returns:
            List of classification dictionaries
        """
        return [self.classify_clause(clause) for clause in clauses]

    def _create_fallback_classification(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback classification when parsing fails.
        
        Args:
            clause: The original clause dictionary
        
        Returns:
            Fallback classification dictionary
        """
        return {
            "clause_id": clause.get('clause_id', 'unknown'),
            "category": "Miscellaneous",
            "confidence": 0.0,
            "reasoning": "Classification failed - assigned to miscellaneous category",
            "subcategory": "Unknown",
            "original_clause": clause
        }
