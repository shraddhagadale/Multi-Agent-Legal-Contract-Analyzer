"""
Clause Classifier Agent

This agent classifies legal clauses into specific categories based on
their content and purpose.
"""

from typing import Dict, Any, List

from utils.schemas import ClassificationResult


class ClauseClassifierAgent:
    """
    Agent responsible for classifying legal clauses into categories.
    
    Uses the LLM to analyze clause content and assign appropriate
    legal categories with confidence scores.
    """

    def __init__(self, llm_manager, prompt_manager):
        """
        Initialize the Clause Classifier Agent.
        
        Args:
            llm_manager: LLMProviderManager instance for making LLM calls
            prompt_manager: PromptManager instance for loading prompts
        """
        self.llm = llm_manager
        self.prompt_manager = prompt_manager
        
        # Agent configuration
        self.role = "Legal Classification Expert"
        self.goal = "Accurately classify NDA clauses into specific legal categories"

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
            
            # Build messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {self.role}. Your goal is to {self.goal}. "
                        "You are a senior legal expert with extensive experience in contract law "
                        "and legal document analysis. You have deep knowledge of NDA structures and can "
                        "quickly identify the purpose and category of any legal clause."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Call the LLM with structured output
            classification = self.llm.structured_chat(
                messages=messages,
                response_model=ClassificationResult
            )
            
            # Convert to dictionary and add original clause
            result = classification.model_dump()
            result['original_clause'] = clause
            
            return result

        except Exception as e:
            print(f"[Classifier Agent] ❌ Error classifying clause: {str(e)}")
            return self._create_fallback_classification(clause)

    def classify_multiple_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple clauses.
        
        Args:
            clauses: List of clause dictionaries
        
        Returns:
            List of classification dictionaries
        """
        classifications = []
        
        for clause in clauses:
            classification = self.classify_clause(clause)
            classifications.append(classification)
        
        return classifications

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("classifier_prompt")

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