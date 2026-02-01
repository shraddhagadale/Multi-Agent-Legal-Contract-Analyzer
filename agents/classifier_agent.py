"""
Clause Classifier Agent

This agent classifies legal clauses into specific categories based on
their content and purpose.
"""

import json
from typing import Dict, Any, List


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
                        "quickly identify the purpose and category of any legal clause. "
                        "Always respond with valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # Call the LLM
            response = self.llm.chat(messages)
            
            # Parse the response
            classification = self._parse_response(response)
            classification['original_clause'] = clause
            
            return classification

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
        
        for i, clause in enumerate(clauses):
            clause_id = clause.get('clause_id', f'clause_{i+1}')
            print(f"[Classifier Agent] 🏷️  Classifying clause: {clause_id}")
            
            classification = self.classify_clause(clause)
            classifications.append(classification)
        
        print(f"[Classifier Agent] ✅ Classified {len(classifications)} clauses")
        return classifications

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("classifier_prompt")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured classification data.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Parsed classification dictionary
        """
        try:
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                classification = json.loads(json_str)
                return classification
            
            print("[Classifier Agent] ⚠️ No valid JSON found in response")
            return self._create_fallback_classification({})
                
        except json.JSONDecodeError as e:
            print(f"[Classifier Agent] ⚠️ JSON parsing error: {str(e)}")
            return self._create_fallback_classification({})
        except Exception as e:
            print(f"[Classifier Agent] ⚠️ Error parsing response: {str(e)}")
            return self._create_fallback_classification({})

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