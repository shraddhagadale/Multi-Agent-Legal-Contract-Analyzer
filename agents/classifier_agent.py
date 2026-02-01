import json
from crewai import Agent
from crewai import Task
from typing import Dict, Any
import os 

class ClauseClassifierAgent:

    def __init__(self, llm, prompt_manager):
        """
        Initialize the Clause Classifier Agent.
        
        Args:
            llm: Language model instance
            prompt_manager: PromptManager instance for loading provider-specific prompts
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        return Agent(
            role = "Legal Classification Expert",
            goal = "Accurately classify NDA clauses into specific legal categories on content and purpose",
            backstory = """You are a senior legal expert with extensive experience in contract law
            and legal document analysis. You have deep knowledge of NDA structures and can 
            quickly identify the purpose and catrgory of any legal clause. Your expertise
            spans confidentiality agreements, intellectual property law, and commercial contracts.""",
            verbose = True,
            allow_delegation = False,
            llm = self.llm,
            memory = False 
        )
    
    def classify_clause(self, clause : Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self._load_prompt_template().format(
                clause_text = clause['clause_text'],
                clause_id = clause['clause_id']
            )
            
            task = Task(
                description=prompt,
                expected_output="JSON classification of the clause"
            )
            result = self.agent.execute_task(task)

            classification = self._parse_response(result)

            classification['original_clause'] = clause

            return classification

        except Exception as e:
            print(f"Error in clause classification: {str(e)}")
            return self._create_fallback_classification(clause)
        
    def classify_multiple_clauses(self, clauses: list) -> list:
        classifications = [] 
        for clause in clauses:
            print(f"CLassifying clause: {clause.get('clause_id','unknown')}")
            classification = self.classify_clause(clause)
            classifications.append(classification)

        return classifications

    def _load_prompt_template(self):
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("classifier_prompt")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the AI's response into structured classification data.
        """
        try:
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                classification = json.loads(json_str)
                return classification
            else:
                print("No valid JSON found in response")
                return self._create_fallback_classification({})
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return self._create_fallback_classification({})
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return self._create_fallback_classification({})
    
    def _create_fallback_classification(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback classification when parsing fails.
        """
        return {
            "clause_id": clause.get('clause_id', 'unknown'),
            "category": "Miscellaneous",
            "confidence": 0.0,
            "reasoning": "Classification failed - assigned to miscellaneous category",
            "subcategory": "Unknown",
            "original_clause": clause
        }
    
    def validate_classification(self, classification: Dict[str, Any]) -> bool:
    
        required_fields = ['clause_id', 'category', 'confidence', 'reasoning']
        
        for field in required_fields:
            if field not in classification:
                print(f"Missing required field '{field}' in classification")
                return False
        
        # Validate confidence score is between 0 and 1
        confidence = classification.get('confidence', 0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            print(f"Invalid confidence score: {confidence}")
            return False
        
        return True