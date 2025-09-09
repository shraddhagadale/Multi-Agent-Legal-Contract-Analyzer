import json
from crewai import Agent
from typing import Dict, Any
import os

class RiskDetectorAgent:

    def __init__(self,llm):
        self.llm = llm
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            role = "Legal Risk Assessment Expert",
            goal = "Identify potential risks, red flags, and problematic language in NDA clauses",
            backstory = """ You are a senior legal risk analyst with decades of experience in
            contract negotiation and risk assessment. You have an exceptional ability to 
            spot problematic language, unfair terms, and potential legal pitfalls in 
            commercial agreements. Your expertise helps protect parties from unfavorable terms.""",
            verbose = True,
            allow_delegation = False,
            llm = self.llm,
            tools = [],
            memory = False
        )

    def detect_risks(self,clause,classification = None):
        try:
            prompt = self._load_prompt_template().format(
                clause_text = clause['clause_text'],
                clause_id = clause['clause_id'],
                clause_category = classification.get('category','Unknown') if classification else 'Unknown'
            )

            # Create a task for the agent to execute
            from crewai import Task
            task = Task(
                description=prompt,
                expected_output="JSON risk assessment of the clause"
            )
            result = self.agent.execute_task(task)

            risk_assessment = self._parse_response(result)

            risk_assessment['original_clause'] = clause
            risk_assessment['classification'] = classification

            return risk_assessment
        
        except Exception as e:
            print(f"Error in risk detection: {str(e)}")
            return self._create_fallback_risk_assessment(clause)

    def _load_prompt_template(self):
        prompt_path = os.path.join("prompts","risk_detector_prompt.txt")
        with open(prompt_path,'r') as f:
            return f.read()

    def detect_risks_multiple_clauses(self, clauses, classifications):
        risk_assessments = []
        for clause, classification in zip(clauses, classifications):
            risk_assessment = self.detect_risks(clause, classification)
            risk_assessments.append(risk_assessment)
        return risk_assessments

    def _parse_response(self,response : str):
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                risk_assessment = json.loads(json_str)
                return risk_assessment
            else:
                print("No valid JSON object found in response")
                return self._create_fallback_risk_assessment({})
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response}")
            return self._create_fallback_risk_assessment({})
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return self._create_fallback_risk_assessment({}) 

    def _create_fallback_risk_assessment(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        return{
            "clause_id": clause.get('clause_id', 'unknown'),
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "identified_risks": [
                {
                    "risk_type": "Analysis Failed",
                    "description": "Risk analysis could not be completed",
                    "severity": "UNKNOWN",
                    "impact": "Unable to assess potential impact"
                }
            ],
            "recommendations": [
                "Manual review recommended due to analysis failure"
            ],
            "overall_assessment": "Risk assessment failed - manual review required",
            "original_clause": clause
        }

    def validate_risk_assessment(self,risk_assessment):
        required_keys = [
            "clause_id", "risk_level", "risk_score", "identified_risks", 
            "recommendations", "overall_assessment", "original_clause"

        ]
        for key in required_keys:
            if key not in risk_assessment:
                print(f"Missing required key: {key}")
                return False
        return True
        
    def detect_risks_multiple_clauses(self, clauses: list, classifications: list = None) -> list:
        """
        Detect risks in multiple clauses.
        
        Args:
            clauses: List of clauses to analyze
            classifications: Optional list of classifications for the clauses
            
        Returns:
            List of risk assessments for each clause
        """
        risk_assessments = []
        
        # If classifications is not provided, use None for each clause
        if classifications is None:
            classifications = [None] * len(clauses)
            
        for clause, classification in zip(clauses, classifications):
            risk_assessment = self.detect_risks(clause, classification)
            risk_assessments.append(risk_assessment)
            
        return risk_assessments
    
    def get_high_risk_clauses(self, risk_assessments: list) -> list:
        high_risk_clauses = []
        for assessment in risk_assessments:
            if assessment.get("risk_level") == "HIGH":
                high_risk_clauses.append(assessment)
        return high_risk_clauses

    
