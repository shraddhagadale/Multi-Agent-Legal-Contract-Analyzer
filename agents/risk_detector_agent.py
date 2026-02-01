"""
Risk Detector Agent

This agent analyzes legal clauses to identify potential risks, red flags,
and problematic language in NDA agreements.
"""

import json
from typing import Dict, Any, List, Optional


class RiskDetectorAgent:
    """
    Agent responsible for detecting risks in legal clauses.
    
    Uses the LLM to identify potential legal risks, unfair terms,
    and problematic language with severity ratings and recommendations.
    """

    def __init__(self, llm_manager, prompt_manager):
        """
        Initialize the Risk Detector Agent.
        
        Args:
            llm_manager: LLMProviderManager instance for making LLM calls
            prompt_manager: PromptManager instance for loading prompts
        """
        self.llm = llm_manager
        self.prompt_manager = prompt_manager
        
        # Agent configuration
        self.role = "Legal Risk Assessment Expert"
        self.goal = "Identify potential risks, red flags, and problematic language in NDA clauses"

    def detect_risks(
        self,
        clause: Dict[str, Any],
        classification: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect risks in a single clause.
        
        Args:
            clause: Clause dictionary with 'clause_id' and 'clause_text'
            classification: Optional classification dictionary for context
        
        Returns:
            Risk assessment dictionary with identified risks and recommendations
        """
        try:
            # Load and format the prompt
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(
                clause_text=clause['clause_text'],
                clause_id=clause['clause_id'],
                clause_category=classification.get('category', 'Unknown') if classification else 'Unknown'
            )
            
            # Build messages for the LLM
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a {self.role}. Your goal is to {self.goal}. "
                        "You are a senior legal risk analyst with decades of experience in "
                        "contract negotiation and risk assessment. You have an exceptional ability to "
                        "spot problematic language, unfair terms, and potential legal pitfalls in "
                        "commercial agreements. Your expertise helps protect parties from unfavorable terms. "
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
            risk_assessment = self._parse_response(response)
            risk_assessment['original_clause'] = clause
            risk_assessment['classification'] = classification
            
            return risk_assessment

        except Exception as e:
            print(f"[Risk Detector] ❌ Error detecting risks: {str(e)}")
            return self._create_fallback_risk_assessment(clause)

    def detect_risks_multiple_clauses(
        self,
        clauses: List[Dict[str, Any]],
        classifications: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect risks in multiple clauses.
        
        Args:
            clauses: List of clause dictionaries
            classifications: Optional list of classification dictionaries
        
        Returns:
            List of risk assessment dictionaries
        """
        risk_assessments = []
        
        # If classifications not provided, use None for each
        if classifications is None:
            classifications = [None] * len(clauses)
        
        for i, (clause, classification) in enumerate(zip(clauses, classifications)):
            clause_id = clause.get('clause_id', f'clause_{i+1}')
            print(f"[Risk Detector] 🔍 Assessing risks for: {clause_id}")
            
            risk_assessment = self.detect_risks(clause, classification)
            risk_assessments.append(risk_assessment)
        
        # Summary
        high_risks = sum(1 for r in risk_assessments if r.get('risk_level') == 'HIGH')
        medium_risks = sum(1 for r in risk_assessments if r.get('risk_level') == 'MEDIUM')
        low_risks = sum(1 for r in risk_assessments if r.get('risk_level') == 'LOW')
        
        print(f"[Risk Detector] ✅ Completed risk assessment for {len(risk_assessments)} clauses")
        print(f"[Risk Detector] 📊 Summary: {high_risks} HIGH, {medium_risks} MEDIUM, {low_risks} LOW")
        
        return risk_assessments

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("risk_detector_prompt")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured risk assessment data.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Parsed risk assessment dictionary
        """
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                risk_assessment = json.loads(json_str)
                return risk_assessment
            
            print("[Risk Detector] ⚠️ No valid JSON found in response")
            return self._create_fallback_risk_assessment({})
            
        except json.JSONDecodeError as e:
            print(f"[Risk Detector] ⚠️ JSON parsing error: {str(e)}")
            print(f"[Risk Detector] Raw response: {response[:500]}...")
            return self._create_fallback_risk_assessment({})
        except Exception as e:
            print(f"[Risk Detector] ⚠️ Error parsing response: {str(e)}")
            return self._create_fallback_risk_assessment({})

    def _create_fallback_risk_assessment(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback risk assessment when parsing fails.
        
        Args:
            clause: The original clause dictionary
        
        Returns:
            Fallback risk assessment dictionary
        """
        return {
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
