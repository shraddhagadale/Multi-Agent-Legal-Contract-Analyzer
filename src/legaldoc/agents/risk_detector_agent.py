"""
Risk Detector Agent

This agent analyzes legal clauses to identify potential risks, red flags,
and problematic language in NDA agreements.
"""

import logging
from typing import Dict, Any, List, Optional

from legaldoc.utils.schemas import RiskAssessmentResult
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class RiskDetectorAgent(BaseAgent):
    """
    Agent responsible for detecting risks in legal clauses.
    
    Uses the LLM to identify potential legal risks, unfair terms,
    and problematic language with severity ratings and recommendations.
    """
    
    @property
    def role(self) -> str:
        return "Legal Risk Assessment Expert"
    
    @property
    def goal(self) -> str:
        return "Identify potential risks, red flags, and problematic language in NDA clauses"
    
    @property
    def prompt_name(self) -> str:
        return "risk_detector_prompt"
    
    @property
    def expertise(self) -> str:
        return (
            "You are a senior legal risk analyst with decades of experience in "
            "contract negotiation and risk assessment. You have an exceptional ability to "
            "spot problematic language, unfair terms, and potential legal pitfalls in "
            "commercial agreements. Your expertise helps protect parties from unfavorable terms."
        )

    def detect_risks(
        self,
        clause: Dict[str, Any],
        classification: Optional[Dict[str, Any]] = None,
        document_summary: str = "No document context available.",
    ) -> Dict[str, Any]:
        """
        Detect risks in a single clause.
        
        Args:
            clause: Clause dictionary with 'clause_id' and 'clause_text'
            classification: Optional classification dictionary for context
            document_summary: Summary context from the Document Analyzer agent
        
        Returns:
            Risk assessment dictionary with identified risks and recommendations
        """
        try:
            # Load and format the prompt
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(
                clause_text=clause['clause_text'],
                clause_id=clause['clause_id'],
                clause_category=classification.get('category', 'Unknown') if classification else 'Unknown',
                document_summary=document_summary,
            )
            
            # Call the LLM with structured output
            risk_assessment = self._call_llm_structured(user_prompt, RiskAssessmentResult)
            
            # Convert to dictionary and add context
            result = risk_assessment.model_dump()
            result['original_clause'] = clause
            result['classification'] = classification
            
            return result

        except Exception as e:
            logger.error(f"Error detecting risks for clause {clause.get('clause_id', 'unknown')}: {e}")
            return self._create_fallback_risk_assessment(clause)

    def detect_risks_multiple_clauses(
        self,
        clauses: List[Dict[str, Any]],
        classifications: Optional[List[Dict[str, Any]]] = None,
        document_summary: str = "No document context available.",
    ) -> List[Dict[str, Any]]:
        """
        Detect risks in multiple clauses.
        
        For clauses that contain numbered sub-sections (e.g., 2.1, 2.2, 2.3),
        each sub-section is analyzed individually so that risky language in one
        sub-section is not buried by surrounding clean sub-sections. Results are
        then merged back to the parent clause level.
        
        Args:
            clauses: List of clause dictionaries
            classifications: Optional list of classification dictionaries
            document_summary: Summary context from the Document Analyzer agent
        
        Returns:
            List of risk assessment dictionaries
        """
        # If classifications not provided, use None for each
        if classifications is None:
            classifications = [None] * len(clauses)

        results = []
        for i, clause in enumerate(clauses):
            classification = classifications[i] if i < len(classifications) else None

            # Analyze the full clause directly to preserve context
            # (Sub-clause splitting removed to avoid losing context between related sub-sections)
            results.append(self.detect_risks(clause, classification, document_summary))

        return results

    # Removed _split_into_subclauses and _merge_subclause_risks (deprecated)

    def _create_fallback_risk_assessment(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback risk assessment when parsing fails.
        
        Uses conservative defaults (MEDIUM risk, 0.5 score) so that clauses
        whose analysis failed are flagged for manual review rather than
        silently passed as safe.
        
        Args:
            clause: The original clause dictionary
        
        Returns:
            Fallback risk assessment dictionary
        """
        return {
            "clause_id": clause.get('clause_id', 'unknown'),
            "risk_level": "MEDIUM",
            "risk_score": 0.5,
            "identified_risks": [
                {
                    "risk_type": "Analysis Failed",
                    "description": "Risk analysis could not be completed",
                    "severity": "MEDIUM",
                    "impact": "Unable to assess potential impact"
                }
            ],
            "recommendations": [
                "Manual review recommended due to analysis failure"
            ],
            "overall_assessment": "Risk assessment failed - manual review required",
            "original_clause": clause
        }
