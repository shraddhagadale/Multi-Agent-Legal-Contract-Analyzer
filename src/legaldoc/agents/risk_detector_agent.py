"""
Risk Detector Agent

This agent analyzes legal clauses to identify potential risks, red flags,
and problematic language in NDA agreements.
"""

import re
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

            # Split into sub-clauses for granular analysis
            subclauses = self._split_into_subclauses(clause)

            if len(subclauses) > 1:
                # Analyze each sub-clause individually
                sub_results = []
                for subclause in subclauses:
                    sub_result = self.detect_risks(subclause, classification, document_summary)
                    sub_results.append(sub_result)

                # Merge: take the HIGHEST risk found across sub-clauses
                merged = self._merge_subclause_risks(clause, sub_results)
                results.append(merged)
            else:
                # Single clause (no sub-sections), analyze directly
                results.append(self.detect_risks(clause, classification, document_summary))

        return results

    def _split_into_subclauses(self, clause: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a clause into sub-clauses if it contains numbered sub-sections.

        Detects patterns like "2.1", "2.2", "(a)", "(b)" etc. and splits accordingly.
        If no sub-sections found, returns the original clause as a single-item list.

        Args:
            clause: Clause dictionary with clause_text, clause_number, clause_id, etc.

        Returns:
            List of clause dictionaries (one per sub-clause, or the original if no split)
        """
        text = clause.get('clause_text', '')
        clause_num = clause.get('clause_number', '')

        if not clause_num or not text:
            return [clause]

        # Pattern: look for "X.Y" sub-section markers where X is the clause number
        # e.g., for clause number "2", match "2.1", "2.2", etc.
        # They can appear at the start of the text or after a newline
        escaped_num = re.escape(clause_num)
        pattern = rf'(?:^|\n)\s*({escaped_num}\.\d+)\s'

        # Find all sub-section markers
        markers = list(re.finditer(pattern, text))

        if len(markers) < 2:
            # Need at least 2 sub-sections to justify splitting
            return [clause]

        subclauses = []

        # Capture any introductory text before the first sub-section marker.
        # This frames the meaning of all sub-sections (e.g., "Confidential
        # Information does not include information that...") and must be
        # preserved so each sub-clause is read in proper context.
        intro_text = text[:markers[0].start()].strip()

        for idx, match in enumerate(markers):
            sub_number = match.group(1)  # e.g., "2.1"
            start = match.start()
            # Clean leading newline from start position
            if text[start] == '\n':
                start += 1

            # End is either the start of the next marker or end of text
            if idx + 1 < len(markers):
                end = markers[idx + 1].start()
            else:
                end = len(text)

            sub_text = text[start:end].strip()

            if sub_text:
                # Prepend introductory context if it exists
                if intro_text:
                    sub_text = f"{intro_text}\n\n{sub_text}"

                # Derive a sub-number index (e.g., "2.1" → 1)
                sub_idx = sub_number.split('.')[-1] if '.' in sub_number else str(idx + 1)
                subclauses.append({
                    'clause_id': f"{clause['clause_id']}_sub_{sub_idx}",
                    'clause_number': sub_number,
                    'clause_title': clause.get('clause_title', '') + f" (§{sub_number})",
                    'clause_text': sub_text,
                    'clause_type': clause.get('clause_type', 'unknown'),
                    '_parent_clause_id': clause.get('clause_id'),
                })

        return subclauses if subclauses else [clause]

    def _merge_subclause_risks(
        self, parent_clause: Dict[str, Any], sub_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge sub-clause risk results into a single parent-level result.

        Takes the highest risk level found across all sub-clauses and collects
        all identified risks and recommendations.

        Args:
            parent_clause: The original (unsplit) parent clause dictionary
            sub_results: List of risk assessment results from each sub-clause

        Returns:
            Merged risk assessment dictionary at the parent clause level
        """
        risk_order = {"HIGH": 4, "MEDIUM": 3, "LOW": 2, "NONE": 1}
        sorted_results = sorted(
            sub_results,
            key=lambda r: risk_order.get(r.get('risk_level', 'NONE'), 0),
            reverse=True,
        )
        highest = sorted_results[0]

        # Collect ALL identified risks from all sub-clauses
        all_risks = []
        all_recommendations = []
        for r in sub_results:
            all_risks.extend(r.get('identified_risks', []))
            all_recommendations.extend(r.get('recommendations', []))

        # Deduplicate recommendations while preserving order
        unique_recs = list(dict.fromkeys(all_recommendations))

        return {
            "clause_id": parent_clause.get('clause_id', 'unknown'),
            "risk_level": highest.get('risk_level', 'NONE'),
            "risk_score": highest.get('risk_score', 0.0),
            "identified_risks": all_risks,
            "recommendations": unique_recs,
            "overall_assessment": highest.get('overall_assessment', ''),
            "original_clause": parent_clause,
            "classification": sub_results[0].get('classification') if sub_results else None,
            "sub_clause_results": sub_results,  # Keep for debug visibility
        }

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
