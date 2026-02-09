"""
Document Analyzer Agent

This agent performs initial analysis of NDA documents to determine
document type, identify parties, and produce a summary for downstream agents.
"""

import logging
from typing import Dict, Any

from legaldoc.utils.schemas import DocumentAnalysis
from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class DocumentAnalyzerAgent(BaseAgent):
    """
    Agent responsible for initial NDA document analysis.

    Determines NDA type (mutual/unilateral), identifies parties,
    and produces a summary that provides context for downstream agents
    (classifier, risk detector).
    """

    @property
    def role(self) -> str:
        return "Legal Document Analyst"

    @property
    def goal(self) -> str:
        return "Analyze NDA documents to determine type, parties, and key characteristics"

    @property
    def prompt_name(self) -> str:
        return "document_analyzer_prompt"

    @property
    def expertise(self) -> str:
        return (
            "You are a senior legal professional with extensive experience reviewing "
            "Non-Disclosure Agreements. You can quickly identify whether an NDA is "
            "mutual or unilateral, extract the parties involved, and spot key structural "
            "characteristics that inform detailed clause-level analysis."
        )

    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """
        Perform initial analysis of an NDA document.

        Args:
            document_text: The full text of the NDA document

        Returns:
            Dictionary containing document type, parties, summary,
            and key observations. Includes a 'formatted_summary' string
            suitable for injection into downstream prompts.
        """
        try:
            prompt_template = self._load_prompt_template()
            user_prompt = prompt_template.format(document_text=document_text)

            response = self._call_llm_structured(user_prompt, DocumentAnalysis)

            result = response.model_dump()

            # Build a formatted summary string for downstream agents
            parties_str = ", ".join(
                f"{p['name']} ({p['role']})" for p in result["parties"]
            )

            result["formatted_summary"] = (
                f"Document Type: {result['document_type']}\n"
                f"Parties: {parties_str}\n"
                f"Summary: {result['summary']}"
            )

            return result

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            # Return minimal context rather than failing the pipeline
            return {
                "document_type": "Unknown",
                "parties": [],
                "effective_date": None,
                "summary": "Document analysis could not be completed.",
                "key_observations": [
                    "Automated analysis failed — manual review recommended"
                ],
                "formatted_summary": (
                    "Document Type: Unknown\n"
                    "Parties: Unknown\n"
                    "Summary: Document analysis could not be completed."
                ),
            }
