"""
Centralized Pydantic Schemas for LLM Structured Outputs

This module defines all response schemas used by the legal document analysis agents.
Using Pydantic models ensures type safety, automatic validation, and consistent
data structures across the application.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Shared Type Definitions
# =============================================================================



CategoryType = Literal[
    "Definitions",
    "Confidentiality",
    "Permitted Disclosures",
    "Obligations",
    "Term and Duration",
    "Termination",
    "Return of Materials",
    "Remedies",
    "Indemnification",
    "Non-Compete",
    "Non-Solicitation",
    "Governing Law",
    "Dispute Resolution",
    "Notices",
    "Assignment",
    "Amendments",
    "Severability",
    "Entire Agreement",
    "Waiver",
    "Recitals",
    "Execution",
    "Miscellaneous",
]

RiskLevel = Literal["NONE", "LOW", "MEDIUM", "HIGH"]

Severity = Literal["LOW", "MEDIUM", "HIGH"]

NDAType = Literal["Mutual_NDA", "Unilateral_NDA", "Unknown"]


# =============================================================================
# Document Analyzer Agent Schemas (for Section 4)
# =============================================================================

class PartyInfo(BaseModel):
    """Information about a party in the NDA."""

    name: str = Field(
        description="Name of the party as stated in the document"
    )
    role: str = Field(
        description="Role in the NDA (e.g., 'Disclosing Party', 'Receiving Party')"
    )


class DocumentAnalysis(BaseModel):
    """Response schema for the Document Analyzer Agent."""

    document_type: NDAType = Field(
        description="The type of NDA: mutual_nda, unilateral_nda, or unknown"
    )
    parties: List[PartyInfo] = Field(
        description="List of parties involved in the NDA"
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Effective date of the NDA if stated"
    )
    summary: str = Field(
        description=(
            "A 2-4 sentence summary of the NDA's purpose, "
            "key terms, and scope"
        )
    )
    key_observations: List[str] = Field(
        description=(
            "Notable structural or substantive observations about the NDA "
            "(e.g., 'unilateral NDA favoring Company X', 'missing standard exclusions')"
        )
    )



# =============================================================================
# Splitter Agent Schemas
# =============================================================================

class Clause(BaseModel):
    """Represents a single clause extracted from an NDA document."""

    clause_id: str = Field(
        description="Unique identifier for the clause (e.g., 'clause_1')"
    )
    clause_number: str = Field(
        description="The section or clause number as it appears in the document (e.g., '1.1', '2.a')"
    )
    clause_title: str = Field(
        description="The heading or title of the clause"
    )
    clause_text: str = Field(
        description="The full verbatim text content of the clause"
    )


class SplitterResponse(BaseModel):
    """Response schema for the Clause Splitter Agent."""

    clauses: List[Clause] = Field(
        description="List of all clauses extracted from the document"
    )



# =============================================================================
# Classifier Agent Schemas
# =============================================================================

class ClassificationResult(BaseModel):
    """Represents the classification result for a single clause."""

    clause_id: str = Field(
        description="The ID of the clause being classified"
    )
    category: CategoryType = Field(
        description="Primary legal category"
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="More specific subcategory within the main category"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Step-by-step explanation of why this classification was chosen"
    )


# =============================================================================
# Risk Detector Agent Schemas
# =============================================================================

class IdentifiedRisk(BaseModel):
    """Represents a single risk identified in a clause."""

    risk_type: str = Field(
        description="Category of the risk (e.g., 'Overbroad Language', 'Missing Limitation')"
    )
    description: str = Field(
        description="Detailed description of the identified risk"
    )
    severity: Severity = Field(
        description="Severity level of this specific risk"
    )
    impact: str = Field(
        description="Potential impact of this risk on the parties"
    )


class RiskAssessmentResult(BaseModel):
    """Response schema for the Risk Detector Agent."""

    clause_id: str = Field(
        description="The ID of the clause being assessed"
    )
    risk_level: RiskLevel = Field(
        description="Overall risk level for this clause"
    )
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numerical risk score between 0.0 and 1.0"
    )
    identified_risks: List[IdentifiedRisk] = Field(
        default_factory=list,
        description="List of specific risks identified in the clause"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of specific, actionable recommendations to mitigate risks"
    )
    overall_assessment: str = Field(
        description="Summary assessment of the clause's risk profile"
    )
