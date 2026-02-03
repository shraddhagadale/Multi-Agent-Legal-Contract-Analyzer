"""
Centralized Pydantic Schemas for LLM Structured Outputs

This module defines all response schemas used by the legal document analysis agents.
Using Pydantic models ensures type safety, automatic validation, and consistent
data structures across the application.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Splitter Agent Schemas
# =============================================================================

class Clause(BaseModel):
    """Represents a single clause extracted from a legal document."""
    
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
    clause_type: str = Field(
        description="The legal category of the clause (e.g., 'Confidentiality', 'Term', 'Definitions')"
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
    category: str = Field(
        description="Primary legal category (e.g., 'Confidentiality Obligations', 'Term and Termination')"
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
        description="Explanation of why this classification was chosen"
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
    severity: str = Field(
        description="Severity level: 'LOW', 'MEDIUM', or 'HIGH'"
    )
    impact: str = Field(
        description="Potential impact of this risk on the parties"
    )


class RiskAssessmentResult(BaseModel):
    """Response schema for the Risk Detector Agent."""
    
    clause_id: str = Field(
        description="The ID of the clause being assessed"
    )
    risk_level: str = Field(
        description="Overall risk level: 'LOW', 'MEDIUM', 'HIGH', or 'NONE'"
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
        description="List of recommendations to mitigate the identified risks"
    )
    overall_assessment: str = Field(
        description="Summary assessment of the clause's risk profile"
    )
