"""
Agents package for LegalDoc AI

This package provides the multi-agent system for legal document analysis:
- BaseAgent: Abstract base class for all agents
- ClauseSplitterAgent: Splits documents into individual clauses
- ClauseClassifierAgent: Classifies clauses into legal categories
- RiskDetectorAgent: Detects risks and problematic language in clauses
"""

from .base_agent import BaseAgent
from .splitter_agent import ClauseSplitterAgent
from .classifier_agent import ClauseClassifierAgent
from .risk_detector_agent import RiskDetectorAgent

__all__ = [
    "BaseAgent",
    "ClauseSplitterAgent",
    "ClauseClassifierAgent",
    "RiskDetectorAgent",
]
