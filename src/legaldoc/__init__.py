"""
LegalDoc AI - Multi-Agent Legal Document Analysis System

A sophisticated AI-powered system for analyzing legal documents (particularly NDAs)
using a multi-agent architecture with automatic LLM provider fallback.

Features:
- Document type detection (mutual/unilateral NDA)
- Clause extraction and splitting
- Legal category classification
- Risk detection and assessment
- PDF report generation

Usage:
    from legaldoc.utils import LLMProviderManager
    from legaldoc.agents import DocumentAnalyzerAgent, ClauseSplitterAgent
    
    llm = LLMProviderManager()
    analyzer = DocumentAnalyzerAgent(llm)
    context = analyzer.analyze_document(document_text)
    
    splitter = ClauseSplitterAgent(llm)
    clauses = splitter.split_document(document_text)
"""

__version__ = "1.0.0"
__author__ = "LegalDoc AI Team"

# Re-export main components for convenience
from legaldoc.agents import (
    BaseAgent,
    DocumentAnalyzerAgent,
    ClauseSplitterAgent,
    ClauseClassifierAgent,
    RiskDetectorAgent,
)
from legaldoc.utils import (
    LLMProviderManager,
    PDFReportGenerator,
    get_config,
)

__all__ = [
    # Version
    "__version__",
    # Agents
    "BaseAgent",
    "DocumentAnalyzerAgent",
    "ClauseSplitterAgent",
    "ClauseClassifierAgent",
    "RiskDetectorAgent",
    # Utils
    "LLMProviderManager",
    "PDFReportGenerator",
    "get_config",
]
