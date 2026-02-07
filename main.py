"""
LegalDoc AI - NDA Analysis System

Main orchestration module that coordinates the multi-agent system
for analyzing legal documents (NDAs) to extract clauses, classify them,
and detect potential risks.

Usage:
    python main.py <document_file>           # Analyze a document and generate PDF report
    python main.py <document_file> -o out.pdf # Specify output file name
    python main.py --help                    # Show usage
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv

from legaldoc.utils import LLMProviderManager, PDFReportGenerator
from legaldoc.agents import (
    DocumentAnalyzerAgent,
    ClauseSplitterAgent,
    ClauseClassifierAgent,
    RiskDetectorAgent,
)

# Load environment variables
load_dotenv()


class LegalDocAI:
    """
    Main orchestrator for the Legal Document AI system.
    
    Coordinates four agents:
    0. DocumentAnalyzerAgent - Detects NDA type, parties, and summary
    1. ClauseSplitterAgent - Splits documents into clauses
    2. ClauseClassifierAgent - Classifies clauses by category
    3. RiskDetectorAgent - Detects risks in clauses
    
    Uses LLMProviderManager for automatic OpenAI/Gemini fallback.
    """
    
    def __init__(self):
        """
        Initialize LegalDocAI with Auto-Fallback Support.
        
        Uses LLMProviderManager to:
        1. Configure OpenAI (primary) based on API key presence
        2. Configure Gemini (fallback) based on API key presence
        3. Fall back automatically at runtime if primary fails
        """
        try:
            # Initialize LLM manager (handles provider configuration)
            self.llm_manager = LLMProviderManager()
            
        except Exception as e:
            sys.exit(f"FATAL ERROR: {e}")

        # Initialize agents with the LLM manager
        self.document_analyzer = DocumentAnalyzerAgent(self.llm_manager)
        self.splitter_agent = ClauseSplitterAgent(self.llm_manager)
        self.classifier_agent = ClauseClassifierAgent(self.llm_manager)
        self.risk_detector_agent = RiskDetectorAgent(self.llm_manager)

    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """
        Analyze a legal document through the full pipeline.
        
        Args:
            document_text: The full text of the legal document
        
        Returns:
            Dictionary containing all analysis results
        """
        # Step 0: Analyze document context
        print("Step 0: Analyzing document...", end=" ", flush=True)
        document_context = self.document_analyzer.analyze_document(document_text)
        doc_summary = document_context.get("formatted_summary", "No context available.")
        print(f"Detected: {document_context['document_type']}")

        # Step 1: Split document into clauses
        print("Step 1: Splitting document into clauses...", end=" ", flush=True)
        clauses = self.splitter_agent.split_document(document_text)
        print(f"Found {len(clauses)} clauses")

        # Step 2: Classify clauses (with document context)
        print(f"Step 2: Classifying {len(clauses)} clauses...", end=" ", flush=True)
        classifications = self.classifier_agent.classify_multiple_clauses(
            clauses, document_summary=doc_summary
        )
        print("Done")

        # Step 3: Assess risks (with document context)
        print(f"Step 3: Assessing risks...", end=" ", flush=True)
        risk_assessments = self.risk_detector_agent.detect_risks_multiple_clauses(
            clauses, classifications, document_summary=doc_summary
        )
        print("Done")
        
        # Categorize risks by level
        high_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'HIGH']
        medium_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'MEDIUM']
        low_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'LOW']

        # Step 4: Compile results
        print("Step 4: Compiling results...", end=" ", flush=True)
        results = {
            "document_context": document_context,
            "total_clauses": len(clauses),
            "clauses": clauses,
            "classifications": classifications,
            "risk_assessments": risk_assessments,
            "high_risk_clauses": high_risk_clauses,
            "medium_risk_clauses": medium_risk_clauses,
            "low_risk_clauses": low_risk_clauses,
            "provider_used": self.llm_manager.get_provider_name()
        }
        print("Done")
        
        return results

    def generate_report(self, results: Dict[str, Any], source_file: str, output_path: str) -> str:
        """
        Generate a PDF report from the analysis results.
        
        Args:
            results: Analysis results dictionary
            source_file: Name of the source document
            output_path: Path to save the PDF report
            
        Returns:
            Path to the generated PDF report
        """
        metadata = {
            "source_file": source_file,
            "analyzed_at": datetime.now().isoformat(),
            "provider": results.get("provider_used", "unknown")
        }
        
        generator = PDFReportGenerator(results, metadata)
        return generator.generate(output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LegalDoc AI - NDA Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py contract.txt              # Analyze and generate PDF report
    python main.py contract.txt -o my_report.pdf  # Specify output file name
        """
    )
    
    parser.add_argument(
        "document",
        type=str,
        help="Path to the document file to analyze"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output PDF file path (default: <document_name>_analysis.pdf)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the LegalDoc AI CLI."""
    # Parse arguments
    args = parse_arguments()
    
    # Define files directory
    FILES_DIR = "files"
    if not os.path.exists(FILES_DIR):
        os.makedirs(FILES_DIR)
    
    # Resolve document path
    doc_path = args.document
    if not os.path.exists(doc_path):
        # Check in files directory
        potential_path = os.path.join(FILES_DIR, doc_path)
        if os.path.exists(potential_path):
            doc_path = potential_path
        else:
            sys.exit(f"Error: Document not found: {args.document} (checked root and {FILES_DIR}/)")
    
    # Read the document
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            document_text = f.read()
    except Exception as e:
        sys.exit(f"Error reading file: {e}")
    
    print(f"LegalDoc AI Analysis System")
    print(f"Document: {doc_path} ({len(document_text)} chars)\n")
    
    # Initialize and run analysis
    try:
        legal_ai = LegalDocAI()
        results = legal_ai.analyze_document(document_text)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Generate output filename from input filename
            base_name = os.path.splitext(os.path.basename(doc_path))[0]
            output_path = os.path.join(FILES_DIR, f"{base_name}_analysis.pdf")
        
        # Generate report
        print(f"Step 5: Generating PDF report...", end=" ", flush=True)
        report_path = legal_ai.generate_report(results, doc_path, output_path)
        print("Done")
        print(f"\nAnalysis complete!")
        print(f"Report saved: {report_path}")
            
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
