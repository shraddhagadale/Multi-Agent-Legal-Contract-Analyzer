"""
LegalDoc AI - NDA Analysis System

Main orchestration module that coordinates the multi-agent system
for analyzing legal documents (NDAs) to extract clauses, classify them,
and detect potential risks.

Usage:
    python main.py <document_file>              # Analyze a document and generate PDF report
    python main.py <document_file> -o out.pdf   # Specify output file name
    python main.py <document_file> -v           # Run with verbose agent output
    python main.py --help                       # Show usage
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

from legaldoc.utils import LLMClient
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
    
    Uses LLMClient for OpenAI-powered analysis.
    """
    
    def __init__(self):
        """
        Initialize LegalDocAI with LLMClient.
        
        Sets up the OpenAI-powered LLM client and initializes
        all analysis agents.
        """
        try:
            # Initialize LLM client
            self.llm_client = LLMClient()

        except Exception as e:
            sys.exit(f"FATAL ERROR: {e}")

        # Initialize agents with the LLM client
        self.document_analyzer = DocumentAnalyzerAgent(self.llm_client)
        self.splitter_agent = ClauseSplitterAgent(self.llm_client)
        self.classifier_agent = ClauseClassifierAgent(self.llm_client)
        self.risk_detector_agent = RiskDetectorAgent(self.llm_client)

    def process_document(self, document_text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Analyze a legal document through the full pipeline.
        
        Args:
            document_text: The full text of the legal document
            verbose: If True, print detailed output after each agent step
        
        Returns:
            Dictionary containing all analysis results
        """
        # Step 0: Analyze document context
        print("Step 0: Analyzing document...", end=" ", flush=True)
        document_context = self.document_analyzer.analyze_document(document_text)
        doc_summary = document_context.get("formatted_summary", "No context available.")
        print(f"Detected: {document_context['document_type']}")

        if verbose:
            print("\n--- Document Analysis ---")
            print(f"  Type: {document_context.get('document_type', 'N/A')}")
            parties = document_context.get('parties', [])
            if parties:
                parties_str = ", ".join(
                    f"{p.get('name', '?')} ({p.get('role', '?')})" if isinstance(p, dict) else str(p)
                    for p in parties
                )
                print(f"  Parties: {parties_str}")
            print(f"  Summary: {document_context.get('summary', 'N/A')}")
            observations = document_context.get('key_observations', [])
            if observations:
                print(f"  Key Observations:")
                for obs in observations:
                    print(f"    • {obs}")
            print(f"  Formatted Summary:\n    {doc_summary}")

        # Step 1: Split document into clauses
        print("Step 1: Splitting document into clauses...", end=" ", flush=True)
        clauses = self.splitter_agent.split_document(document_text)
        print(f"Found {len(clauses)} clauses")

        if verbose:
            print("\n--- Clauses Extracted ---")
            for c in clauses:
                ctext_preview = c.get('clause_text', '')[:150].replace('\n', ' ')
                if len(c.get('clause_text', '')) > 150:
                    ctext_preview += "..."
                print(f"  [{c.get('clause_id', '?')}] {c.get('clause_title', 'Untitled')}")
                print(f"    Text: {ctext_preview}")

        # Step 2: Classify clauses (with document context)
        print(f"Step 2: Classifying {len(clauses)} clauses...", end=" ", flush=True)
        classifications = self.classifier_agent.classify_multiple_clauses(
            clauses, document_summary=doc_summary
        )
        print("Done")

        if verbose:
            print("\n--- Classifications ---")
            for cl in classifications:
                conf = cl.get('confidence', 0)
                print(f"  [{cl.get('clause_id', '?')}] {cl.get('category', '?')} (confidence: {conf:.2f})")
                reasoning = cl.get('reasoning', '')
                reasoning_preview = reasoning[:120]
                if len(reasoning) > 120:
                    reasoning_preview += "..."
                print(f"    Reasoning: {reasoning_preview}")

        # Step 3: Assess risks (with document context)
        print(f"Step 3: Assessing risks...", end=" ", flush=True)
        risk_assessments = self.risk_detector_agent.detect_risks_multiple_clauses(
            clauses, classifications, document_summary=doc_summary
        )
        print("Done")

        if verbose:
            print("\n--- Risk Assessments ---")
            for r in risk_assessments:
                level = r.get('risk_level', 'NONE')
                score = r.get('risk_score', 0)
                print(f"  [{r.get('clause_id', '?')}] {level} (score: {score:.2f})")
                if r.get('identified_risks'):
                    for risk in r['identified_risks']:
                        desc = risk.get('description', '')
                        desc_preview = desc[:100]
                        if len(desc) > 100:
                            desc_preview += "..."
                        print(f"    ⚠ {risk.get('risk_type', '?')}: {desc_preview}")
                if r.get('recommendations'):
                    for rec in r['recommendations']:
                        rec_preview = rec[:100]
                        if len(rec) > 100:
                            rec_preview += "..."
                        print(f"    → {rec_preview}")
        
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
            "model_used": self.llm_client.get_model_name()
        }
        print("Done")
        
        return results

    def display_console_summary(self, results: Dict[str, Any]):
        """
        Display a structured summary of risks to the console using standard ANSI colors.
        """
        high_risks = results.get("high_risk_clauses", [])
        medium_risks = results.get("medium_risk_clauses", [])
        low_risks = results.get("low_risk_clauses", [])
        
        # ANSI formatting
        RED = "\033[91m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        HR = "-" * 60
        
        print(f"Model: {results.get('model_used', 'Unknown')}")
        
        print(f"\n{BOLD}=== ANALYSIS SUMMARY ==={RESET}")
        print(f"Total Clauses: {results.get('total_clauses', 0)}")
        print(f"High Risks:    {RED}{len(high_risks)}{RESET}")
        print(f"Medium Risks:  {YELLOW}{len(medium_risks)}{RESET}")
        print(f"Low Risks:     {GREEN}{len(low_risks)}{RESET}")
        print("\n")
        
        def print_risk_block(risk_data, title_color):
            original = risk_data.get('original_clause', {})
            # Clean up ID: "clause_3" -> "Clause 3"
            raw_id = original.get('clause_id', '?')
            clean_id = raw_id.replace('clause_', 'Clause ').replace('_', ' ').title()
            
            title = original.get('clause_title', 'Untitled')
            
            print(f"{HR}")
            # Format: Clause 3: Title
            print(f"{title_color}{BOLD}{clean_id}: {title}{RESET}")
            
            # Risks
            print(f"\n{BOLD}Identified Risks:{RESET}")
            for r in risk_data.get('identified_risks', []):
                risk_type = r.get('risk_type', 'General Risk')
                desc = r.get('description', '')
                print(f"  • {title_color}{risk_type}{RESET}: {desc}")
                
            # Recommendations
            recs = risk_data.get('recommendations', [])
            if recs:
                print(f"\n{BOLD}Recommendations:{RESET}")
                for rec in recs:
                    print(f"  • {rec}")
            
            print("\n")  # Extra spacing

        if high_risks:
            print(f"{RED}{BOLD}>>> HIGH RISK CLAUSES ({len(high_risks)}) <<<{RESET}")
            for item in high_risks:
                print_risk_block(item, RED)
                
        if medium_risks:
            print(f"{YELLOW}{BOLD}>>> MEDIUM RISK CLAUSES ({len(medium_risks)}) <<<{RESET}")
            for item in medium_risks:
                print_risk_block(item, YELLOW)
        
        if low_risks:
            print(f"{GREEN}{BOLD}>>> LOW RISK CLAUSES ({len(low_risks)}) <<<{RESET}")
            for item in low_risks:
                print_risk_block(item, GREEN)
    




def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LegalDoc AI - NDA Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py contract.txt              # Analyze document
    python main.py contract.txt -v           # Show detailed steps
        """
    )
    
    parser.add_argument(
        "document",
        type=str,
        help="Path to the document file to analyze"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Show detailed output from each agent step"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the LegalDoc AI CLI."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate document path
    doc_path = args.document
    if not os.path.exists(doc_path):
        sys.exit(f"Error: Document not found: {args.document}")
    
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
        results = legal_ai.process_document(document_text, verbose=args.verbose)
        
        # Display Console Summary
        legal_ai.display_console_summary(results)
        
        print(f"Analysis complete!")
            
    except Exception as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
