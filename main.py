"""
LegalDoc AI - NDA Analysis System

Main orchestration module that coordinates the multi-agent system
for analyzing legal documents (NDAs) to extract clauses, classify them,
and detect potential risks.

Usage:
    python main.py <document_file>           # Analyze a document
    python main.py <document_file> --no-save # Analyze without saving JSON files
    python main.py --help                    # Show usage
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

from dotenv import load_dotenv

from utils.llm_provider_manager import LLMProviderManager
from utils.prompt_manager import PromptManager
from agents.splitter_agent import ClauseSplitterAgent
from agents.classifier_agent import ClauseClassifierAgent
from agents.risk_detector_agent import RiskDetectorAgent

# Load environment variables
load_dotenv()


class LegalDocAI:
    """
    Main orchestrator for the Legal Document AI system.
    
    Coordinates three agents:
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
        4. Load the correct prompts for the active provider
        """
        print("=" * 60)
        print("INITIALIZING LEGAL DOC AI SYSTEM")
        print("=" * 60)
        
        try:
            # Initialize LLM manager (handles provider configuration)
            self.llm_manager = LLMProviderManager()
            
            # Initialize Prompt Manager with the active provider
            self.prompt_manager = PromptManager(self.llm_manager)
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR During Initialization: {e}")
            raise e

        print("-" * 60)
        print(f"🚀 ACTIVE SYSTEM: {self.llm_manager.get_provider_name().upper()}")
        print(f"📁 PROMPTS: {self.prompt_manager.get_provider()}")
        print("-" * 60)
        
        # Initialize agents with the LLM manager and PromptManager
        self.splitter_agent = ClauseSplitterAgent(self.llm_manager, self.prompt_manager)
        self.classifier_agent = ClauseClassifierAgent(self.llm_manager, self.prompt_manager)
        self.risk_detector_agent = RiskDetectorAgent(self.llm_manager, self.prompt_manager)

        print("\n✅ LegalDocAI initialized successfully\n")

    def analyze_document(self, document_text: str, save_output: bool = True) -> Dict[str, Any]:
        """
        Analyze a legal document through the full pipeline.
        
        Args:
            document_text: The full text of the legal document
            save_output: Whether to save results to JSON file
        
        Returns:
            Dictionary containing all analysis results
        """
        print("\n" + "=" * 60)
        print("ANALYZING DOCUMENT")
        print("=" * 60)

        # Step 1: Split document into clauses
        print("📄 Step 1: Splitting document into clauses...", end=" ", flush=True)
        clauses = self.splitter_agent.split_document(document_text)
        print(f"✅ Found {len(clauses)} clauses")

        # Step 2: Classify clauses
        print(f"🏷️  Step 2: Classifying {len(clauses)} clauses...", end=" ", flush=True)
        classifications = self.classifier_agent.classify_multiple_clauses(clauses)
        print("✅ Done")

        # Step 3: Assess risks
        print(f"🔍 Step 3: Assessing risks for {len(clauses)} clauses...", end=" ", flush=True)
        risk_assessments = self.risk_detector_agent.detect_risks_multiple_clauses(
            clauses, classifications
        )
        print("✅ Done")
        
        # Categorize risks by level
        high_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'HIGH']
        medium_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'MEDIUM']
        low_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'LOW']

        # Step 4: Compile results
        print("📊 Step 4: Compiling results...", end=" ", flush=True)
        results = {
            "total_clauses": len(clauses),
            "clauses": clauses,
            "classifications": classifications,
            "risk_assessments": risk_assessments,
            "high_risk_clauses": high_risk_clauses,
            "medium_risk_clauses": medium_risk_clauses,
            "low_risk_clauses": low_risk_clauses,
            "provider_used": self.llm_manager.get_provider_name()
        }
        print("✅ Done")
        
        # Save only the final analysis file
        if save_output:
            output_file = "nda_analysis_report.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Report saved: {output_file}")
        
        return results

    def print_summary(self, results: Dict[str, Any]):
        """
        Print a formatted summary of the analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        print("\n" + "=" * 60)
        print("DOCUMENT ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📊 Total clauses analyzed: {results['total_clauses']}")
        print(f"🤖 Provider used: {results.get('provider_used', 'unknown').upper()}")
        
        # Count risk levels
        risk_counts = {
            "HIGH": len(results.get('high_risk_clauses', [])),
            "MEDIUM": len(results.get('medium_risk_clauses', [])),
            "LOW": len(results.get('low_risk_clauses', [])),
            "NONE": 0,
            "UNKNOWN": 0
        }
        
        # Count remaining clauses
        for risk in results.get('risk_assessments', []):
            level = risk.get('risk_level', 'UNKNOWN')
            if level not in ["HIGH", "MEDIUM", "LOW"]:
                risk_counts[level] = risk_counts.get(level, 0) + 1

        print(f"\n📈 Risk Distribution:")
        print(f"   🔴 High Risk: {risk_counts['HIGH']}")
        print(f"   🟡 Medium Risk: {risk_counts['MEDIUM']}")
        print(f"   🟢 Low Risk: {risk_counts['LOW']}")
        print(f"   ⚪ No Risk: {risk_counts['NONE']}")
        
        # Display high risk clauses
        if risk_counts['HIGH'] > 0:
            print("\n" + "=" * 60)
            print("⚠️  DETAILED HIGH RISK CLAUSES")
            print("=" * 60)
        
            for i, risk_assessment in enumerate(results.get('high_risk_clauses', [])):
                clause_id = risk_assessment.get('clause_id')
                original_clause = risk_assessment.get('original_clause', {})
                
                # Clause header
                print(f"\n{i+1}. {original_clause.get('clause_title', 'Untitled')} (ID: {clause_id})")
                print("-" * 40)
                
                # Risk information
                print(f"   Risk Score: {risk_assessment.get('risk_score', 'N/A')}")
                print(f"   Assessment: {risk_assessment.get('overall_assessment', 'No assessment')}")
                
                # Original clause text (truncated)
                clause_text = original_clause.get('clause_text', 'No text available')
                if len(clause_text) > 200:
                    clause_text = clause_text[:200] + "..."
                print(f"\n   Original Text:\n   {clause_text}")
                
                # Identified risks
                risks = risk_assessment.get('identified_risks', [])
                if risks:
                    print("\n   Identified Risks:")
                    for risk in risks:
                        print(f"     • {risk.get('risk_type', 'Unknown')} ({risk.get('severity', 'Unknown')})")
                        print(f"       {risk.get('description', 'No description')}")
                
                # Recommendations
                recommendations = risk_assessment.get('recommendations', [])
                if recommendations:
                    print("\n   Recommendations:")
                    for rec in recommendations:
                        print(f"     • {rec}")
                
                print("-" * 40)
        
        # Medium risk summary
        if risk_counts['MEDIUM'] > 0:
            print("\n🟡 Medium Risk Clauses to Review:")
            for risk_assessment in results.get('medium_risk_clauses', []):
                original_clause = risk_assessment.get('original_clause', {})
                assessment = risk_assessment.get('overall_assessment', '')[:80]
                print(f"   • {original_clause.get('clause_title', 'Untitled')}: {assessment}...")
        
        print("\n" + "=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LegalDoc AI - NDA Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py contract.txt              # Analyze a contract
    python main.py nda.txt --no-save         # Analyze without saving JSON files
    python main.py document.pdf              # Note: PDF support requires pypdf2
        """
    )
    
    parser.add_argument(
        "document",
        type=str,
        help="Path to the document file to analyze"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output JSON files"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the LegalDoc AI CLI."""
    print("=" * 60)
    print("LegalDoc AI - NDA Analysis System")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if document exists
    if not os.path.exists(args.document):
        print(f"\n❌ Error: Document not found: {args.document}")
        sys.exit(1)
    
    # Read the document
    try:
        with open(args.document, "r", encoding="utf-8") as f:
            document_text = f.read()
    except UnicodeDecodeError:
        print(f"\n❌ Error: Unable to read file. Make sure it's a text file.")
        print("   For PDF files, convert to text first or use a PDF library.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        sys.exit(1)
    
    print(f"\n📄 Document loaded: {args.document}")
    print(f"   Size: {len(document_text)} characters")
    
    # Initialize and run analysis
    try:
        legal_ai = LegalDocAI()
        save_output = not args.no_save
        results = legal_ai.analyze_document(document_text, save_output=save_output)
        legal_ai.print_summary(results)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
