import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from agents import splitter_agent
from agents.splitter_agent import ClauseSplitterAgent
from agents.classifier_agent import ClauseClassifierAgent
from agents.risk_detector_agent import RiskDetectorAgent

class LegalDocAI:
    def __init__(self):
        """
        Initialize LegalDocAI with Auto-Fallback Support.
        
        Uses LLMProviderManager to:
        1. Try starting OpenAI (GPT-4o-mini)
        2. Test the connection
        3. Fallback to Gemini if OpenAI fails
        4. Load the correct prompts for the active provider
        """
        print("="*60)
        print("INITIALIZING LEGAL DOC AI SYSTEM")
        print("="*60)
        
        # Initialize LLM with Auto-Fallback
        try:
            from utils.llm_provider_manager import LLMProviderManager
            
            # This handles priority, validation, and fallback automatically
            # Configuration is loaded internally from utils.load_env
            self.llm_manager = LLMProviderManager()
            self.llm = self.llm_manager.get_llm()
            
        except Exception as e:
            print(f"\n❌ FATAL ERROR During Initialization: {e}")
            raise e

        # Initialize Prompt Manager
        # It detects the provider from the validated LLM instance
        from utils.prompt_manager import PromptManager
        self.prompt_manager = PromptManager(self.llm)
        
        print("-" * 60)
        print(f"🚀 ACTIVE SYSTEM: {self.prompt_manager.get_provider().upper()}")
        print("-" * 60)
        
        # Initialize the agents with the active LLM and PromptManager
        self.splitter_agent = ClauseSplitterAgent(self.llm, self.prompt_manager)
        self.classifier_agent = ClauseClassifierAgent(self.llm, self.prompt_manager)
        self.risk_detector_agent = RiskDetectorAgent(self.llm, self.prompt_manager)

        print("LegalDocAI initialized successfully\n")


    def analyze_document(self, document_text: str, save_output=True):
        print("Starting NDA Analysis")

        # Step 1
        print("Step 1: Splitting document into clauses...")
        clauses = self.splitter_agent.split_document(document_text)
        print(f"Found {len(clauses)} clauses")
        
        if save_output:
            import json
            with open("output_clauses.json", "w") as f:
                json.dump(clauses, f, indent=2)
            print("Clauses saved to output_clauses.json")

        # Step 2 
        print("Step 2: Classifying clauses...")
        classifications = self.classifier_agent.classify_multiple_clauses(clauses)
        print(f"Classified {len(classifications)} clauses")
        
        if save_output:
            with open("output_classifications.json", "w") as f:
                json.dump(classifications, f, indent=2)
            print("Classifications saved to output_classifications.json")

        # Step 3
        print("Step 3: Assessing risks...")
        risk_assessments = self.risk_detector_agent.detect_risks_multiple_clauses(clauses, classifications)
        
        # Categorize risks by level
        high_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'HIGH']
        medium_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'MEDIUM']
        low_risk_clauses = [r for r in risk_assessments if r.get('risk_level') == 'LOW']

        print(f"Found {len(high_risk_clauses)} high risk clauses")
        print(f"Found {len(medium_risk_clauses)} medium risk clauses")
        print(f"Found {len(low_risk_clauses)} low risk clauses")
        
        if save_output:
            with open("output_risk_assessments.json", "w") as f:
                json.dump(risk_assessments, f, indent=2)
            print("Risk assessments saved to output_risk_assessments.json")

        # Step 4: Compile results
        print("Step 4: Compiling results...")
        results = {
            "total_clauses": len(clauses),
            "clauses": clauses,
            "classifications": classifications,
            "risk_assessments": risk_assessments,
            "high_risk_clauses": high_risk_clauses,
            "medium_risk_clauses": medium_risk_clauses,
            "low_risk_clauses": low_risk_clauses
        }
        
        if save_output:
            with open("output_all_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("All results saved to output_all_results.json")
        
        print("Analysis complete!")
        return results

    def print_summary(self, results):
        print("\n" + "="*60)
        print("DOCUMENT ANALYSIS SUMMARY")
        print("="*60)

        print(f"Total clauses: {results['total_clauses']}")
        
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
        
            print(f"\nRisk Distribution:")
            print(f"High Risk Clauses: {risk_counts['HIGH']}")
            print(f"Medium Risk Clauses: {risk_counts['MEDIUM']}")
            print(f"Low Risk Clauses: {risk_counts['LOW']}")
            print(f"No Risk Clauses: {risk_counts['NONE']}")
            
            # Display high risk clauses
            if risk_counts['HIGH'] > 0:
                print("\n" + "="*80)
                print("DETAILED HIGH RISK CLAUSES ANALYSIS")
                print("="*80)
            
            for i, risk_assessment in enumerate(results.get('high_risk_clauses', [])):
                clause_id = risk_assessment.get('clause_id')
                original_clause = risk_assessment.get('original_clause', {})
                
                # 1. Clause Title
                print(f"\n{i+1}. {original_clause.get('clause_title', 'Untitled')} (ID: {clause_id})")
                
                # 2. Risk information
                print(f"   Risk Score: {risk_assessment.get('risk_score', 'N/A')}")
                print(f"   Overall Assessment: {risk_assessment.get('overall_assessment', 'No assessment provided')}")
                
                # 3. Original Clause Text
                clause_text = original_clause.get('clause_text', 'No text available')
                print("\n   Original Clause Text:")
                print(f"     {clause_text}")
                
                # 4. Identified Risks
                risks = risk_assessment.get('identified_risks', [])
                if risks:
                    print("\n   Identified Risks:")
                    for risk in risks:
                        print(f"     - {risk.get('risk_type', 'Unknown')} (Severity: {risk.get('severity', 'Unknown')})")
                        print(f"       {risk.get('description', 'No description')}")
                
                # 5. Recommendations
                recommendations = risk_assessment.get('recommendations', [])
                if recommendations:
                    print("\n   Recommendations:")
                    for rec in recommendations:
                        print(f"     - {rec}")
                else:
                    print("\n   Recommendations: None provided")
                    
                print("-" * 80)
        
        # Important medium risk clauses that might need attention
        if risk_counts['MEDIUM'] > 0:
            print("\nMedium Risk Clauses to Review:")
            for i, risk_assessment in enumerate(results.get('medium_risk_clauses', [])):
                original_clause = risk_assessment.get('original_clause', {})
                print(f"  • {original_clause.get('clause_title', 'Untitled')} - {risk_assessment.get('overall_assessment', '')[:100]}...")
        
        print("\n" + "="*60)
   

# Test Function
def test_with_sample_nda(save_output=True):
    # Read the sample NDA document with risks
    with open("sample_nda_with_risks.txt", "r") as f:
        document_text = f.read()
    
    print(f"Document loaded, length: {len(document_text)} characters")
    
    legal_ai = LegalDocAI()
    results = legal_ai.analyze_document(document_text, save_output=save_output)

    legal_ai.print_summary(results)

    return results

if __name__ == "__main__":
    print("LegalDoc AI - NDA Analysis System")
    print("-----------------------------------------------------------")
    print("Running full analysis with sample NDA document...")
    test_with_sample_nda()










