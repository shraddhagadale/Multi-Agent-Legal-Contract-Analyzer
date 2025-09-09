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
    def __init__(self, api_key = None):
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable")

        #Initialize the language model
        self.llm = ChatOpenAI( 
            model = "gpt-4o-mini",
            temperature = 0.1,
            api_key = api_key
        )

        #Initialize the agents
        self.splitter_agent = ClauseSplitterAgent(self.llm)
        self.classifier_agent = ClauseClassifierAgent(self.llm)
        self.risk_detector_agent = RiskDetectorAgent(self.llm)

        print("LegalDocAI initialized successfully")

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
        high_risk_clauses = self.risk_detector_agent.get_high_risk_clauses(risk_assessments)
        print(f"Found {len(high_risk_clauses)} high risk clauses")
        
        if save_output:
            with open("output_risk_assessments.json", "w") as f:
                json.dump(risk_assessments, f, indent=2)
            print("Risk assessments saved to output_risk_assessments.json")

        # Step 4
        print("Step 4: Compiling results...")
        results = {
            "total_clauses": len(clauses),
            "clauses": clauses,
            "classifications": classifications,
            "risk_assessments": risk_assessments,
            "high_risk_clauses": high_risk_clauses
        }
        
        if save_output:
            with open("output_all_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("All results saved to output_all_results.json")

        print("Analysis complete!")
        return results
    
    def print_summary(self, results):
        print("\n" + "="*60)
        print("Document Analysis Summary")
        print("="*60)

        print(f"Total clauses : {results['total_clauses']}")

        # Print clauses for debugging
        print("\nClauses:")
        for i, clause in enumerate(results.get('clauses', [])):
            print(f"{i+1}. {clause.get('clause_title', 'Untitled')}: {clause.get('clause_text', '')[:50]}...")

        # Print classifications
        print("\nClassifications:")
        for i, classification in enumerate(results.get('classifications', [])):
            print(f"{i+1}. {classification.get('category', 'Unknown')} - {classification.get('subcategory', 'N/A')}")
            print(f"   Confidence: {classification.get('confidence', 'N/A')}")
            print(f"   Reasoning: {classification.get('reasoning', 'No reasoning provided')[:100]}...")
       
        # Count risk levels
        risk_levels = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0, "UNKNOWN": 0}
        for risk in results.get('risk_assessments', []):
            level = risk.get('risk_level', 'UNKNOWN')
            risk_levels[level] = risk_levels.get(level,0)+1

        print(f"High Risk Clauses: {risk_levels['HIGH']}")
        print(f"Medium Risk Clauses: {risk_levels['MEDIUM']}")
        print(f"Low Risk Clauses: {risk_levels['LOW']}")
        print(f"No Risk Clauses: {risk_levels['NONE']}")

        if risk_levels['HIGH'] > 0:
            print("\n" + "="*80)
            print("DETAILED HIGH RISK CLAUSES ANALYSIS")
            print("="*80)
            
            for i, clause in enumerate(results.get('high_risk_clauses', [])):
                # Get the original clause from the risk assessment
                original_clause = clause.get('original_clause', {})
                
                # If original_clause is not found, try to find it in the clauses list
                if not original_clause:
                    clause_id = clause.get('clause_id')
                    for c in results.get('clauses', []):
                        if c.get('clause_id') == clause_id:
                            original_clause = c
                            break
                
                # 1. Clause Name/Title
                print(f"\n{i+1}. {original_clause.get('clause_title','Untitled')} (ID: {clause.get('clause_id', 'Unknown')})")
                
                # 2. Original Clause Text
                clause_text = original_clause.get('clause_text', 'No text available')
                print("\n   Original Clause Text:")
                print(f"     {clause_text}")
                
                # 3. Recommendations
                recommendations = clause.get('recommendations', [])
                if recommendations:
                    print("\n   Recommendations:")
                    for rec in recommendations:
                        print(f"     - {rec}")
                else:
                    print("\n   Recommendations: None provided")
                    
                print("-" * 80)
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










