"""
LegalDoc AI - Agent-by-Agent Debug Runner

Runs each agent independently so you can inspect each step's output,
identify exactly where quality breaks down, and iterate on individual
agents without running the full pipeline.

Usage:
    python debug_runner.py <document_file> --agent <agent_name> [--output json|text]
    python debug_runner.py <document_file> --all [--save]

Agents:
    analyzer   — Run DocumentAnalyzerAgent only
    splitter   — Run ClauseSplitterAgent only
    classifier — Run ClauseClassifierAgent (requires splitter output)
    risk       — Run RiskDetectorAgent (requires splitter + classifier output)
    all        — Run full pipeline with verbose output at each step
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

from dotenv import load_dotenv

from legaldoc.utils import LLMProviderManager
from legaldoc.agents import (
    DocumentAnalyzerAgent,
    ClauseSplitterAgent,
    ClauseClassifierAgent,
    RiskDetectorAgent,
)

# Load environment variables
load_dotenv()


# ── Terminal Colors ──────────────────────────────────────────────────────────

class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


RISK_COLORS = {
    "HIGH": Colors.RED,
    "MEDIUM": Colors.YELLOW,
    "LOW": Colors.GREEN,
    "NONE": Colors.GRAY,
}


def colored_risk(level: str) -> str:
    """Return a color-coded risk level string."""
    color = RISK_COLORS.get(level, Colors.RESET)
    return f"{color}{Colors.BOLD}{level}{Colors.RESET}"


def header(title: str) -> None:
    """Print a section header."""
    width = 70
    print(f"\n{Colors.CYAN}{'═' * width}")
    print(f"  {Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * width}{Colors.RESET}\n")


def subheader(title: str) -> None:
    """Print a sub-section header."""
    print(f"\n{Colors.BOLD}── {title} ──{Colors.RESET}\n")


# ── Agent Runners ────────────────────────────────────────────────────────────

def init_system():
    """Initialize the LLM manager and all agents."""
    llm_manager = LLMProviderManager()
    agents = {
        "analyzer": DocumentAnalyzerAgent(llm_manager),
        "splitter": ClauseSplitterAgent(llm_manager),
        "classifier": ClauseClassifierAgent(llm_manager),
        "risk": RiskDetectorAgent(llm_manager),
    }
    return llm_manager, agents


def run_analyzer(agents: dict, document_text: str) -> Dict[str, Any]:
    """Run the DocumentAnalyzerAgent and print results."""
    header("Agent: DocumentAnalyzerAgent")

    start = time.time()
    result = agents["analyzer"].analyze_document(document_text)
    elapsed = time.time() - start

    print(f"  Document Type:      {result.get('document_type', 'N/A')}")
    print(f"  Effective Date:     {result.get('effective_date', 'N/A')}")

    parties = result.get("parties", [])
    if parties:
        print(f"  Parties:")
        for p in parties:
            name = p.get("name", "Unknown") if isinstance(p, dict) else str(p)
            role = p.get("role", "") if isinstance(p, dict) else ""
            print(f"    - {name} ({role})" if role else f"    - {name}")

    print(f"\n  Summary:\n    {result.get('summary', 'N/A')}")

    observations = result.get("key_observations", [])
    if observations:
        print(f"\n  Key Observations:")
        for obs in observations:
            print(f"    • {obs}")

    print(f"\n  Formatted Summary (passed to downstream agents):")
    for line in result.get("formatted_summary", "").split("\n"):
        print(f"    {line}")

    print(f"\n  ⏱  Completed in {elapsed:.2f}s")
    return result


def run_splitter(agents: dict, document_text: str) -> List[Dict[str, Any]]:
    """Run the ClauseSplitterAgent and print results."""
    header("Agent: ClauseSplitterAgent")

    start = time.time()
    clauses = agents["splitter"].split_document(document_text)
    elapsed = time.time() - start

    print(f"  Total clauses extracted: {len(clauses)}\n")

    for c in clauses:
        cid = c.get("clause_id", "?")
        cnum = c.get("clause_number", "?")
        ctitle = c.get("clause_title", "Untitled")
        ctext = c.get("clause_text", "")

        print(f"  [{cid}] Clause {cnum}: {ctitle}")
        # Show first 200 chars of text
        preview = ctext[:200].replace("\n", " ")
        if len(ctext) > 200:
            preview += "..."
        print(f"    Text: {preview}")
        print()

    print(f"  ⏱  Completed in {elapsed:.2f}s")
    return clauses


def run_classifier(
    agents: dict, document_text: str, clauses: List[Dict] = None, doc_summary: str = None
) -> List[Dict[str, Any]]:
    """Run the ClauseClassifierAgent and print results."""
    header("Agent: ClauseClassifierAgent")

    # Run splitter first if clauses not provided
    if clauses is None:
        subheader("Pre-step: Running Splitter")
        clauses = agents["splitter"].split_document(document_text)
        print(f"  Got {len(clauses)} clauses from splitter")

    # Run analyzer for summary if not provided
    if doc_summary is None:
        subheader("Pre-step: Running Analyzer")
        context = agents["analyzer"].analyze_document(document_text)
        doc_summary = context.get("formatted_summary", "No context available.")
        print(f"  Got document summary")

    subheader("Classification Results")

    start = time.time()
    classifications = agents["classifier"].classify_multiple_clauses(
        clauses, document_summary=doc_summary
    )
    elapsed = time.time() - start

    # Sort by confidence (lowest first to spot problem classifications)
    sorted_cls = sorted(classifications, key=lambda x: x.get("confidence", 0))

    for cl in sorted_cls:
        cid = cl.get("clause_id", "?")
        cat = cl.get("category", "Unknown")
        subcat = cl.get("subcategory", "")
        conf = cl.get("confidence", 0)
        reasoning = cl.get("reasoning", "N/A")

        # Color-code confidence
        if conf < 0.5:
            conf_color = Colors.RED
        elif conf < 0.75:
            conf_color = Colors.YELLOW
        else:
            conf_color = Colors.GREEN

        print(f"  [{cid}] {cat}")
        print(f"    Subcategory: {subcat}")
        print(f"    Confidence:  {conf_color}{conf:.2f}{Colors.RESET}")
        reasoning_preview = reasoning[:120]
        if len(reasoning) > 120:
            reasoning_preview += "..."
        print(f"    Reasoning:   {reasoning_preview}")
        print()

    print(f"  ⏱  Completed in {elapsed:.2f}s")
    return classifications


def run_risk_detector(
    agents: dict,
    document_text: str,
    clauses: List[Dict] = None,
    classifications: List[Dict] = None,
    doc_summary: str = None,
) -> List[Dict[str, Any]]:
    """Run the RiskDetectorAgent and print results."""
    header("Agent: RiskDetectorAgent")

    # Run prerequisite agents if needed
    if clauses is None:
        subheader("Pre-step: Running Splitter")
        clauses = agents["splitter"].split_document(document_text)
        print(f"  Got {len(clauses)} clauses from splitter")

    if doc_summary is None:
        subheader("Pre-step: Running Analyzer")
        context = agents["analyzer"].analyze_document(document_text)
        doc_summary = context.get("formatted_summary", "No context available.")
        print(f"  Got document summary")

    if classifications is None:
        subheader("Pre-step: Running Classifier")
        classifications = agents["classifier"].classify_multiple_clauses(
            clauses, document_summary=doc_summary
        )
        print(f"  Got {len(classifications)} classifications")

    subheader("Risk Assessment Results")

    start = time.time()
    risk_assessments = agents["risk"].detect_risks_multiple_clauses(
        clauses, classifications, document_summary=doc_summary
    )
    elapsed = time.time() - start

    # Group by risk level (HIGH first)
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    sorted_risks = sorted(
        risk_assessments,
        key=lambda r: risk_order.get(r.get("risk_level", "NONE"), 4),
    )

    # Counts
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for r in risk_assessments:
        level = r.get("risk_level", "NONE")
        counts[level] = counts.get(level, 0) + 1

    print(f"  Risk Summary: "
          f"{colored_risk('HIGH')}: {counts['HIGH']}  "
          f"{colored_risk('MEDIUM')}: {counts['MEDIUM']}  "
          f"{colored_risk('LOW')}: {counts['LOW']}  "
          f"{colored_risk('NONE')}: {counts['NONE']}\n")

    # Build a clause text lookup for displaying alongside risks
    clause_lookup = {c.get("clause_id"): c for c in clauses}

    for r in sorted_risks:
        cid = r.get("clause_id", "?")
        level = r.get("risk_level", "NONE")
        score = r.get("risk_score", 0)
        risks = r.get("identified_risks", [])
        recs = r.get("recommendations", [])
        assessment = r.get("overall_assessment", "")

        print(f"  [{cid}] {colored_risk(level)} (score: {score:.2f})")

        # Show clause text preview
        clause_data = clause_lookup.get(cid, {})
        ctext = clause_data.get("clause_text", "")
        ctitle = clause_data.get("clause_title", "")
        if ctitle:
            print(f"    Title: {ctitle}")
        if ctext:
            preview = ctext[:150].replace("\n", " ")
            if len(ctext) > 150:
                preview += "..."
            print(f"    Clause: {preview}")

        if risks:
            print(f"    Identified Risks:")
            for risk in risks:
                rtype = risk.get("risk_type", "Unknown")
                desc = risk.get("description", "")
                sev = risk.get("severity", "")
                desc_preview = desc[:100]
                if len(desc) > 100:
                    desc_preview += "..."
                print(f"      ⚠  [{sev}] {rtype}: {desc_preview}")

        if recs:
            print(f"    Recommendations:")
            for rec in recs:
                rec_preview = rec[:100]
                if len(rec) > 100:
                    rec_preview += "..."
                print(f"      → {rec_preview}")

        if assessment:
            assessment_preview = assessment[:150]
            if len(assessment) > 150:
                assessment_preview += "..."
            print(f"    Assessment: {assessment_preview}")

        # Show sub-clause breakdown if available
        sub_results = r.get("sub_clause_results")
        if sub_results and len(sub_results) > 1:
            print(f"    Sub-clause breakdown:")
            for sr in sub_results:
                sub_id = sr.get("clause_id", "?")
                sub_level = sr.get("risk_level", "NONE")
                sub_score = sr.get("risk_score", 0)
                sub_risks_count = len(sr.get("identified_risks", []))
                print(f"      {sub_id}: {colored_risk(sub_level)} (score: {sub_score:.2f}, {sub_risks_count} risks)")

        print()

    print(f"  ⏱  Completed in {elapsed:.2f}s")
    return risk_assessments


def run_all(agents: dict, document_text: str, save_path: str = None) -> Dict[str, Any]:
    """Run the full pipeline with verbose output at each step."""
    header("FULL PIPELINE — All Agents")
    total_start = time.time()

    all_results = {}

    # Step 1: Analyzer
    analyzer_result = run_analyzer(agents, document_text)
    all_results["analyzer"] = analyzer_result
    doc_summary = analyzer_result.get("formatted_summary", "No context available.")

    # Step 2: Splitter
    clauses = run_splitter(agents, document_text)
    all_results["splitter"] = {"total_clauses": len(clauses), "clauses": clauses}

    # Step 3: Classifier
    classifications = run_classifier(agents, document_text, clauses=clauses, doc_summary=doc_summary)
    all_results["classifier"] = classifications

    # Step 4: Risk Detector
    risk_assessments = run_risk_detector(
        agents, document_text, clauses=clauses, classifications=classifications, doc_summary=doc_summary
    )
    all_results["risk_detector"] = risk_assessments

    total_elapsed = time.time() - total_start

    header("PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Clauses:    {len(clauses)}")

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for r in risk_assessments:
        level = r.get("risk_level", "NONE")
        counts[level] = counts.get(level, 0) + 1

    print(f"  Risks:      "
          f"{colored_risk('HIGH')}: {counts['HIGH']}  "
          f"{colored_risk('MEDIUM')}: {counts['MEDIUM']}  "
          f"{colored_risk('LOW')}: {counts['LOW']}  "
          f"{colored_risk('NONE')}: {counts['NONE']}")

    # Save to JSON if requested
    if save_path:
        # Make results JSON-serializable (strip non-serializable objects)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {save_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LegalDoc AI — Agent-by-Agent Debug Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python debug_runner.py files/mutual_risky.txt --agent analyzer
    python debug_runner.py files/mutual_risky.txt --agent risk
    python debug_runner.py files/mutual_risky.txt --all
    python debug_runner.py files/mutual_risky.txt --all --save
        """,
    )

    parser.add_argument(
        "document",
        type=str,
        help="Path to the document file to analyze",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--agent",
        type=str,
        choices=["analyzer", "splitter", "classifier", "risk"],
        help="Run a specific agent (analyzer, splitter, classifier, risk)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all agents in sequence with verbose output",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "text"],
        default="text",
        help="Output format: json (raw) or text (formatted, default)",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save all results to files/<docname>_debug.json (only with --all)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Resolve document path
    FILES_DIR = "files"
    doc_path = args.document
    if not os.path.exists(doc_path):
        potential_path = os.path.join(FILES_DIR, doc_path)
        if os.path.exists(potential_path):
            doc_path = potential_path
        else:
            sys.exit(f"Error: Document not found: {args.document} (checked root and {FILES_DIR}/)")

    # Read document
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            document_text = f.read()
    except Exception as e:
        sys.exit(f"Error reading file: {e}")

    print(f"{Colors.BOLD}LegalDoc AI — Debug Runner{Colors.RESET}")
    print(f"Document: {doc_path} ({len(document_text)} chars)\n")

    # Initialize system
    try:
        llm_manager, agents = init_system()
        print(f"Provider: {llm_manager.get_provider_name()}\n")
    except Exception as e:
        sys.exit(f"Failed to initialize: {e}")

    # Route to appropriate runner
    if args.all:
        save_path = None
        if args.save:
            base_name = os.path.splitext(os.path.basename(doc_path))[0]
            save_path = os.path.join(FILES_DIR, f"{base_name}_debug.json")

        result = run_all(agents, document_text, save_path=save_path)

        if args.output == "json":
            print("\n" + json.dumps(result, indent=2, default=str))

    elif args.agent == "analyzer":
        result = run_analyzer(agents, document_text)
        if args.output == "json":
            print("\n" + json.dumps(result, indent=2, default=str))

    elif args.agent == "splitter":
        result = run_splitter(agents, document_text)
        if args.output == "json":
            print("\n" + json.dumps(result, indent=2, default=str))

    elif args.agent == "classifier":
        result = run_classifier(agents, document_text)
        if args.output == "json":
            print("\n" + json.dumps(result, indent=2, default=str))

    elif args.agent == "risk":
        result = run_risk_detector(agents, document_text)
        if args.output == "json":
            print("\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
