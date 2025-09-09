## Multi-Agent Legal Contract Analyzer

The Multi-Agent Legal Contract Analyzer is an AI-powered system that analyzes Non-Disclosure Agreements (NDAs) to identify risky clauses, provide risk assessments, and suggest improvements.

## Overview

The system uses a pipeline of specialized agents to process and analyze NDAs:

1. **Clause Splitter Agent**: Breaks down the NDA document into logical clauses
2. **Clause Classifier Agent**: Categorizes each clause (e.g., Confidentiality, Term, Obligations)
3. **Risk Detector Agent**: Identifies potentially risky clauses and provides detailed risk assessments
4. **Rewriter Agent** (Coming Soon): Suggests safer alternatives for high-risk clauses

## Features

- Automated clause identification and extraction
- Detailed classification of clauses by type and category
- Risk assessment with severity scoring
- Identification of high-risk clauses with detailed explanations
- Actionable recommendations for improving problematic clauses
- JSON output for further processing or integration

## Project Structure

```
LegalDoc/
  - agents/
    - splitter_agent.py      # Clause Splitter Agent implementation
    - classifier_agent.py    # Clause Classifier Agent implementation
    - risk_detector_agent.py # Risk Detector Agent implementation
  - prompts/
    - splitter_prompt.txt    # Prompt template for Clause Splitter
    - classifier_prompt.txt  # Prompt template for Clause Classifier
    - risk_detector_prompt.txt # Prompt template for Risk Detector
  - utils/
    - load_env.py            # Utility for loading environment variables
  - main.py                  # Main orchestrator and pipeline
  - requirements.txt         # Project dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/shraddhagadale/Multi-Agent-Legal-Contract-Analyzer
cd Multi-Agent-Legal-Contract-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the full pipeline

```python
from main import LegalDocAI

# Initialize the system
analyzer = LegalDocAI()

# Analyze an NDA document
with open("your_nda_document.txt", "r") as f:
    document_text = f.read()

# Run the analysis
results = analyzer.analyze_document(document_text)

# Print a summary of the results
analyzer.print_summary(results)
```

### Sample Output

The system will generate a detailed analysis including:
- Total number of clauses identified
- Classification of each clause
- Risk assessment for each clause
- Detailed analysis of high-risk clauses with recommendations

## Current Status

This is the first draft of the LegalDoc AI system. Currently implemented:
- Clause Splitter Agent
- Clause Classifier Agent
- Risk Detector Agent

Coming soon:
- Support for PDF and DOCX documents
- Enhanced risk scoring and recommendations

## Dependencies

- Python 3.8+
- OpenAI API
- CrewAI
- LangChain
- Other dependencies listed in requirements.txt


