# Multi-Agent Legal Contract Analyzer

A multi-agent AI system for analyzing legal documents, with a focus on Non-Disclosure Agreements (NDAs) and other contract types. The system automatically extracts clauses, classifies them by category, detects potential risks, and generates comprehensive PDF reports with actionable recommendations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Supported Document Types](#supported-document-types)
- [Output](#output)
- [Development](#development)
- [License](#license)

## Overview

The Multi-Agent Legal Contract Analyzer uses a pipeline of specialized AI agents to process and analyze legal documents:

1. **Clause Splitter Agent**: Parses the document and breaks it down into individual logical clauses
2. **Clause Classifier Agent**: Categorizes each clause by type (e.g., Confidentiality, Term, Obligations, Indemnification)
3. **Risk Detector Agent**: Evaluates each clause for potential legal risks, assigns severity levels, and provides recommendations for safer alternatives

The system supports multiple LLM providers with automatic fallback capabilities, ensuring reliability and flexibility.

## Features

- **Automated Clause Extraction**: Intelligently identifies and extracts individual clauses from legal documents
- **Multi-Category Classification**: Classifies clauses into standard legal categories
- **Risk Assessment**: Assigns risk levels (HIGH, MEDIUM, LOW) with detailed explanations
- **Actionable Recommendations**: Provides specific suggestions for improving problematic clauses
- **PDF Report Generation**: Creates professional, comprehensive analysis reports
- **Multi-Provider LLM Support**: Works with OpenAI and Google Gemini with automatic fallback
- **Multiple File Format Support**: Accepts `.txt`, `.md`, and other text-based formats
- **Command-Line Interface**: Simple CLI for easy integration into workflows

## Architecture

```
                    +-------------------+
                    |    Document       |
                    |    Input          |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |  Clause Splitter  |
                    |      Agent        |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    | Clause Classifier |
                    |      Agent        |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |   Risk Detector   |
                    |      Agent        |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |   PDF Report      |
                    |   Generator       |
                    +-------------------+
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

1. Clone the repository:

```bash
git clone https://github.com/shraddhagadale/Multi-Agent-Legal-Contract-Analyzer.git
cd Multi-Agent-Legal-Contract-Analyzer
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package and dependencies:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```env
# Required: At least one API key must be configured
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Model configuration
OPENAI_MODEL_NAME=gpt-4
GOOGLE_MODEL_NAME=gemini-pro
```

The system uses OpenAI as the primary provider and Google Gemini as a fallback. If the primary provider fails or is not configured, it automatically switches to the fallback.

## Usage

### Command Line Interface

Analyze a document and generate a PDF report:

```bash
python main.py path/to/document.txt
```

Specify a custom output file name:

```bash
python main.py path/to/document.txt -o custom_report.pdf
```

View help and available options:

```bash
python main.py --help
```

### Programmatic Usage

```python
from main import LegalDocAI

# Initialize the analyzer
analyzer = LegalDocAI()

# Read your document
with open("your_document.txt", "r") as f:
    document_text = f.read()

# Run the analysis
results = analyzer.analyze_document(document_text)

# Generate a PDF report
report_path = analyzer.generate_report(
    results,
    source_file="your_document.txt",
    output_path="analysis_report.pdf"
)

print(f"Report saved to: {report_path}")
```

### Analysis Results Structure

The `analyze_document` method returns a dictionary containing:

| Key | Description |
|-----|-------------|
| `total_clauses` | Total number of clauses identified |
| `clauses` | List of extracted clause texts |
| `classifications` | Classification results for each clause |
| `risk_assessments` | Risk analysis for each clause |
| `high_risk_clauses` | Clauses flagged as high risk |
| `medium_risk_clauses` | Clauses flagged as medium risk |
| `low_risk_clauses` | Clauses flagged as low risk |
| `provider_used` | LLM provider used for analysis |

## Project Structure

```
Multi-Agent-Legal-Contract-Analyzer/
├── main.py                      # CLI entry point and orchestrator
├── pyproject.toml               # Project configuration and dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project documentation
├── files/                       # Input/output directory for documents
└── src/
    └── legaldoc/
        ├── __init__.py
        ├── agents/              # AI agent implementations
        │   ├── __init__.py
        │   ├── base_agent.py           # Base agent class
        │   ├── splitter_agent.py       # Clause extraction agent
        │   ├── classifier_agent.py     # Clause classification agent
        │   └── risk_detector_agent.py  # Risk assessment agent
        ├── prompts/             # Agent prompt templates
        │   ├── splitter_prompt.txt
        │   ├── classifier_prompt.txt
        │   └── risk_detector_prompt.txt
        └── utils/               # Utility modules
            ├── __init__.py
            ├── llm_provider_manager.py  # LLM provider abstraction
            ├── pdf_generator.py         # PDF report generation
            ├── schemas.py               # Pydantic data models
            └── load_env.py              # Environment configuration
```

## Supported Document Types

The analyzer is designed to work with various legal documents:

- Non-Disclosure Agreements (NDAs)
- Employment Agreements
- Consulting Agreements
- Master Service Agreements (MSAs)
- Software License Agreements
- Data Processing Agreements

Documents can be provided in `.txt`, `.md`, or other text-based formats.

## Output

### PDF Report

The generated PDF report includes:

- Document metadata and analysis timestamp
- Executive summary with risk distribution
- Detailed clause-by-clause analysis
- Risk classification for each clause
- Specific recommendations for high-risk clauses
- Visual indicators for risk severity levels

Reports are saved to the `files/` directory by default, or to a custom path specified via the `-o` option.

## Development

### Running Tests

```bash
pytest
```

With coverage report:

```bash
pytest --cov=src/legaldoc --cov-report=term-missing
```

### Code Quality

Lint the codebase:

```bash
ruff check .
```

Type checking:

```bash
mypy src/
```

### Adding New Agents

1. Create a new agent file in `src/legaldoc/agents/`
2. Extend the `BaseAgent` class
3. Add corresponding prompt template in `src/legaldoc/prompts/`
4. Register the agent in `src/legaldoc/agents/__init__.py`

## Dependencies

| Package | Purpose |
|---------|---------|
| openai | OpenAI API client |
| google-genai | Google Gemini API client |
| python-dotenv | Environment variable management |
| pydantic | Data validation and schemas |
| instructor | Structured LLM outputs |
| tenacity | Retry logic and resilience |
| reportlab | PDF generation |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
