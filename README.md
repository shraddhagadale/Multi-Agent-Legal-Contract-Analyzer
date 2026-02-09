# Multi-Agent Legal Contract Analyzer

A multi-agent AI system for analyzing legal documents, with a focus on Non-Disclosure Agreements (NDAs). The system intelligently extracts operative clauses, classifies them by category, detects potential risks, and provides actionable recommendations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Supported Document Types](#supported-document-types)
- [Development](#development)
- [License](#license)

## Overview

The Multi-Agent Legal Contract Analyzer uses a pipeline of specialized AI agents to process and analyze legal documents:

1. **Document Analyzer Agent**: Detects the document type (e.g., Unilateral vs. Mutual NDA), identifies parties, and provides a high-level summary.
2. **Clause Splitter Agent**: Parses the document and breaks it down into individual logical **operative clauses** (skipping preamble, recitals, and boilerplate introduction).
3. **Clause Classifier Agent**: Categorizes each clause by type (e.g., Confidentiality, Term, Obligations, Indemnification).
4. **Risk Detector Agent**: Evaluates each clause for potential legal risks, assigns severity levels, and provides specific recommendations for safer alternatives.

## Features

- **Operative Clause Extraction**: Intelligently identifies and extracts legally binding clauses while ignoring non-operative text like recitals and preambles.
- **Automated Summary & Metadata**: Automatically identifies parties and provides a concise document summary.
- **Multi-Category Classification**: Classifies clauses into standard legal categories.
- **Risk Assessment**: Assigns risk levels (HIGH, MEDIUM, LOW) with detailed explanations.
- **Actionable Recommendations**: Provides specific suggestions for improving problematic clauses.
- **OpenAI Integration**: Powered by OpenAI models (e.g., gpt-4o-mini) with structured output handling.
- **Command-Line Interface**: Simple CLI for easy integration into workflows.

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

2. Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
```

## Usage

### Command Line Interface

Analyze a document and view the results in the console:

```bash
python main.py path/to/document.txt
```

Run with verbose output to see each agent's step:

```bash
python main.py path/to/document.txt -v
```

View help and available options:

```bash
python main.py --help
```

### Analysis Results Structure

The analysis results include:

| Key | Description |
|-----|-------------|
| `total_clauses` | Total number of operative clauses identified |
| `clauses` | List of extracted operative clause texts |
| `classifications` | Classification results for each clause |
| `risk_assessments` | Risk analysis for each clause |
| `high_risk_clauses` | Clauses flagged as high risk |
| `medium_risk_clauses` | Clauses flagged as medium risk |
| `low_risk_clauses` | Clauses flagged as low risk |
| `model_used` | LLM model used for analysis |

## Project Structure

```
Multi-Agent-Legal-Contract-Analyzer/
├── main.py                      # CLI entry point and orchestrator
├── debug_runner.py              # Tool for debugging individual agents
├── pyproject.toml               # Project configuration and dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project documentation
├── files/                       # Input directory for documents
└── src/
    └── legaldoc/
        ├── __init__.py
        ├── agents/              # AI agent implementations
        │   ├── __init__.py
        │   ├── base_agent.py           # Base agent class
        │   ├── document_analyzer_agent.py # Document context agent
        │   ├── splitter_agent.py       # Clause extraction agent
        │   ├── classifier_agent.py     # Clause classification agent
        │   └── risk_detector_agent.py  # Risk assessment agent
        ├── prompts/             # Agent prompt templates
        │   ├── analyzer_prompt.txt
        │   ├── splitter_prompt.txt
        │   ├── classifier_prompt.txt
        │   └── risk_detector_prompt.txt
        └── utils/               # Utility modules
            ├── __init__.py
            ├── llm_client.py            # OpenAI client abstraction
            ├── schemas.py               # Pydantic data models
            └── load_env.py              # Environment configuration
```

## Supported Document Types

The analyzer is optimized for:
- Non-Disclosure Agreements (NDAs)

But can be extended for:
- Employment Agreements
- Consulting Agreements
- Master Service Agreements (MSAs)

## Development

### Running Tests

```bash
pytest
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

## Dependencies

| Package | Purpose |
|---------|---------|
| openai | OpenAI API client |
| python-dotenv | Environment variable management |
| pydantic | Data validation and schemas |
| instructor | Structured LLM outputs |
| tenacity | Retry logic and resilience |

## License

This project is licensed under the MIT License.
