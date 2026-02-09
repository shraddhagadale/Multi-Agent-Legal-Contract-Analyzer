# Multi-Agent Legal Contract Analyzer

A multi-agent AI system for analyzing legal documents, with a focus on Non-Disclosure Agreements (NDAs). The system extracts operative clauses, classifies them by category, detects potential risks, and provides actionable recommendations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Overview

The Multi-Agent Legal Contract Analyzer uses a pipeline of specialized AI agents to process and analyze legal documents:

| Agent | Purpose |
|-------|---------|
| Document Analyzer | Identifies document type (Mutual/Unilateral NDA), parties, and provides summary |
| Clause Splitter | Extracts operative clauses, skipping preamble and recitals |
| Clause Classifier | Categorizes clauses (Confidentiality, Term, Indemnification, etc.) |
| Risk Detector | Evaluates risks, assigns severity levels, and provides recommendations |

## Features

- **Operative Clause Extraction**: Focuses on legally binding sections, ignoring preamble and recitals
- **Document Context Analysis**: Automatically identifies parties, agreement type, and key terms
- **Multi-Category Classification**: Classifies clauses into standard legal categories with confidence scores
- **Risk Assessment**: Assigns risk levels (HIGH, MEDIUM, LOW) with detailed explanations
- **Actionable Recommendations**: Provides specific suggestions for improving problematic clauses
- **Structured Outputs**: Uses Pydantic models for type-safe, validated LLM responses
- **Retry Logic**: Built-in exponential backoff for API resilience

## Architecture

```
                    +-------------------+
                    |   Legal Document  |
                    +--------+----------+
                             |
                    +--------v----------+
                    | Document Analyzer |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Clause Splitter  |
                    +--------+----------+
                             |
                    +--------v----------+
                    | Clause Classifier |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   Risk Detector   |
                    +--------+----------+
                             |
                    +--------v----------+
                    |  Analysis Report  |
                    +-------------------+
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/shraddhagadale/Multi-Agent-Legal-Contract-Analyzer.git
cd Multi-Agent-Legal-Contract-Analyzer
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package:

```bash
pip install -e .
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-5

## Usage

### Command Line Interface

Analyze a document:

```bash
python main.py path/to/document.txt
```

Run with verbose output to see each agent's reasoning:

```bash
python main.py path/to/document.txt -v
```

View help:

```bash
python main.py --help
```

### Debug Runner

Test individual agents for development and prompt iteration:

```bash
python debug_runner.py files/sample.txt --agent splitter
python debug_runner.py files/sample.txt --agent classifier
python debug_runner.py files/sample.txt --agent risk
python debug_runner.py files/sample.txt --all
```

### Sample Output

```
=== ANALYSIS SUMMARY ===
Total Clauses: 3
High Risks:    1
Medium Risks:  1
Low Risks:     1

>>> HIGH RISK CLAUSES (1) <<<
Clause 3: Term and Termination

Identified Risks:
  - Perpetual Confidentiality: Obligations survive indefinitely.
  - Indefinite Term: Agreement remains in effect in perpetuity.

Recommendations:
  - Limit confidentiality obligations to 3-5 years for non-trade-secret information.
  - Specify a clear term with renewal options.
```

## Project Structure

```
Multi-Agent-Legal-Contract-Analyzer/
├── main.py                          # CLI entry point and orchestrator
├── debug_runner.py                  # Agent-by-agent debugging utility
├── pyproject.toml                   # Project configuration
├── .env.example                     # Environment template
├── files/                           # Input documents directory
└── src/legaldoc/
    ├── __init__.py
    ├── agents/
    │   ├── base_agent.py            # Abstract base class for agents
    │   ├── document_analyzer_agent.py
    │   ├── splitter_agent.py
    │   ├── classifier_agent.py
    │   └── risk_detector_agent.py
    ├── prompts/
    │   ├── analyzer_prompt.txt
    │   ├── splitter_prompt.txt
    │   ├── classifier_prompt.txt
    │   └── risk_detector_prompt.txt
    └── utils/
        ├── llm_client.py            # OpenAI client with structured outputs
        ├── schemas.py               # Pydantic data models
        └── load_env.py              # Environment configuration
```
