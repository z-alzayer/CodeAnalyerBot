# Code Analysis Agent

A code analysis tool built during Google's 5-Day GenAI Intensive Course. Uses LangGraph, Gemini models, and RAG to analyze Python codebases.

## Features

- Interactive file-by-file code review
- Filesystem integration (list/read Python files)
- RAG-powered whole-codebase analysis
- Identifies redundant code and refactoring opportunities



## Setup

```bash
# Install the requirements
# Add your Google API key to API_Key.py
python CodeAgent.py
```


## How It Works

Combines:

- File I/O tools
- ChromaDB vector storage
- Gemini embeddings
- LangGraph agent orchestration


## Usage

```
python CodeAgent.py
```

*Created during Google's 5-Day GenAI Intensive, April 2025*

