# RAG-Based Code Modification Tool ![](static/favicon.ico)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) workflow for **understanding, retrieving, and modifying complex R package codebases**.  
It is optimized for multi-file, dependency-heavy R packages such as `gsDesign2`, enabling structured code understanding and precise modifications.

The system:
- Retrieves relevant code chunks and descriptions from an indexed knowledge base.
- Incorporates **external user-provided sources** (e.g., PDFs) for additional context.
- Supports section-based rendering for improved readability.
- Provides a clean HTML interface for querying and viewing results.

### 1. **Knowledge Base Construction**

We build a structured knowledge base from each R file in the target package, storing:
- **File name**
- **Full code**
- **Detailed natural language description** (line-by-line)
- **Dependencies** (files/functions it relies on)
- **Function list**
- **Notes and labels** for categorization

Embedding-based vector search enables semantic retrieval, and dependency-aware retrieval ensures that when one file is retrieved, all its related dependency files are also included.

#### **Data Source**
Our primary data source is the [`gsDesign2` R package](https://merck.github.io/gsDesign2/index.html), developed by Merck.  
`gsDesign2` is a statistical package for designing and analyzing clinical trials—especially **group sequential** and **non-proportional hazards** designs.  
It provides tools for:
- Computing sample sizes and power under various trial designs
- Creating weighted log-rank (WLR) tests for survival analysis
- Simulating trial operating characteristics
- Supporting fixed-sample designs, interim analyses, and adaptive designs

Because `gsDesign2` contains **complex interdependent R scripts**, it is an ideal case study for our retrieval and modification pipeline.

For reference:
- **User query & agent response history** is stored in [./service/conversations.jsonl](./service/conversations.jsonl)  
- **User's file upload history** can be found under [./upload](./upload)

### 2. **Flexible Retrieval Pipeline**
- **Primary search:** Embedding similarity using `text-embedding-3-large`.
- **Dependency expansion:** Automatically retrieves files listed in the `Dependencies` column for relevant results.
- **Optional external input:** If the user uploads a PDF, content from it is also incorporated into the LLM context.

### 3. **Response Rendering**
- Sections are clearly separated with `[[SECTION]] ... [[/SECTION]]` markers for accurate parsing.
- Original LLM output can be optionally printed for debugging.
- HTML interface uses styled **cards** for each section:
  - Summary
  - Dependency Impact
  - Code Changes
  - Additional Notes

### 4. **User Interface**
- **FastAPI** backend with `/interface` endpoint for the web view.
- Styled HTML + CSS with adjustable width for better readability.
- Real-time query processing and rendering.

---

## Setting up the Virtual Environment
```bash
# Create the virtual environment:
python3.11 -m venv venv

# Activate the virtual environment:
source venv/bin/activate

# Install required packages:
python3 -m pip install -r requirements.txt
```
---

## Run the Application
```bash
# Run the application:
uvicorn app.main:app --reload
```
