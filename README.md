# LangChain News Processing API ![](static/favicon.ico)

## Introduction

Welcome to the **LangChain News Processing API** – a cutting-edge, fast, and interactive API built using FastAPI and LangChain. This API leverages Retrieval-Augmented Generation (RAG) techniques to deliver real-time insights and summaries from a curated dataset of news articles.

### Key Features

- **Smart Retrieval:**  
  Perform an advanced similarity search among news articles. This feature can optionally filter results by date if a specific date is mentioned in the query, and uses a powerful language model to rank the most relevant news items.

- **Insight Retrieval:**  
  Obtain concise, analytical summaries of trends and key takeaways from the news articles without referencing individual sources. This provides a high-level overview of market sentiment and emerging trends.

### Data Source

The API processes data from CSV files that include essential fields such as the news publication date, the full text of the news article, and precomputed embeddings. Data is sourced from leading publications including:
- **The Wall Street Journal**
- **Bloomberg**
- **Financial Post**
- **The Verge**

*(Data used in this API covers the period from January to March in 2025.)*

### How It Works

1. **Data Processing:**  
   The API reads a CSV file containing news articles, NER tags and GPT embeddings. It then creates LangChain Document objects and builds a FAISS vector store for efficient similarity search.

2. **Query Handling:**  
   Two types of queries are supported:
   - **Smart Retrieval:** Returns the most relevant news articles based on similarity search and LLM-based ranking.
   - **Insight Retrieval:** Summarizes the key trends and insights derived from the news articles.
   
3. **Interactive Interface:**  
   A simple web interface is provided, featuring a detailed welcome page and a query interface. This interface enables users to select the query type, enter their query, and view results—all styled for a user-friendly experience.

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