# SEC Filing ETL + LLM Summarization Engine

A fully local, zero-cost RAG pipeline that fetches SEC 10-K/10-Q filings,
parses and chunks them, embeds with Ollama, stores in ChromaDB, and answers
natural language questions about any public company.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data Source | SEC EDGAR API (free, no key) |
| Parsing | BeautifulSoup4, lxml |
| Storage | SQLite (metadata) + Parquet (text) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB (local) |
| LLM | Ollama (Llama 3) |
| RAG Framework | LangChain |
| Frontend | Streamlit |

---

## Project Structure

```
sec-rag-pipeline/
├── data/
│   ├── raw/          # Raw HTML/XML filings from EDGAR
│   ├── parsed/       # Cleaned text + metadata (Parquet)
│   └── embeddings/   # ChromaDB vector store
├── src/
│   ├── ingestion/    # EDGAR API fetcher
│   ├── parsing/      # HTML → clean text + section splitter
│   ├── embedding/    # Chunking + Ollama embedding + ChromaDB
│   └── rag/          # LangChain retrieval chain
├── notebooks/        # Exploration notebooks
├── app.py            # Streamlit frontend
├── pipeline.py       # End-to-end pipeline runner
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama
```bash
# Mac/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the models you need
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Run the pipeline for a company
```bash
# Fetch + parse + embed filings for Apple (ticker: AAPL)
python pipeline.py --ticker AAPL --form 10-K --limit 3
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## Usage Example

```python
from src.rag.chain import SECQueryChain

chain = SECQueryChain(ticker="AAPL")
answer = chain.ask("What are Apple's top risk factors this year?")
print(answer)
```

---

## Phases

- [x] Phase 1 — EDGAR ingestion & ETL
- [ ] Phase 2 — Chunking & embedding pipeline
- [ ] Phase 3 — RAG + LLM Q&A layer
- [ ] Phase 4 — Streamlit UI
- [ ] Phase 5 — Polish & portfolio prep
