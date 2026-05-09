# 📊 SEC Filing ETL + RAG Q&A Engine

> Ask natural language questions about any public company's official SEC filings — powered by a fully local, zero-cost RAG pipeline.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square)
![SEC EDGAR](https://img.shields.io/badge/SEC_EDGAR-Free_API-blue?style=flat-square)

---

## 🧠 What This Project Does

Every public company is legally required to file detailed annual (10-K) and quarterly (10-Q) reports with the SEC. These filings contain critical information about risks, financials, and strategy — but they're often 80,000+ words long.

This project builds an **end-to-end data engineering + AI pipeline** that:

1. **Fetches** official SEC filings from the free EDGAR API for any US ticker
2. **Parses & cleans** raw HTML into structured sections (Risk Factors, MD&A, Financials, etc.)
3. **Chunks & embeds** the text using a local Ollama model and stores vectors in ChromaDB
4. **Answers questions** using a RAG chain that retrieves relevant chunks and feeds them to a local LLM
5. **Serves everything** through a Bloomberg-style Streamlit web interface

**Zero API costs. Everything runs locally on your machine.**

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Web App                  │
└────────────────────┬────────────────────────────┘
                     │
     ┌───────────────▼───────────────┐
     │         RAG Chain             │
     │  (LangChain + Ollama phi3)    │
     └───────┬───────────────────────┘
             │
    ┌────────▼────────┐
    │   ChromaDB      │  ◄── Vector similarity search
    │  (Local VectorDB)│
    └────────▲────────┘
             │
    ┌────────┴────────────────────────┐
    │     Embedding Pipeline          │
    │  (Ollama nomic-embed-text)      │
    └────────▲────────────────────────┘
             │
    ┌────────┴────────────────────────┐
    │      ETL Pipeline               │
    │  EDGAR API → HTML → Clean Text  │
    │  → Section Split → Parquet      │
    └─────────────────────────────────┘
```

---

## 🚀 Features

- **Real financial data** — pulls directly from SEC EDGAR (official US government source)
- **Fully local AI** — no OpenAI, no Anthropic API, no subscription fees ever
- **Smart section parsing** — splits filings into Risk Factors, MD&A, Business, Financials
- **Vector semantic search** — finds relevant content by meaning, not just keywords
- **Source citations** — every answer shows exactly which section it came from
- **Multi-ticker support** — index any US public company and switch between them
- **Bloomberg-style UI** — professional dark terminal aesthetic

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Data Source | SEC EDGAR API | Free, official SEC filings |
| ETL | Python, BeautifulSoup, lxml | HTML parsing & cleaning |
| Storage | Parquet, SQLite | Structured filing storage |
| Embeddings | Ollama (nomic-embed-text) | Local text → vector conversion |
| Vector DB | ChromaDB | Semantic similarity search |
| LLM | Ollama (phi3) | Local answer generation |
| RAG Framework | LangChain | Retrieval chain orchestration |
| Frontend | Streamlit | Web interface |

---

## 📁 Project Structure

```
sec-rag-pipeline/
├── data/
│   ├── raw/          # Downloaded HTML filings from SEC
│   ├── parsed/       # Cleaned text saved as Parquet files
│   └── embeddings/   # ChromaDB vector store
├── src/
│   ├── ingestion/
│   │   └── edgar_fetcher.py    # SEC EDGAR API client
│   ├── parsing/
│   │   └── filing_parser.py    # HTML → clean text + section splitter
│   ├── embedding/
│   │   └── embedder.py         # Chunking + Ollama embedding + ChromaDB
│   └── rag/
│       └── chain.py            # LangChain RAG retrieval chain
├── pipeline.py       # Master ETL orchestrator
├── app.py            # Streamlit web application
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone the repository
```bash
git clone https://github.com/nish0104/sec-rag-pipeline.git
cd sec-rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-ollama langchain-text-splitters
```

### 4. Pull Ollama models
```bash
ollama pull nomic-embed-text
ollama pull phi3
```

### 5. Run the full pipeline
```bash
python pipeline.py --ticker AAPL --form 10-K --limit 3
python -m src.embedding.embedder --ticker AAPL
streamlit run app.py
```

---

## 💬 Example Questions

- *"What are Apple's main risk factors?"*
- *"How did revenue change compared to last year?"*
- *"What is the company's AI strategy?"*
- *"Are there any significant legal proceedings?"*

---

## 🔮 Future Improvements

- [ ] Multi-company comparison (AAPL vs MSFT side by side)
- [ ] Year-over-year change detection
- [ ] Automatic financial metric extraction
- [ ] Deploy to Streamlit Community Cloud

---

## 👩‍💻 Author

**Nishthaben Vaghani**
- Portfolio: [nishthaben-vaghani.vercel.app](https://nishthaben-vaghani.vercel.app)
- GitHub: [@nish0104](https://github.com/nish0104)
