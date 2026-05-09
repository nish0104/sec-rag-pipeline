"""
src/embedding/embedder.py
--------------------------
Phase 2: Load parsed filings, chunk the text, embed using Ollama
(nomic-embed-text), and store in ChromaDB for later retrieval.

Usage:
    python -m src.embedding.embedder --ticker AAPL
"""

import os
import uuid
import pandas as pd
import chromadb
from pathlib import Path
from loguru import logger
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH   = Path("data/embeddings")
PARSED_PATH   = Path("data/parsed")
COLLECTION    = "sec_filings"

PRIORITY_SECTIONS = [
    "business",
    "risk_factors",
    "mda",
    "financials",
    "full_text",
]

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64


# ── Step 1: Load Parsed Parquet Files ─────────────────────────────────────────

def load_parsed_filings(ticker: str) -> pd.DataFrame:
    ticker_dir = PARSED_PATH / ticker.upper()
    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"No parsed data found for {ticker}. "
            f"Run pipeline.py first: python pipeline.py --ticker {ticker}"
        )
    parquet_files = list(ticker_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {ticker_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} sections from {len(parquet_files)} filings for {ticker}")
    return df


# ── Step 2: Filter & Chunk Text ───────────────────────────────────────────────

def chunk_dataframe(df: pd.DataFrame) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    filtered = df[df["section"].isin(PRIORITY_SECTIONS)].copy()
    sections_present = filtered["section"].unique().tolist()
    has_sections = any(s in sections_present for s in ["risk_factors", "mda", "business"])
    if has_sections:
        filtered = filtered[filtered["section"] != "full_text"]

    logger.info(f"Chunking {len(filtered)} sections...")

    for _, row in filtered.iterrows():
        text      = str(row["text"])
        section   = str(row["section"])
        ticker    = str(row["ticker"])
        date      = str(row["filing_date"])
        accession = str(row.get("accession", ""))

        if len(text) < 100:
            continue

        text_chunks = splitter.split_text(text)
        for i, chunk in enumerate(text_chunks):
            if len(chunk.strip()) < 50:
                continue
            chunks.append({
                "id":           f"{ticker}_{accession}_{section}_{i}",
                "text":         chunk,
                "ticker":       ticker,
                "section":      section,
                "filing_date":  date,
                "accession":    accession,
            })

    logger.success(f"Created {len(chunks)} chunks total")
    return chunks


# ── Step 3: Embed & Store in ChromaDB ─────────────────────────────────────────

def embed_and_store(chunks: list[dict], ticker: str, reset: bool = False):
    if not chunks:
        logger.warning("No chunks to embed!")
        return

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if reset:
        logger.info(f"Resetting embeddings for {ticker}...")
        existing = collection.get(where={"ticker": ticker})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    logger.info("Connecting to Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    BATCH_SIZE = 50
    total      = len(chunks)
    stored     = 0

    logger.info(f"Embedding {total} chunks in batches of {BATCH_SIZE}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch     = chunks[batch_start : batch_start + BATCH_SIZE]
        texts     = [c["text"]    for c in batch]
        ids       = [c["id"]      for c in batch]
        metadatas = [{
            "ticker":      c["ticker"],
            "section":     c["section"],
            "filing_date": c["filing_date"],
            "accession":   c["accession"],
        } for c in batch]

        logger.info(f"  Batch {batch_start//BATCH_SIZE + 1}/"
                    f"{(total + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)...")

        vectors = embeddings.embed_documents(texts)
        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids,
        )
        stored += len(batch)
        logger.info(f"  Stored {stored}/{total}")

    logger.success(f"Embedding complete! {stored} chunks stored in ChromaDB")
    return collection


# ── Step 4: Test Retrieval ────────────────────────────────────────────────────

def test_retrieval(ticker: str, query: str = "What are the main risk factors?"):
    logger.info(f"Testing retrieval for: '{query}'")
    client     = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    query_vec  = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        where={"ticker": ticker},
    )

    print(f"\n{'='*60}")
    print(f"Top 3 results for: '{query}'")
    print(f"{'='*60}")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Section : {meta['section']}")
        print(f"Date    : {meta['filing_date']}")
        print(f"Preview : {doc[:300]}...")
    print(f"\n✅ Retrieval working! Found {len(results['documents'][0])} relevant chunks.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_embedding(ticker: str, reset: bool = False):
    ticker = ticker.upper()
    logger.info(f"{'='*50}")
    logger.info(f"Starting Phase 2 (Embedding) for {ticker}")
    logger.info(f"{'='*50}")

    df     = load_parsed_filings(ticker)
    chunks = chunk_dataframe(df)

    if not chunks:
        logger.error("No chunks created. Check your parsed data.")
        return

    embed_and_store(chunks, ticker, reset=reset)
    test_retrieval(ticker)

    logger.success(f"Phase 2 complete for {ticker}!")
    logger.info("Next step → build the RAG chain: python -m src.rag.chain")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Embed SEC filings into ChromaDB")
    parser.add_argument("--ticker", required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--reset",  action="store_true", help="Clear existing embeddings first")
    args = parser.parse_args()
    run_embedding(ticker=args.ticker, reset=args.reset)
