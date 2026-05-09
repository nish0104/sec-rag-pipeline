"""
src/rag/chain.py
-----------------
Phase 3: RAG chain that wires ChromaDB retriever to Ollama Llama3 LLM.

Prerequisites:
    1. Complete Phase 2 first (embeddings must exist in ChromaDB)
    2. ollama pull llama3 (already done)

Usage:
    python -m src.rag.chain --ticker AAPL --question "What are Apple's risks?"
"""

import chromadb
from pathlib import Path
from loguru import logger
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH = Path("data/embeddings")
COLLECTION  = "sec_filings"

# Prompt template — tells the LLM how to behave
PROMPT_TEMPLATE = """You are a financial analyst assistant with expertise in
SEC filings. Use ONLY the following excerpts from official SEC filings to
answer the question. Be specific, concise, and always mention which section
the information comes from (e.g. "According to the Risk Factors section...").
If the answer is not in the provided context, say "I don't have enough
information in the provided filings to answer that."

SEC Filing Excerpts:
{context}

Question: {question}

Answer:"""


# ── RAG Chain Class ───────────────────────────────────────────────────────────

class SECQueryChain:
    """
    Full RAG pipeline:
      Query → ChromaDB retrieval → Llama3 LLM → Answer with sources
    """

    def __init__(self, ticker: str, n_results: int = 5):
        self.ticker   = ticker.upper()
        self.n_results = n_results
        self._setup()

    def _setup(self):
        """Initialize embeddings, vector store, LLM, and chain."""
        logger.info(f"Setting up RAG chain for {self.ticker}...")

        # ── Embeddings (same model used in Phase 2) ───────────────────────
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # ── ChromaDB vector store ─────────────────────────────────────────
        if not CHROMA_PATH.exists():
            raise FileNotFoundError(
                "No embeddings found. Run Phase 2 first: "
                "python -m src.embedding.embedder --ticker {self.ticker}"
            )

        self.vectorstore = Chroma(
            client=chromadb.PersistentClient(path=str(CHROMA_PATH)),
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
        )

        # Filter retrieval to only this ticker's documents
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.n_results,
                "filter": {"ticker": self.ticker},
            },
        )

        # ── Ollama LLM (Llama3) ───────────────────────────────────────────
        logger.info("Connecting to Ollama (llama3)...")
        self.llm = Ollama(model="llama3.2", temperature=0)

        # ── Prompt ────────────────────────────────────────────────────────
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        # ── LCEL Chain: retriever | prompt | llm | parser ─────────────────
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.success(f"RAG chain ready for {self.ticker}!")

    def ask(self, question: str) -> dict:
        """
        Ask a question about the company's SEC filings.

        Returns:
            {
                "answer":  str,
                "sources": list of { section, filing_date, preview }
            }
        """
        logger.info(f"Question: {question}")

        # Get answer from chain
        answer = self.chain.invoke(question)

        # Also retrieve source docs separately for display
        src_docs = self.retriever.invoke(question)

        # Format sources for display
        sources = []
        for doc in src_docs:
            sources.append({
                "section":     doc.metadata.get("section", "unknown"),
                "filing_date": doc.metadata.get("filing_date", "unknown"),
                "preview":     doc.page_content[:200] + "...",
            })

        logger.success("Answer generated!")
        return {
            "answer":  answer,
            "sources": sources,
        }

    def ask_pretty(self, question: str):
        """Ask a question and print a nicely formatted answer."""
        result = self.ask(question)

        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        print(f"\n📊 ANSWER:\n{result['answer']}")
        print(f"\n{'─'*60}")
        print(f"📎 SOURCES ({len(result['sources'])} chunks retrieved):")
        for i, src in enumerate(result["sources"]):
            print(f"\n  [{i+1}] Section: {src['section']} | Date: {src['filing_date']}")
            print(f"       {src['preview']}")
        print(f"{'='*60}\n")


# ── CLI: Interactive Q&A ──────────────────────────────────────────────────────

def interactive_mode(ticker: str):
    """Run an interactive Q&A session in the terminal."""
    chain = SECQueryChain(ticker=ticker)

    print(f"\n🚀 SEC Filing Q&A — {ticker}")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            chain.ask_pretty(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEC RAG Q&A Chain")
    parser.add_argument("--ticker",   required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--question", default=None,  help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive or not args.question:
        interactive_mode(args.ticker)
    else:
        chain = SECQueryChain(ticker=args.ticker)
        chain.ask_pretty(args.question)
