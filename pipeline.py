"""
pipeline.py
-----------
End-to-end orchestrator for the SEC RAG pipeline.
Run this to ingest + parse filings for any ticker.

Usage:
    python pipeline.py --ticker AAPL --form 10-K --limit 3
"""

import json
import argparse
from pathlib import Path
from loguru import logger

from src.ingestion.edgar_fetcher import ingest_ticker
from src.parsing.filing_parser import parse_all_filings

# Configure loguru to also write to a log file
logger.add("logs/pipeline.log", rotation="10 MB", level="DEBUG")


def run_pipeline(ticker: str, form_type: str = "10-K", limit: int = 3):
    """
    Run Phase 1 + Phase 2 (parsing) of the pipeline.
    Phase 3 (embedding) will be added next.
    """
    ticker = ticker.upper()
    logger.info(f"{'='*50}")
    logger.info(f"Starting pipeline for {ticker} | {form_type} | limit={limit}")
    logger.info(f"{'='*50}")

    # ── Phase 1: Ingestion ────────────────────────────────────
    logger.info("PHASE 1: Ingesting filings from EDGAR...")
    downloaded = ingest_ticker(
        ticker=ticker,
        form_type=form_type,
        limit=limit,
        raw_dir=Path("data/raw"),
    )

    if not downloaded:
        logger.error("No filings downloaded. Exiting.")
        return

    # Save metadata for later use
    meta_dir = Path("data/raw") / ticker
    meta_path = meta_dir / "filings_meta.json"
    with open(meta_path, "w") as f:
        json.dump(downloaded, f, indent=2)
    logger.info(f"Saved filing metadata to {meta_path}")

    # ── Phase 2: Parsing ──────────────────────────────────────
    logger.info("PHASE 2: Parsing HTML filings into clean text...")
    parsed = parse_all_filings(
        downloaded_filings=downloaded,
        ticker=ticker,
        output_dir=Path("data/parsed"),
    )

    # ── Summary ───────────────────────────────────────────────
    logger.info(f"{'='*50}")
    logger.success(f"Pipeline complete for {ticker}")
    logger.info(f"  Filings downloaded : {len(downloaded)}")
    logger.info(f"  Filings parsed     : {len(parsed)}")
    for p in parsed:
        logger.info(f"  → {p.get('filing_date')} | sections: {p.get('sections_found')}")
    logger.info(f"{'='*50}")

    logger.info("Next step → run embedding pipeline:")
    logger.info(f"  python -m src.embedding.embedder --ticker {ticker}")

    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC RAG Pipeline Runner")
    parser.add_argument("--ticker", required=True,    help="Stock ticker e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--form",   default="10-K",   help="Filing type: 10-K or 10-Q")
    parser.add_argument("--limit",  type=int, default=3, help="Number of filings to fetch (default: 3)")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        form_type=args.form,
        limit=args.limit,
    )
