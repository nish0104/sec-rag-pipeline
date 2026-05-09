"""
src/ingestion/edgar_fetcher.py
-------------------------------
Fetches SEC 10-K / 10-Q filings from the free EDGAR API.
No API key required. SEC only asks for a descriptive User-Agent.

EDGAR API docs: https://efts.sec.gov/LATEST/search-index?q=%22form+type%22&dateRange=custom
"""

import time
import json
import requests
from pathlib import Path
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────

# SEC requires a User-Agent header identifying who you are
USER_AGENT = "SEC-RAG-Pipeline research@example.com"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

BASE_URL = "https://data.sec.gov"
SUBMISSIONS_URL = f"{BASE_URL}/submissions/CIK{{cik}}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{filename}"

# SEC rate limit: max 10 requests/second — we stay safe at 0.12s between calls
REQUEST_DELAY = 0.15


# ── Ticker → CIK Lookup ───────────────────────────────────────────────────────

def get_cik_from_ticker(ticker: str) -> str:
    """
    Convert a stock ticker (e.g. 'AAPL') to SEC CIK number.
    CIK is zero-padded to 10 digits as required by the API.
    """
    logger.info(f"Looking up CIK for ticker: {ticker.upper()}")

    url = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom"
    tickers_url = "https://www.sec.gov/files/company_tickers.json"

    resp = requests.get(tickers_url, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)

    data = resp.json()

    # Data is a dict of {index: {cik_str, ticker, title}}
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            logger.success(f"Found CIK: {cik} for {ticker.upper()} ({entry['title']})")
            return cik

    raise ValueError(f"Ticker '{ticker}' not found in SEC company list.")


# ── Filing Metadata Fetcher ───────────────────────────────────────────────────

def get_filing_metadata(cik: str, form_type: str = "10-K", limit: int = 5) -> list[dict]:
    """
    Fetch a list of recent filings of a given type for a CIK.

    Returns a list of dicts with keys:
        accession_number, filing_date, report_date, primary_document
    """
    logger.info(f"Fetching {form_type} filing metadata for CIK {cik}")

    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)

    submissions = resp.json()
    filings = submissions.get("filings", {}).get("recent", {})

    # Extract parallel arrays from SEC response
    forms        = filings.get("form", [])
    accessions   = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    report_dates = filings.get("reportDate", [])
    documents    = filings.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form == form_type:
            results.append({
                "cik": cik,
                "form_type": form,
                "accession_number": accessions[i],
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "primary_document": documents[i],
            })
        if len(results) >= limit:
            break

    logger.success(f"Found {len(results)} {form_type} filings")
    return results


# ── Filing Document Downloader ────────────────────────────────────────────────

def download_filing(filing_meta: dict, output_dir: Path) -> Path:
    """
    Download the primary HTML document for a filing and save to output_dir.

    Returns the path to the saved file.
    """
    cik = filing_meta["cik"]
    accession = filing_meta["accession_number"]
    filename  = filing_meta["primary_document"]
    date      = filing_meta["filing_date"]

    # Accession number without dashes for URL
    accession_no_dashes = accession.replace("-", "")

    url = FILING_URL.format(
        cik=int(cik),  # CIK without leading zeros in archive path
        accession_no_dashes=accession_no_dashes,
        filename=filename,
    )

    # Save as: data/raw/AAPL_10-K_2024-11-01.html
    ticker_hint = output_dir.name  # We'll pass ticker as part of dir name
    safe_date   = date.replace("/", "-")
    out_path    = output_dir / f"{accession}_{safe_date}.html"

    if out_path.exists():
        logger.info(f"Already downloaded: {out_path.name} — skipping")
        return out_path

    logger.info(f"Downloading: {url}")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)

    out_path.write_bytes(resp.content)
    logger.success(f"Saved: {out_path}")
    return out_path


# ── Filing Index Fetcher (fallback for missing primaryDocument) ───────────────

def get_filing_index(cik: str, accession_number: str) -> list[dict]:
    """
    Fetch the filing index page to list all documents in a filing.
    Useful when primaryDocument is empty or you want the full document list.
    """
    accession_no_dashes = accession_number.replace("-", "")
    url = f"{BASE_URL}/submissions/{accession_no_dashes}-index.json"

    # Alternative: use the EDGAR filing index
    url = (
        f"https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={cik}"
        f"&type=10-K&dateb=&owner=include&count=10&search_text="
    )

    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_no_dashes}/{accession_no_dashes}-index.json"
    )

    resp = requests.get(index_url, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)

    return resp.json().get("directory", {}).get("item", [])


# ── Main: Run ingestion for a ticker ─────────────────────────────────────────

def ingest_ticker(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 3,
    raw_dir: Path = Path("data/raw"),
) -> list[dict]:
    """
    Full ingestion pipeline for a single ticker:
      1. Look up CIK
      2. Get filing metadata
      3. Download raw HTML files
      4. Return metadata list (for use in next pipeline stage)
    """
    # Create a per-ticker subdirectory
    ticker_dir = raw_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: CIK lookup
    cik = get_cik_from_ticker(ticker)

    # Step 2: Get filing list
    filings = get_filing_metadata(cik, form_type=form_type, limit=limit)

    if not filings:
        logger.warning(f"No {form_type} filings found for {ticker}")
        return []

    # Step 3: Download each filing
    downloaded = []
    for filing in filings:
        try:
            path = download_filing(filing, ticker_dir)
            filing["local_path"] = str(path)
            downloaded.append(filing)
        except Exception as e:
            logger.error(f"Failed to download {filing['accession_number']}: {e}")

    logger.success(f"Ingestion complete: {len(downloaded)}/{len(filings)} filings downloaded")
    return downloaded


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SEC filings from EDGAR")
    parser.add_argument("--ticker", required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--form",   default="10-K", help="Filing type: 10-K or 10-Q")
    parser.add_argument("--limit",  type=int, default=3, help="Number of filings to fetch")
    args = parser.parse_args()

    results = ingest_ticker(
        ticker=args.ticker,
        form_type=args.form,
        limit=args.limit,
    )

    print(json.dumps(results, indent=2))
