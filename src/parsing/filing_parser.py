"""
src/parsing/filing_parser.py
------------------------------
Converts raw EDGAR HTML filings into clean, structured text.
Splits filings into meaningful sections (Risk Factors, MD&A, etc.)
for better chunking quality downstream.
"""

import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from loguru import logger


# ── SEC 10-K Section Patterns ─────────────────────────────────────────────────
# These are standard section headers found in most 10-K filings

SECTION_PATTERNS = {
    "business":          r"item\s*1[^a][\.\s]*business",
    "risk_factors":      r"item\s*1a[\.\s]*risk\s*factors",
    "properties":        r"item\s*2[\.\s]*properties",
    "legal_proceedings": r"item\s*3[\.\s]*legal\s*proceedings",
    "mda":               r"item\s*7[\.\s]*management.{0,10}discussion",
    "financials":        r"item\s*8[\.\s]*financial\s*statements",
    "controls":          r"item\s*9a[\.\s]*controls",
    "executive_comp":    r"item\s*11[\.\s]*executive\s*compensation",
}


# ── HTML → Clean Text ─────────────────────────────────────────────────────────

def html_to_text(html_path: Path) -> str:
    """
    Parse raw EDGAR HTML and extract clean plain text.
    Handles both standard HTML and EDGAR's SGML-wrapped format.
    """
    logger.info(f"Parsing HTML: {html_path.name}")

    raw = html_path.read_bytes()

    # EDGAR files are sometimes SGML-wrapped — extract the HTML portion
    raw_str = raw.decode("utf-8", errors="replace")
    if "<DOCUMENT>" in raw_str:
        # Extract just the HTML inside <TEXT> tags
        match = re.search(r"<TEXT>(.*?)</TEXT>", raw_str, re.DOTALL | re.IGNORECASE)
        if match:
            raw_str = match.group(1)

    soup = BeautifulSoup(raw_str, "lxml")

    # Remove script, style, and hidden elements
    for tag in soup(["script", "style", "meta", "head"]):
        tag.decompose()

    # Extract text with single-space separator between tags
    text = soup.get_text(separator=" ", strip=True)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    logger.success(f"Extracted {len(text):,} characters of clean text")
    return text


# ── Section Splitter ──────────────────────────────────────────────────────────

def split_into_sections(text: str) -> dict[str, str]:
    """
    Split full filing text into named sections based on Item headers.

    Returns a dict: { section_name: section_text }
    The 'full_text' key always contains the complete document.
    """
    sections = {"full_text": text}
    text_lower = text.lower()

    # Find the character position of each section header
    positions = {}
    for section_name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text_lower)
        if match:
            positions[section_name] = match.start()

    # Sort sections by position in document
    sorted_sections = sorted(positions.items(), key=lambda x: x[1])

    # Extract text between consecutive section headers
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            end_pos = sorted_sections[i + 1][1]
        else:
            end_pos = len(text)

        section_text = text[start_pos:end_pos].strip()
        sections[section_name] = section_text
        logger.debug(f"  Section '{section_name}': {len(section_text):,} chars")

    logger.info(f"Split into {len(sections) - 1} sections")
    return sections


# ── Metadata Extractor ────────────────────────────────────────────────────────

def extract_metadata(text: str, filing_meta: dict) -> dict:
    """
    Extract key metadata from filing text + known metadata.
    """
    # Try to extract fiscal year from text
    fy_match = re.search(r"fiscal\s+year\s+(?:ended|ending)\s+([\w\s,]+\d{4})", text[:5000], re.IGNORECASE)
    fiscal_year = fy_match.group(1).strip() if fy_match else filing_meta.get("report_date", "")

    # Try to extract company name
    company_match = re.search(r"(?:annual report|form 10-[kq])\s+(?:of\s+)?([A-Z][A-Z\s,\.]+(?:Inc|Corp|LLC|Ltd|Co)\.?)", text[:3000])
    company_name = company_match.group(1).strip() if company_match else ""

    return {
        **filing_meta,
        "fiscal_year": fiscal_year,
        "company_name": company_name,
        "text_length": len(text),
    }


# ── Main Parser Function ──────────────────────────────────────────────────────

def parse_filing(html_path: Path, filing_meta: dict, output_dir: Path) -> dict:
    """
    Full parse pipeline for a single filing:
      1. HTML → clean text
      2. Split into sections
      3. Save each section as a Parquet row
      4. Return parsed metadata

    Output Parquet schema:
      ticker, form_type, filing_date, section, text, char_count
    """
    # Step 1: Extract clean text
    full_text = html_to_text(html_path)

    # Step 2: Split into sections
    sections = split_into_sections(full_text)

    # Step 3: Build records for each section
    ticker = filing_meta.get("ticker", "UNKNOWN")
    records = []
    for section_name, section_text in sections.items():
        if len(section_text) < 100:  # Skip empty/tiny sections
            continue
        records.append({
            "ticker":       ticker,
            "cik":          filing_meta.get("cik", ""),
            "form_type":    filing_meta.get("form_type", ""),
            "filing_date":  filing_meta.get("filing_date", ""),
            "report_date":  filing_meta.get("report_date", ""),
            "accession":    filing_meta.get("accession_number", ""),
            "section":      section_name,
            "text":         section_text,
            "char_count":   len(section_text),
        })

    # Step 4: Save to Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}_{filing_meta.get('accession_number', 'unknown')}.parquet"

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    logger.success(f"Saved parsed filing: {out_path} ({len(records)} sections)")

    return {
        **filing_meta,
        "parsed_path": str(out_path),
        "sections_found": [r["section"] for r in records],
        "total_chars": sum(r["char_count"] for r in records),
    }


# ── Batch Parser ──────────────────────────────────────────────────────────────

def parse_all_filings(
    downloaded_filings: list[dict],
    ticker: str,
    output_dir: Path = Path("data/parsed"),
) -> list[dict]:
    """
    Parse all downloaded filings for a ticker.
    """
    parsed_results = []
    for filing in downloaded_filings:
        html_path = Path(filing["local_path"])
        if not html_path.exists():
            logger.warning(f"File not found: {html_path}")
            continue
        try:
            result = parse_filing(
                html_path=html_path,
                filing_meta={**filing, "ticker": ticker},
                output_dir=output_dir / ticker.upper(),
            )
            parsed_results.append(result)
        except Exception as e:
            logger.error(f"Failed to parse {html_path.name}: {e}")

    logger.success(f"Parsed {len(parsed_results)}/{len(downloaded_filings)} filings")
    return parsed_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, argparse

    parser = argparse.ArgumentParser(description="Parse downloaded EDGAR filings")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--input-dir",  default="data/raw")
    parser.add_argument("--output-dir", default="data/parsed")
    args = parser.parse_args()

    # Load downloaded filing metadata from JSON (saved by ingestion step)
    meta_path = Path(args.input_dir) / args.ticker.upper() / "filings_meta.json"
    with open(meta_path) as f:
        filings = json.load(f)

    results = parse_all_filings(
        downloaded_filings=filings,
        ticker=args.ticker,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(results, indent=2))
