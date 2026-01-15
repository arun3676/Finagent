#!/usr/bin/env python3
"""
Batch Ingestion Script

Ingests SEC filings for specified companies into the vector store.
Handles downloading, chunking, embedding, and indexing.

Usage:
    python scripts/ingest_filings.py --tickers AAPL,MSFT,GOOGL --years 2022,2023
    python scripts/ingest_filings.py --config ingestion_config.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.ingestion.sec_edgar_loader import SECEdgarLoader
from app.chunking.sec_chunker import SECChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


async def ingest_company(
    ticker: str,
    years: List[int],
    loader: SECEdgarLoader,
    chunker: SECChunker,
    embedding_service: Optional[EmbeddingService],
    vector_store: Optional[VectorStore],
    filing_types: List[str] = ["10-K", "10-Q"]
) -> dict:
    """
    Ingest all filings for a company.
    
    Args:
        ticker: Stock ticker symbol
        years: Years to ingest
        loader: SEC EDGAR loader
        chunker: Document chunker
        embedding_service: Embedding service (optional for testing)
        vector_store: Vector store client (optional for testing)
        filing_types: Types of filings to ingest
        
    Returns:
        Ingestion statistics
    """
    from app.models import DocumentMetadata, DocumentType
    from app.utils.temporal import derive_fiscal_metadata
    
    stats = {
        "ticker": ticker,
        "filings_processed": 0,
        "chunks_created": 0,
        "sections_extracted": {},
        "errors": []
    }
    
    print(f"\n📊 Processing {ticker}...")
    
    for year in years:
        for filing_type in filing_types:
            try:
                print(f"  📄 Fetching {filing_type} for {year}...")
                
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
                
                filings = await loader.get_filings(
                    ticker=ticker,
                    filing_type=filing_type,
                    limit=1,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not filings:
                    print(f"    ⚠️  No {filing_type} found for {year}")
                    continue
                
                for filing in filings:
                    print(f"    📥 Downloading {filing['filing_date'].strftime('%Y-%m-%d')}...")
                    
                    primary_doc = filing.get("primary_document")
                    filing_text = await loader.download_filing(
                        accession_number=filing["accession_number"],
                        cik=filing["cik"],
                        primary_document=primary_doc
                    )
                    
                    print(f"    📑 Parsing sections...")
                    sections = await loader.parse_filing_sections(filing_text, filing_type)
                    
                    print(f"    ✂️  Extracted {len(sections)} sections:")
                    for section_name in sections.keys():
                        section_len = len(sections[section_name])
                        print(f"       - {section_name}: {section_len:,} chars")
                        stats["sections_extracted"][section_name] = section_len
                    
                    report_date = filing.get("report_date")
                    if isinstance(report_date, str):
                        try:
                            report_date = datetime.strptime(report_date, "%Y-%m-%d")
                        except ValueError:
                            report_date = None

                    derived = derive_fiscal_metadata(
                        report_date=report_date,
                        fiscal_year_end_mmdd=filing.get("fiscal_year_end"),
                        document_type=DocumentType(filing_type)
                    )

                    metadata = DocumentMetadata(
                        ticker=ticker,
                        company_name=filing["company_name"],
                        document_type=DocumentType(filing_type),
                        filing_date=filing["filing_date"],
                        fiscal_year=derived.fiscal_year,
                        fiscal_quarter=derived.fiscal_quarter,
                        fiscal_period=derived.fiscal_period,
                        period_end_date=derived.period_end_date,
                        source_url=filing["url"],
                        accession_number=filing["accession_number"]
                    )
                    
                    print(f"    🔪 Chunking document...")
                    full_text = "\n\n".join(sections.values())
                    chunks = chunker.chunk_document(full_text, metadata)
                    
                    stats["chunks_created"] += len(chunks)
                    stats["filings_processed"] += 1
                    
                    print(f"    ✅ Created {len(chunks)} chunks")
                    
                    if chunks:
                        print(f"\n    📊 Chunk Statistics:")
                        token_counts = [len(chunker.tokenizer.encode(c.content)) for c in chunks]
                        print(f"       - Total chunks: {len(chunks)}")
                        print(f"       - Avg tokens/chunk: {sum(token_counts)/len(token_counts):.0f}")
                        print(f"       - Min tokens: {min(token_counts)}")
                        print(f"       - Max tokens: {max(token_counts)}")
                        
                        section_distribution = {}
                        for chunk in chunks:
                            section = chunk.section or "unknown"
                            section_distribution[section] = section_distribution.get(section, 0) + 1
                        
                        print(f"\n    📂 Chunks by Section:")
                        for section, count in sorted(section_distribution.items()):
                            print(f"       - {section}: {count} chunks")
                
            except Exception as e:
                import traceback
                error_msg = f"{filing_type} {year}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"    ❌ Error: {error_msg}")
                print(f"    {traceback.format_exc()}")
    
    return stats


async def main(
    tickers: List[str],
    years: List[int],
    filing_types: List[str] = ["10-K", "10-Q"]
):
    """
    Main ingestion function.
    
    Args:
        tickers: List of stock tickers
        years: List of years to ingest
        filing_types: Types of filings to ingest
    """
    print("=" * 60)
    print("🚀 FinAgent Batch Ingestion")
    print("=" * 60)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Years: {', '.join(map(str, years))}")
    print(f"Filing Types: {', '.join(filing_types)}")
    print("=" * 60)
    
    # Initialize components
    loader = SECEdgarLoader()
    chunker = SECChunker()
    embedding_service = None
    vector_store = None
    
    # Process each company
    all_stats = []
    for ticker in tickers:
        stats = await ingest_company(
            ticker=ticker,
            years=years,
            loader=loader,
            chunker=chunker,
            embedding_service=embedding_service,
            vector_store=vector_store,
            filing_types=filing_types
        )
        all_stats.append(stats)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 Ingestion Summary")
    print("=" * 60)
    
    total_filings = sum(s["filings_processed"] for s in all_stats)
    total_chunks = sum(s["chunks_created"] for s in all_stats)
    total_errors = sum(len(s["errors"]) for s in all_stats)
    
    print(f"Total Filings Processed: {total_filings}")
    print(f"Total Chunks Created: {total_chunks}")
    print(f"Total Errors: {total_errors}")
    
    if total_errors > 0:
        print("\nErrors:")
        for stats in all_stats:
            for error in stats["errors"]:
                print(f"  - {stats['ticker']}: {error}")
    
    print("\n✅ Ingestion complete!")
    
    return all_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into FinAgent vector store"
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of stock tickers (e.g., AAPL,MSFT,GOOGL)"
    )
    
    parser.add_argument(
        "--years",
        type=str,
        default="2023",
        help="Comma-separated list of years (e.g., 2022,2023)"
    )
    
    parser.add_argument(
        "--filing-types",
        type=str,
        default="10-K,10-Q",
        help="Comma-separated list of filing types (e.g., 10-K,10-Q,8-K)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file (overrides other arguments)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.config:
        # Load from config file
        with open(args.config) as f:
            config = json.load(f)
        tickers = config.get("tickers", [])
        years = config.get("years", [2023])
        filing_types = config.get("filing_types", ["10-K", "10-Q"])
    else:
        # Parse from command line
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        years = [int(y.strip()) for y in args.years.split(",")]
        filing_types = [ft.strip() for ft in args.filing_types.split(",")]
    
    # Run ingestion
    asyncio.run(main(tickers, years, filing_types))
