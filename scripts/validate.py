#!/usr/bin/env python
"""CLI to validate raw scraped data into PropertyRecords"""
import argparse
import sys
import json
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.scraper import DarGlobalScraper, WasaltScraper
from app.data.processor import ValidationProcessor

settings = get_settings()


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def load_raw_records(filepath: Path):
    """Load raw JSONL records"""
    from app.types import ScrapedRecord
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(ScrapedRecord(
                    source=data.pop("source"),
                    source_url=data.pop("source_url"),
                    raw_data=data
                ))
    return records


def validate_source(source: str, raw_dir: Path, processed_dir: Path, rejected_dir: Path):
    """Validate records for a single source"""
    raw_file = raw_dir / f"{source.lower()}_raw.jsonl"
    if not raw_file.exists():
        logger.error(f"Raw file not found: {raw_file}")
        return None
    
    logger.info(f"Loading raw records from {raw_file}")
    raw_records = load_raw_records(raw_file)
    logger.info(f"Loaded {len(raw_records)} raw records")
    
    # Validate
    processor = ValidationProcessor(source)
    valid, rejected = processor.process(raw_records)
    
    # Save outputs
    processed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    validated_file = processed_dir / f"{source.lower()}_validated.jsonl"
    rejected_file = rejected_dir / f"{source.lower()}_rejected.jsonl"
    low_quality_file = rejected_dir / f"{source.lower()}_low_quality.jsonl"
    
    processor.save_validated(validated_file)
    processor.save_rejected(rejected_file)
    processor.save_low_quality(low_quality_file)
    
    # Create a mock scrape report for final report
    from app.types import ScrapeReport
    scrape_report = ScrapeReport(source=source)
    scrape_report.pages_discovered = len(raw_records)
    scrape_report.pages_fetched = len(raw_records)
    scrape_report.pages_parsed = len(raw_records)
    
    processor.print_final_report(scrape_report)
    
    return {
        "valid": len(valid),
        "rejected": len(rejected),
        "low_quality": len(processor.low_quality_records)
    }


def main():
    parser = argparse.ArgumentParser(description="RealtyAI Data Validation")
    parser.add_argument("--source", choices=["darglobal", "wasalt", "both"], default="both")
    parser.add_argument("--raw-dir", default="dataset/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", default="dataset/processed", help="Processed data directory")
    parser.add_argument("--rejected-dir", default="dataset/rejected", help="Rejected data directory")
    args = parser.parse_args()
    
    setup_logging()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    rejected_dir = Path(args.rejected_dir)
    
    results = {}
    if args.source in ["darglobal", "both"]:
        results["DarGlobal"] = validate_source("DarGlobal", raw_dir, processed_dir, rejected_dir)
    if args.source in ["wasalt", "both"]:
        results["Wasalt"] = validate_source("Wasalt", raw_dir, processed_dir, rejected_dir)
    
    # Final summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    for source, counts in results.items():
        if counts:
            print(f"{source}:")
            print(f"  Valid:         {counts['valid']}")
            print(f"  Rejected:      {counts['rejected']}")
            print(f"  Low quality:   {counts['low_quality']}")
    print("="*50)


if __name__ == "__main__":
    main()