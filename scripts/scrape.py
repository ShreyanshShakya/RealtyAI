#!/usr/bin/env python
"""CLI to run scrapers and save raw data"""
import argparse
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.scraper import DarGlobalScraper, WasaltScraper

settings = get_settings()


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/scrape_{time}.log", rotation="10 MB", level="DEBUG")


def run_darglobal(output_dir: Path, use_playwright: bool = False):
    logger.info("Starting DarGlobal scraper")
    scraper = DarGlobalScraper()
    if use_playwright:
        scraper.enable_playwright()
    try:
        records = scraper.scrape()
        output_file = output_dir / "darglobal_raw.jsonl"
        scraper.save_jsonl(str(output_file))
        scraper.report.print_report()
        logger.success(f"DarGlobal: {len(records)} records saved to {output_file}")
        return scraper.report
    finally:
        if use_playwright:
            scraper.close_playwright()


def run_wasalt(output_dir: Path, use_playwright: bool = False, max_pages: int = 30):
    logger.info("Starting Wasalt scraper")
    scraper = WasaltScraper(max_pages=max_pages)
    if use_playwright:
        scraper.enable_playwright()
    try:
        records = scraper.scrape()
        output_file = output_dir / "wasalt_raw.jsonl"
        scraper.save_jsonl(str(output_file))
        scraper.report.print_report()
        logger.success(f"Wasalt: {len(records)} records saved to {output_file}")
        return scraper.report
    finally:
        if use_playwright:
            scraper.close_playwright()


def main():
    parser = argparse.ArgumentParser(description="RealtyAI Scrapers")
    parser.add_argument("--source", choices=["darglobal", "wasalt", "both"], default="both")
    parser.add_argument("--output", default="dataset/raw", help="Output directory")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for JS-rendered content")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages to scrape (Wasalt)")
    args = parser.parse_args()
    
    setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scrape_reports = []
    if args.source in ["darglobal", "both"]:
        scrape_reports.append(run_darglobal(output_dir, args.playwright))
    if args.source in ["wasalt", "both"]:
        scrape_reports.append(run_wasalt(output_dir, args.playwright, args.max_pages))
    
    # Summary
    total_records = sum(r.pages_parsed for r in scrape_reports)
    logger.success(f"Total records scraped: {total_records}")


if __name__ == "__main__":
    main()