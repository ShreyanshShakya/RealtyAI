"""Validation pipeline: Raw scraped records -> Validated PropertyRecords."""
from pathlib import Path
from typing import List, Tuple
import json
from loguru import logger

from app.data.schema import PropertyRecord
from app.types import ScrapedRecord


class ValidationProcessor:
    """Convert raw scraped records to validated PropertyRecords"""
    
    def __init__(self, source: str):
        self.source = source
        self.valid_records: List[PropertyRecord] = []
        self.rejected_records: List[dict] = []
        self.low_quality_records: List[PropertyRecord] = []
    
    def process(self, raw_records: List[ScrapedRecord]) -> Tuple[List[PropertyRecord], List[dict]]:
        """Process raw records through Pydantic validation"""
        # Reset state for each processing run
        self.valid_records = []
        self.rejected_records = []
        self.low_quality_records = []
        
        for record in raw_records:
            try:
                # Merge source metadata with raw data - raw_data does NOT override source/source_url
                # Fixed merge order: raw_data first, then trusted fields
                data = {
                    **record.raw_data,
                    "source": record.source,
                    "source_url": record.source_url
                }
                
                # Validate with Pydantic
                validated = PropertyRecord(**data)
                
                # Check data quality
                quality = validated.data_quality_score()
                if quality < 0.3:  # Less than 30% fields filled
                    self.low_quality_records.append(validated)
                    logger.warning(f"Low data quality ({quality:.1%}): {validated.source_url}")
                
                self.valid_records.append(validated)
                
            except Exception as e:
                self.rejected_records.append({
                    "source_url": record.source_url,
                    "raw_data": record.raw_data,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                logger.warning(f"Rejected {record.source_url}: {e}")
        
        logger.info(f"{self.source}: {len(self.valid_records)} valid, {len(self.rejected_records)} rejected, {len(self.low_quality_records)} low-quality")
        return self.valid_records, self.rejected_records
    
    def save_validated(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.valid_records:
                f.write(record.model_dump_json() + "\n")
        logger.info(f"Saved {len(self.valid_records)} validated records to {output_path}")
    
    def save_rejected(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.rejected_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.rejected_records)} rejected records to {output_path}")
    
    def save_low_quality(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.low_quality_records:
                f.write(record.model_dump_json() + "\n")
        logger.info(f"Saved {len(self.low_quality_records)} low-quality records to {output_path}")
    
    def generate_final_report(self, scrape_report) -> dict:
        """Generate combined scraping + validation report"""
        return {
            "source": self.source,
            "pages_discovered": scrape_report.pages_discovered,
            "pages_fetched": scrape_report.pages_fetched,
            "pages_parsed": scrape_report.pages_parsed,
            "valid_records": len(self.valid_records),
            "rejected_records": len(self.rejected_records),
            "low_quality_records": len(self.low_quality_records),
            "failed_requests": scrape_report.failed_requests,
            "errors": scrape_report.errors,
        }
    
    def print_final_report(self, scrape_report):
        """Print combined report"""
        report = self.generate_final_report(scrape_report)
        print(f"\n{'='*50}")
        print(f"REALTYAI FINAL REPORT: {self.source}")
        print(f"{'='*50}")
        print(f"Pages discovered:    {report['pages_discovered']}")
        print(f"Pages fetched:       {report['pages_fetched']}")
        print(f"Pages parsed:        {report['pages_parsed']}")
        print(f"Valid records:       {report['valid_records']}")
        print(f"Low quality records: {report['low_quality_records']}")
        print(f"Rejected records:    {report['rejected_records']}")
        print(f"Failed requests:     {report['failed_requests']}")
        if report['errors']:
            print(f"Errors:")
            for err, count in report['errors'].items():
                print(f"  {err}: {count}")
        print(f"{'='*50}\n")