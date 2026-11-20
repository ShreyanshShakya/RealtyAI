"""Shared data types to avoid circular imports."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ScrapedRecord:
    """Raw scraped data before schema validation"""
    source: str
    source_url: str
    raw_data: Dict[str, Any]


@dataclass
class ScrapeReport:
    """Scraping diagnostics report"""
    source: str
    pages_discovered: int = 0
    pages_fetched: int = 0
    pages_parsed: int = 0
    valid_records: int = 0
    rejected_records: int = 0
    failed_requests: int = 0
    errors: Dict[str, int] = field(default_factory=dict)
    
    def print_report(self):
        print(f"\n{'='*50}")
        print(f"REALTYAI SCRAPING REPORT")
        print(f"{'='*50}")
        print(f"Source:              {self.source}")
        print(f"Pages discovered:    {self.pages_discovered}")
        print(f"Pages fetched:       {self.pages_fetched}")
        print(f"Pages parsed:        {self.pages_parsed}")
        print(f"Valid records:       {self.valid_records}")
        print(f"Rejected records:    {self.rejected_records}")
        print(f"Failed requests:     {self.failed_requests}")
        if self.errors:
            print(f"Errors:")
            for err, count in self.errors.items():
                print(f"  {err}: {count}")
        print(f"{'='*50}\n")