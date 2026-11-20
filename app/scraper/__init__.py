"""Scraper package exports."""
from app.scraper.base_scraper import BaseScraper
from app.scraper.darglobal import DarGlobalScraper
from app.scraper.wasalt import WasaltScraper
from app.types import ScrapedRecord, ScrapeReport

__all__ = [
    "BaseScraper",
    "ScrapedRecord",
    "ScrapeReport",
    "DarGlobalScraper",
    "WasaltScraper",
]