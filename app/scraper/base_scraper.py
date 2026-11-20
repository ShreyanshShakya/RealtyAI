"""Base scraper class with common functionality."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import random
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from app.config import get_settings
from app.utils.browser import get_browser_pool, close_browser_pool
from app.utils.url import canonicalize_url, is_same_domain
from app.data.jsonld import extract_all_jsonld, extract_property_data
from app.types import ScrapedRecord, ScrapeReport

settings = get_settings()


class FetchResult:
    """Result of a fetch attempt"""
    def __init__(self, soup: Optional[BeautifulSoup] = None, error: Optional[Exception] = None, 
                 method: str = "http", status_code: Optional[int] = None):
        self.soup = soup
        self.error = error
        self.method = method
        self.status_code = status_code
    
    @property
    def success(self) -> bool:
        return self.soup is not None


class BaseScraper(ABC):
    """Abstract base scraper with common functionality"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.session = self._create_session()
        self.records: List[ScrapedRecord] = []
        self.seen_urls: set = set()
        self.report = ScrapeReport(source=source_name)
        self._browser_pool = None
        self._use_playwright = False
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": settings.USER_AGENT})
        return session
    
    def enable_playwright(self):
        """Enable Playwright for JavaScript-rendered content"""
        self._use_playwright = True
        self._browser_pool = get_browser_pool(headless=settings.PLAYWRIGHT_HEADLESS)
    
    def close_playwright(self):
        """Close Playwright browser pool"""
        if self._browser_pool:
            close_browser_pool()
            self._browser_pool = None
    
    def _apply_rate_limit(self):
        """Apply configured delay with jitter between requests"""
        delay = settings.REQUEST_DELAY + random.uniform(0, settings.REQUEST_JITTER)
        time.sleep(delay)
    
    def _fetch_xml(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse XML content (for sitemaps)"""
        try:
            logger.debug(f"Fetching XML: {url}")
            resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
            resp.raise_for_status()
            self._apply_rate_limit()
            return BeautifulSoup(resp.content, "xml")
        except Exception as e:
            logger.warning(f"XML fetch failed for {url}: {e}")
            return None
    
    def _fetch_with_retry(self, url: str) -> 'FetchResult':
        """Fetch with retry logic for transient errors"""
        last_exception = None
        
        for attempt in range(settings.MAX_RETRIES):
            try:
                logger.debug(f"Fetching (attempt {attempt + 1}/{settings.MAX_RETRIES}): {url}")
                resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
                
                # Check for access denied / blocking
                if resp.status_code in (403, 429, 503):
                    logger.warning(f"Access denied or rate limited ({resp.status_code}): {url}")
                    return FetchResult(error=Exception(f"HTTP {resp.status_code}"), 
                                       status_code=resp.status_code, method="http")
                
                resp.raise_for_status()
                
                # Check for empty or minimal content
                if not resp.content or len(resp.content) < 100:
                    logger.warning(f"Empty or minimal response: {url}")
                    return FetchResult(error=Exception("Empty response"), method="http")
                
                self._apply_rate_limit()
                return FetchResult(soup=BeautifulSoup(resp.content, "lxml"), method="http", 
                                   status_code=resp.status_code)
                
            except requests.exceptions.Timeout:
                last_exception = Exception(f"Timeout after {settings.REQUEST_TIMEOUT}s")
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                last_exception = Exception(f"Connection error: {e}")
                logger.warning(f"Connection error for {url} (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                last_exception = e
                logger.warning(f"HTTP error for {url}: {e}")
                break  # Don't retry permanent HTTP errors
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request error for {url} (attempt {attempt + 1}): {e}")
            except Exception as e:
                # Catch decoding errors (e.g., Brotli)
                last_exception = e
                logger.warning(f"Decoding/parsing error for {url} (attempt {attempt + 1}): {e}")
            
            # Exponential backoff with jitter
            if attempt < settings.MAX_RETRIES - 1:
                wait_time = min(2 ** attempt + random.uniform(0, 1), 30)
                logger.debug(f"Waiting {wait_time:.1f}s before retry")
                time.sleep(wait_time)
        
        return FetchResult(error=last_exception or Exception("Max retries exceeded"), method="http")
    
    def fetch(self, url: str, use_xml: bool = False) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML/XML with retry logic and fallback"""
        # Check robots.txt if checker is available
        if hasattr(self, 'robots_checker') and self.robots_checker:
            if not self.robots_checker.can_fetch("*", url):
                logger.warning(f"Robots.txt disallows: {url}")
                self.report.failed_requests += 1
                return None
        
        # For XML content (sitemaps), use dedicated XML fetcher
        if use_xml:
            xml_soup = self._fetch_xml(url)
            if xml_soup:
                self.report.pages_fetched += 1
                return xml_soup
            logger.warning(f"XML fetch failed for {url}, trying HTML parser")
        
        # Try HTTP first
        result = self._fetch_with_retry(url)
        
        if result.success:
            self.report.pages_fetched += 1
            return result.soup
        
        # Log the failure
        logger.warning(f"HTTP fetch failed for {url}: {result.error}")
        
        # Try Playwright fallback if enabled and HTTP failed
        if self._use_playwright:
            logger.info(f"Attempting Playwright fallback for: {url}")
            try:
                if not self._browser_pool:
                    self.enable_playwright()
                
                html = self._browser_pool.fetch(url, wait_for=None, timeout=settings.PLAYWRIGHT_TIMEOUT)
                if html and len(html) > 100:
                    self._apply_rate_limit()
                    self.report.pages_fetched += 1
                    logger.info(f"Playwright fallback succeeded for: {url}")
                    return BeautifulSoup(html, "lxml")
                else:
                    logger.warning(f"Playwright returned empty content for: {url}")
            except Exception as e:
                logger.warning(f"Playwright fallback failed for {url}: {e}")
        
        # Record failure
        self.report.failed_requests += 1
        err_type = type(result.error).__name__ if result.error else "UnknownError"
        self.report.errors[err_type] = self.report.errors.get(err_type, 0) + 1
        return None
    
    def fetch_with_playwright(self, url: str, wait_for: str = None) -> Optional[BeautifulSoup]:
        """Fetch using Playwright for JS-rendered content (direct call)"""
        if not self._browser_pool:
            self.enable_playwright()
        
        try:
            html = self._browser_pool.fetch(url, wait_for=wait_for, timeout=settings.PLAYWRIGHT_TIMEOUT)
            if html:
                self._apply_rate_limit()
                return BeautifulSoup(html, "lxml")
        except Exception as e:
            logger.warning(f"Playwright failed for {url}: {e}")
        return None
    
    def _apply_rate_limit(self):
        """Apply configured delay with jitter between requests"""
        delay = settings.REQUEST_DELAY + random.uniform(0, settings.REQUEST_JITTER)
        time.sleep(delay)
    
    def deduplicate(self, url: str) -> bool:
        """Check and record URL to prevent duplicates (canonicalized)"""
        canon_url = canonicalize_url(url, self.base_url if hasattr(self, 'base_url') else None)
        if canon_url in self.seen_urls:
            return False
        self.seen_urls.add(canon_url)
        return True
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the source"""
        pass
    
    @abstractmethod
    def discover_urls(self) -> List[str]:
        """Return list of project/listing URLs to scrape"""
        pass
    
    @abstractmethod
    def parse_detail(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract structured data from detail page"""
        pass
    
    def _extract_jsonld(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract property data from JSON-LD"""
        jsonld_objects = extract_all_jsonld(soup)
        return extract_property_data(jsonld_objects)
    
    def scrape(self) -> List[ScrapedRecord]:
        """Main scraping workflow"""
        logger.info(f"Starting {self.source_name} scrape")
        urls = self.discover_urls()
        self.report.pages_discovered = len(urls)
        logger.info(f"Discovered {len(urls)} URLs")
        
        for url in urls:
            if not self.deduplicate(url):
                logger.debug(f"Skipping duplicate: {url}")
                continue
            
            try:
                # Try HTML first, then XML for sitemaps
                soup = self.fetch(url)
                if not soup and self._use_playwright:
                    soup = self.fetch_with_playwright(url)
                if not soup:
                    # Failure already logged in fetch()
                    continue
                
                data = self.parse_detail(soup, url)
                if data:
                    self.report.pages_parsed += 1
                    self.records.append(ScrapedRecord(
                        source=self.source_name,
                        source_url=url,
                        raw_data=data
                    ))
                    logger.debug(f"Scraped: {url}")
                else:
                    logger.warning(f"Failed to parse: {url}")
                    
            except Exception as e:
                self.report.failed_requests += 1
                err_type = type(e).__name__
                self.report.errors[err_type] = self.report.errors.get(err_type, 0) + 1
                logger.error(f"Error scraping {url}: {e}")
        
        logger.info(f"Completed {self.source_name}: {len(self.records)} records")
        return self.records
    
    def save_jsonl(self, filepath: str) -> None:
        """Save records as JSONL"""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps({
                    "source": record.source,
                    "source_url": record.source_url,
                    **record.raw_data
                }, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.records)} records to {filepath}")