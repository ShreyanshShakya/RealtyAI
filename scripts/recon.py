#!/usr/bin/env python
"""Website reconnaissance for RealtyAI scrapers.

Inspects target websites to discover actual structure, URL patterns,
JSON-LD schemas, and JavaScript requirements before implementing scrapers.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Set
from collections import Counter

import requests
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.utils.url import canonicalize_url, is_same_domain
from app.utils.robots import RobotsChecker
from app.data.jsonld import extract_all_jsonld, extract_property_data

settings = get_settings()


@dataclass
class ReconReport:
    source: str
    base_url: str
    timestamp: str
    
    homepage_accessible: bool = False
    homepage_status_code: int = 0
    homepage_response_time_ms: int = 0
    
    robots_txt_accessible: bool = False
    robots_txt_content: str = ""
    crawl_allowed: bool = True
    crawl_delay: Optional[float] = None
    disallowed_paths: List[str] = field(default_factory=list)
    sitemap_urls: List[str] = field(default_factory=list)
    
    sitemap_accessible: bool = False
    sitemap_urls_found: List[str] = field(default_factory=list)
    sitemap_type: str = ""
    
    discovered_urls: List[str] = field(default_factory=list)
    url_patterns: Dict[str, int] = field(default_factory=dict)
    pagination_detected: bool = False
    pagination_pattern: str = ""
    sample_project_urls: List[str] = field(default_factory=list)
    
    sample_pages_analyzed: int = 0
    requires_javascript: bool = False
    javascript_indicators: List[str] = field(default_factory=list)
    css_selectors_found: Dict[str, List[str]] = field(default_factory=dict)
    
    jsonld_schemas_found: List[str] = field(default_factory=list)
    jsonld_sample: List[Dict[str, Any]] = field(default_factory=list)
    property_jsonld_found: bool = False
    
    content_language: str = ""
    charset: str = ""
    
    errors: List[str] = field(default_factory=list)


class WebsiteRecon:
    def __init__(self, source: str, base_url: str, max_pages: int = 5):
        self.source = source
        self.base_url = base_url
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.USER_AGENT})
        self.report = ReconReport(
            source=source,
            base_url=base_url,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self.robots_checker = RobotsChecker(base_url)
        self.visited_urls: Set[str] = set()
    
    def run(self) -> ReconReport:
        logger.info(f"Starting reconnaissance for {self.source} ({self.base_url})")
        
        self._check_homepage()
        self._check_robots_txt()
        self._check_sitemap()
        self._discover_urls()
        self._analyze_sample_pages()
        self._generate_summary()
        
        return self.report
    
    def _check_homepage(self):
        try:
            start = time.time()
            resp = self.session.get(self.base_url, timeout=settings.REQUEST_TIMEOUT)
            elapsed = int((time.time() - start) * 1000)
            
            self.report.homepage_status_code = resp.status_code
            self.report.homepage_response_time_ms = elapsed
            self.report.homepage_accessible = resp.status_code == 200
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "lxml")
                self.report.content_language = soup.html.get("lang", "") if soup.html else ""
                self.report.charset = resp.encoding or ""
                self._detect_javascript_requirements(soup, resp.text)
                
        except Exception as e:
            self.report.errors.append(f"Homepage check failed: {e}")
            logger.error(f"Homepage check failed: {e}")
    
    def _detect_javascript_requirements(self, soup: BeautifulSoup, html: str):
        indicators = []
        
        if "react" in html.lower() or "next.js" in html.lower() or "__NEXT_DATA__" in html:
            indicators.append("React/Next.js detected")
        if "vue" in html.lower() or "__VUE__" in html:
            indicators.append("Vue.js detected")
        if "angular" in html.lower():
            indicators.append("Angular detected")
        
        main_content = soup.select_one("main, #main, .main, #content, .content")
        if main_content and len(main_content.get_text(strip=True)) < 100:
            indicators.append("Main content container nearly empty (likely JS-rendered)")
        
        if soup.select(".skeleton, .placeholder, [data-loading], .loading"):
            indicators.append("Loading placeholders detected")
        
        if "enable javascript" in html.lower() or "javascript required" in html.lower():
            indicators.append("Explicit JavaScript requirement message")
        
        self.report.javascript_indicators = indicators
        self.report.requires_javascript = len(indicators) > 0
    
    def _check_robots_txt(self):
        try:
            # Use the RobotsChecker which has proper caching
            if self.robots_checker.fetch_and_parse():
                self.report.robots_txt_accessible = True
                self.report.robots_txt_content = self.robots_checker.parser._rules.get("*", {}).get("disallow", [])[:5000] if self.robots_checker.parser._rules else ""
                self.report.crawl_allowed = self.robots_checker.can_fetch("*", self.base_url)
                self.report.crawl_delay = self.robots_checker.crawl_delay("*")
                self.report.disallowed_paths = self.robots_checker.disallowed_paths("*")
                self.report.sitemap_urls = self.robots_checker.sitemaps()
                
        except Exception as e:
            self.report.errors.append(f"robots.txt check failed: {e}")
            logger.warning(f"robots.txt check failed: {e}")
    
    def _check_sitemap(self):
        sitemap_urls = self.report.sitemap_urls.copy()
        
        common_sitemaps = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap/sitemap.xml",
        ]
        
        for path in common_sitemaps:
            url = urljoin(self.base_url, path)
            if url not in sitemap_urls:
                sitemap_urls.append(url)
        
        all_urls = []
        for sitemap_url in sitemap_urls:
            try:
                resp = self.session.get(sitemap_url, timeout=settings.REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    self.report.sitemap_accessible = True
                    self.report.sitemap_urls_found.append(sitemap_url)
                    
                    if "index" in sitemap_url:
                        self.report.sitemap_type = "sitemap_index"
                    else:
                        self.report.sitemap_type = "sitemap"
                    
                    soup = BeautifulSoup(resp.content, "xml")
                    urls = soup.find_all("url")
                    for url_elem in urls:
                        loc = url_elem.find("loc")
                        if loc and loc.text:
                            all_urls.append(loc.text.strip())
                    
                    sitemaps = soup.find_all("sitemap")
                    for sm in sitemaps:
                        loc = sm.find("loc")
                        if loc and loc.text:
                            sub_url = loc.text.strip()
                            if sub_url not in self.report.sitemap_urls_found:
                                self.report.sitemap_urls_found.append(sub_url)
                                try:
                                    sub_resp = self.session.get(sub_url, timeout=settings.REQUEST_TIMEOUT)
                                    if sub_resp.status_code == 200:
                                        sub_soup = BeautifulSoup(sub_resp.content, "xml")
                                        sub_urls = sub_soup.find_all("url")
                                        for u in sub_urls:
                                            loc = u.find("loc")
                                            if loc and loc.text:
                                                all_urls.append(loc.text.strip())
                                except:
                                    pass
                    
            except Exception as e:
                logger.debug(f"Sitemap check failed for {sitemap_url}: {e}")
        
        for url in all_urls:
            if is_same_domain(url, urlparse(self.base_url).netloc):
                self.report.discovered_urls.append(canonicalize_url(url, self.base_url))
        
        self.report.discovered_urls = list(dict.fromkeys(self.report.discovered_urls))
    
    def _discover_urls(self):
        if not self.report.homepage_accessible:
            return
        
        try:
            resp = self.session.get(self.base_url, timeout=settings.REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.content, "lxml")
            
            all_links = soup.find_all("a", href=True)
            
            patterns = Counter()
            project_urls = []
            
            for link in all_links:
                href = link.get("href", "")
                if not href:
                    continue
                
                full_url = urljoin(self.base_url, href)
                canon_url = canonicalize_url(full_url, self.base_url)
                
                if not is_same_domain(canon_url, urlparse(self.base_url).netloc):
                    continue
                
                parsed = urlparse(canon_url)
                path = parsed.path
                
                pattern = self._generalize_path(path)
                patterns[pattern] += 1
                
                if self._is_likely_content_url(path):
                    project_urls.append(canon_url)
            
            self.report.url_patterns = dict(patterns.most_common(20))
            
            project_urls = list(dict.fromkeys(project_urls))
            self.report.sample_project_urls = project_urls[:20]
            
            self._detect_pagination(soup)
            
        except Exception as e:
            self.report.errors.append(f"URL discovery failed: {e}")
            logger.error(f"URL discovery failed: {e}")
    
    def _generalize_path(self, path: str) -> str:
        import re
        path = re.sub(r'/\d+', '/{id}', path)
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path, flags=re.IGNORECASE)
        path = re.sub(r'/[a-z0-9-]{20,}', '/{slug}', path)
        return path
    
    def _is_likely_content_url(self, path: str) -> bool:
        path_lower = path.lower()
        content_indicators = [
            "/project", "/property", "/listing", "/development",
            "/projects/", "/properties/", "/listings/", "/developments/"
        ]
        has_indicator = any(ind in path_lower for ind in content_indicators)
        is_listing_page = path_lower.rstrip("/") in ["/projects", "/properties", "/listings", "/developments"]
        is_pagination = "page=" in path_lower or "/page/" in path_lower
        return has_indicator and not is_listing_page and not is_pagination
    
    def _detect_pagination(self, soup: BeautifulSoup):
        pagination_selectors = [
            ".pagination a", ".pager a", ".page-numbers a",
            "[class*='pagination'] a", "[class*='pager'] a",
            "nav[aria-label='pagination'] a"
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            if links:
                self.report.pagination_detected = True
                for link in links[:3]:
                    href = link.get("href", "")
                    if href:
                        self.report.pagination_pattern = href
                        break
                break
    
    def _analyze_sample_pages(self):
        urls_to_analyze = self.report.sample_project_urls[:self.max_pages]
        
        if not urls_to_analyze:
            common_paths = self._get_common_paths()
            for path in common_paths:
                url = urljoin(self.base_url, path)
                if is_same_domain(url, urlparse(self.base_url).netloc):
                    urls_to_analyze.append(url)
        
        all_selectors = Counter()
        all_schemas = Counter()
        jsonld_samples = []
        
        for url in urls_to_analyze:
            if url in self.visited_urls:
                continue
            self.visited_urls.add(url)
            
            try:
                resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
                
                self.report.sample_pages_analyzed += 1
                soup = BeautifulSoup(resp.content, "lxml")
                
                selectors = self._extract_key_selectors(soup)
                for key, vals in selectors.items():
                    all_selectors[key].update(vals)
                
                jsonld_objects = extract_all_jsonld(soup)
                if jsonld_objects:
                    property_data = extract_property_data(jsonld_objects)
                    if any(property_data.values()):
                        self.report.property_jsonld_found = True
                    
                    for obj in jsonld_objects:
                        obj_type = obj.get("@type", "")
                        if isinstance(obj_type, list):
                            for t in obj_type:
                                all_schemas[t] += 1
                        else:
                            all_schemas[obj_type] += 1
                    
                    if len(jsonld_samples) < 2:
                        jsonld_samples.append({
                            "url": url,
                            "schemas": [obj.get("@type", "") for obj in jsonld_objects],
                            "property_data": property_data
                        })
                
                self._detect_javascript_requirements(soup, resp.text)
                
            except Exception as e:
                logger.debug(f"Failed to analyze {url}: {e}")
        
        self.report.css_selectors_found = {
            k: [v for v, _ in vals.most_common(10)]
            for k, vals in all_selectors.items()
        }
        self.report.jsonld_schemas_found = list(all_schemas.keys())
        self.report.jsonld_sample = jsonld_samples
    
    def _get_common_paths(self) -> List[str]:
        if self.source == "DarGlobal":
            return [
                "/projects/", "/developments/", "/our-projects/",
                "/projects", "/developments"
            ]
        elif self.source == "Wasalt":
            return [
                "/en/properties", "/en/properties/", "/properties",
                "/en/listings", "/listings", "/en/buy", "/en/rent"
            ]
        return []
    
    def _extract_key_selectors(self, soup: BeautifulSoup) -> Dict[str, Counter]:
        selectors = {
            "title": Counter(),
            "price": Counter(),
            "location": Counter(),
            "description": Counter(),
            "amenities": Counter(),
            "images": Counter(),
            "property_type": Counter(),
            "bedrooms": Counter(),
            "bathrooms": Counter(),
            "area": Counter(),
            "status": Counter(),
        }
        
        for sel in ["h1", "h1[class*='title']", "[class*='title'] h1", ".entry-title", ".project-title", ".listing-title", ".property-title"]:
            elems = soup.select(sel)
            if elems:
                selectors["title"][sel] += len(elems)
        
        for sel in ["[class*='price']", "[itemprop='price']", ".price", "[class*='Price']"]:
            elems = soup.select(sel)
            if elems:
                selectors["price"][sel] += len(elems)
        
        for sel in ["[class*='location']", "[itemprop='address']", ".location", ".address", "[class*='Location']"]:
            elems = soup.select(sel)
            if elems:
                selectors["location"][sel] += len(elems)
        
        for sel in ["[class*='description']", "[itemprop='description']", ".description", "[class*='Description']", "article", ".content"]:
            elems = soup.select(sel)
            if elems:
                selectors["description"][sel] += len(elems)
        
        for sel in ["[class*='amenit'] li", "[class*='feature'] li", "[class*='spec'] li", ".amenities li", ".features li"]:
            elems = soup.select(sel)
            if elems:
                selectors["amenities"][sel] += len(elems)
        
        for sel in ["[class*='gallery'] img", "[class*='image'] img", "[itemprop='image']", ".hero img", ".main-image img"]:
            elems = soup.select(sel)
            if elems:
                selectors["images"][sel] += len(elems)
        
        detail_selectors = {
            "property_type": ["[class*='type']", "[itemprop='propertyType']", "[class*='Type']"],
            "bedrooms": ["[class*='bed']", "[class*='Bed']"],
            "bathrooms": ["[class*='bath']", "[class*='Bath']"],
            "area": ["[class*='area']", "[class*='size']", "[class*='Area']", "[class*='Size']"],
            "status": ["[class*='status']", "[itemprop='status']", "[class*='Status']"],
        }
        
        for key, sels in detail_selectors.items():
            for sel in sels:
                elems = soup.select(sel)
                if elems:
                    selectors[key][sel] += len(elems)
        
        return selectors
    
    def _generate_summary(self):
        logger.info(f"Reconnaissance complete for {self.source}")
        logger.info(f"  Homepage accessible: {self.report.homepage_accessible}")
        logger.info(f"  Robots.txt accessible: {self.report.robots_txt_accessible}")
        logger.info(f"  Crawl allowed: {self.report.crawl_allowed}")
        logger.info(f"  URLs discovered: {len(self.report.discovered_urls)}")
        logger.info(f"  Sample project URLs: {len(self.report.sample_project_urls)}")
        logger.info(f"  Pages analyzed: {self.report.sample_pages_analyzed}")
        logger.info(f"  Requires JavaScript: {self.report.requires_javascript}")
        logger.info(f"  JSON-LD schemas: {self.report.jsonld_schemas_found}")
        logger.info(f"  Property JSON-LD found: {self.report.property_jsonld_found}")


def save_report(report: ReconReport, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{report.source.lower()}_recon.json"
    
    data = asdict(report)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Report saved to {output_file}")
    return output_file


def print_summary(report: ReconReport):
    check = "[OK]"
    cross = "[FAIL]"
    
    print(f"\n{'='*60}")
    print(f"RECONNAISSANCE REPORT: {report.source}")
    print(f"{'='*60}")
    print(f"Base URL:           {report.base_url}")
    print(f"Timestamp:          {report.timestamp}")
    print(f"\n--- Accessibility ---")
    print(f"Homepage:           {check if report.homepage_accessible else cross} (HTTP {report.homepage_status_code}, {report.homepage_response_time_ms}ms)")
    print(f"Robots.txt:         {check if report.robots_txt_accessible else cross}")
    print(f"Crawl allowed:      {check if report.crawl_allowed else cross}")
    if report.crawl_delay:
        print(f"Crawl delay:        {report.crawl_delay}s")
    print(f"Disallowed paths:   {len(report.disallowed_paths)}")
    print(f"\n--- Sitemap ---")
    print(f"Sitemap accessible: {check if report.sitemap_accessible else cross}")
    print(f"Sitemap URLs:       {len(report.sitemap_urls_found)}")
    print(f"Sitemap type:       {report.sitemap_type or 'N/A'}")
    print(f"\n--- URL Discovery ---")
    print(f"Total URLs found:   {len(report.discovered_urls)}")
    print(f"Sample content URLs: {len(report.sample_project_urls)}")
    print(f"Pagination detected: {check if report.pagination_detected else cross}")
    if report.pagination_pattern:
        print(f"Pagination pattern: {report.pagination_pattern}")
    print(f"\n--- Top URL Patterns ---")
    for pattern, count in list(report.url_patterns.items())[:10]:
        print(f"  {pattern}: {count}")
    print(f"\n--- Content Analysis ---")
    print(f"Pages analyzed:     {report.sample_pages_analyzed}")
    print(f"Requires JS:        {check if report.requires_javascript else cross}")
    if report.javascript_indicators:
        for ind in report.javascript_indicators:
            print(f"  - {ind}")
    print(f"Language:           {report.content_language or 'N/A'}")
    print(f"Charset:            {report.charset or 'N/A'}")
    print(f"\n--- JSON-LD ---")
    print(f"Schemas found:      {len(report.jsonld_schemas_found)}")
    for schema in report.jsonld_schemas_found[:10]:
        print(f"  - {schema}")
    print(f"Property JSON-LD:   {check if report.property_jsonld_found else cross}")
    if report.jsonld_sample:
        print(f"Sample extractions: {len(report.jsonld_sample)}")
    print(f"\n--- CSS Selectors (top 3 per category) ---")
    for category, selectors in report.css_selectors_found.items():
        if selectors:
            print(f"  {category}:")
            for sel in selectors[:3]:
                print(f"    - {sel}")
    if report.errors:
        print(f"\n--- Errors ---")
        for err in report.errors:
            print(f"  - {err}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Website reconnaissance for RealtyAI")
    parser.add_argument("--source", choices=["darglobal", "wasalt", "both"], default="both")
    parser.add_argument("--max-pages", type=int, default=5, help="Max sample pages to analyze")
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sources = []
    if args.source in ["darglobal", "both"]:
        sources.append(("DarGlobal", settings.DARGLOBAL_BASE_URL))
    if args.source in ["wasalt", "both"]:
        sources.append(("Wasalt", settings.WASALT_BASE_URL))
    
    for source_name, base_url in sources:
        recon = WebsiteRecon(source_name, base_url, max_pages=args.max_pages)
        report = recon.run()
        save_report(report, output_dir)
        print_summary(report)


if __name__ == "__main__":
    main()