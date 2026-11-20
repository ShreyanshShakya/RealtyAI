"""DarGlobal scraper implementation."""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

from loguru import logger
from app.scraper.base_scraper import BaseScraper
from app.config import get_settings
from app.utils.robots import RobotsChecker

settings = get_settings()


class DarGlobalScraper(BaseScraper):
    def __init__(self, use_playwright: bool = True):
        super().__init__("DarGlobal")
        self.robots_checker = RobotsChecker(settings.DARGLOBAL_BASE_URL)
        if use_playwright:
            self.enable_playwright()
    
    @property
    def base_url(self) -> str:
        return settings.DARGLOBAL_BASE_URL
    
    def discover_urls(self) -> List[str]:
        """Find all project URLs from sitemap"""
        urls = []
        sitemap_url = urljoin(self.base_url, "/sitemap.xml")
        
        # Use XML parser for sitemap
        soup = self.fetch(sitemap_url, use_xml=True)
        if not soup:
            return urls
        
        # Case-insensitive exclude patterns for non-project pages
        exclude_patterns = [
            "/about", "/blog", "/press", "/partners", "/careers", "/faq",
            "/contact", "/privacy", "/terms", "/investor", "/commercial",
            "/hospitality", "/development-management", "/why-invest",
            "/get-in-touch", "/bookyourunit", "/pay-online", "/success",
            "/thank-you", "/tokenization", "/landing", "/internship",
            "/exclusive-membership", "/dg-circle", "/insignia", "/campaigns",
            "/become-a-broker", "/become-an-agent", "/win-a-trip",
            "/exclusive-plots", "/invest-crypto", "/broker-contest",
            "/amour-sans-detour", "/great-escape", "/pre-launch",
            "/investment-catalogue", "/new-home-page", "/live-all-in",
            "/luxury-golf-communities", "/sohar-islamic",
            "/insights", "/cryptocurrency", "/crypto",
        ]
        
        # Normalize exclude patterns to lowercase for case-insensitive matching
        exclude_patterns_lower = [p.lower() for p in exclude_patterns]
        
        urls_elem = soup.find_all("url")
        for url_elem in urls_elem:
            loc = url_elem.find("loc")
            if loc and loc.text:
                url = loc.text.strip()
                parsed = urlparse(url)
                path = parsed.path
                path_lower = path.lower()
                
                # Skip root and known non-project pages (case-insensitive)
                if path in ("/", "") or any(path_lower.startswith(p) for p in exclude_patterns_lower):
                    continue
                
                # Skip URLs with query parameters (usually tracking)
                if parsed.query:
                    continue
                
                # Project URLs are typically flat slugs with hyphens
                # Heuristic: path has hyphens and is not a known section
                if path.count("-") >= 1 and len(path) > 5:
                    urls.append(url)
        
        return urls
    
    def parse_detail(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract project details from project page"""
        try:
            data = {
                "title": "",
                "property_type": None,
                "country": None,
                "city": None,
                "description": None,
                "amenities": [],
                "status": None,
                "price": None,
                "currency": None,
                "image_url": None,
            }
            
            # Title - try multiple selectors
            title_elem = soup.select_one("h1.project-title, h1.entry-title, .project-header h1, main h1, h1")
            if title_elem:
                data["title"] = title_elem.get_text(strip=True)
            
            # Description
            desc_elem = soup.select_one(".project-description, .entry-content, .project-details, main article, .description, .content")
            if desc_elem:
                data["description"] = desc_elem.get_text(strip=True)[:5000]
            
            # Location - look for country/city in meta or content
            location_elem = soup.select_one(".project-location, .location, [itemprop='address'], .address, .project-meta")
            if location_elem:
                loc_text = location_elem.get_text(strip=True)
                parts = [p.strip() for p in loc_text.split(",")]
                if len(parts) >= 2:
                    data["city"] = parts[0]
                    data["country"] = parts[1]
                elif len(parts) == 1:
                    data["city"] = parts[0]
            
            # Property type
            type_elem = soup.select_one(".property-type, [itemprop='propertyType'], [class*='type'], .project-category")
            if type_elem:
                data["property_type"] = type_elem.get_text(strip=True)
            
            # Amenities
            amenity_elems = soup.select(".amenities li, .features li, .project-amenities li, [class*='amenit'] li, .facilities li")
            data["amenities"] = [a.get_text(strip=True) for a in amenity_elems if a.get_text(strip=True)]
            
            # Status
            status_elem = soup.select_one(".project-status, .status, [itemprop='status'], [class*='status']")
            if status_elem:
                data["status"] = status_elem.get_text(strip=True)
            
            # Price - if publicly disclosed
            price_elem = soup.select_one(".price, .project-price, [itemprop='price'], [class*='price']")
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                match = re.search(r"([\d,]+\.?\d*)\s*([A-Z]{3}|[£$€])", price_text)
                if match:
                    data["price"] = float(match.group(1).replace(",", ""))
                    data["currency"] = match.group(2)
            
            # Image
            img_elem = soup.select_one(".project-image img, .hero img, [itemprop='image'], [class*='image'] img, .banner img")
            if img_elem:
                src = img_elem.get("src") or img_elem.get("data-src")
                if src:
                    data["image_url"] = urljoin(self.base_url, src)
            
            # Extract JSON-LD data as supplement
            jsonld_data = self._extract_jsonld(soup)
            for key, value in jsonld_data.items():
                if value and not data.get(key):
                    data[key] = value
            
            # Validate minimum required fields
            if not data["title"]:
                logger.warning(f"No title found for {url}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Parse error for {url}: {e}")
            return None