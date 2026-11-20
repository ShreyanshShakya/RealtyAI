"""Wasalt scraper implementation."""
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import re
import json
import gzip

from loguru import logger
from app.scraper.base_scraper import BaseScraper
from app.config import get_settings
from app.utils.robots import RobotsChecker

settings = get_settings()

# Try to import brotli for Wasalt
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


class WasaltScraper(BaseScraper):
    def __init__(self, max_pages: int = 30):
        super().__init__("Wasalt")
        self.max_pages = max_pages
        self.robots_checker = RobotsChecker(settings.WASALT_BASE_URL)
        # Enable Playwright for Wasalt due to Brotli encoding and potential JS rendering
        self.enable_playwright()
    
    @property
    def base_url(self) -> str:
        return settings.WASALT_BASE_URL
    
    def discover_urls(self) -> List[str]:
        """Find all property listing URLs from sitemap"""
        urls = []
        
        # Wasalt sitemap URLs from robots.txt (try both .gz and non-.gz)
        sitemap_urls = [
            "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_ar_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_ar_sa.xml.gz",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml",
            "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml",
            "https://cdn.wasalt.sa/sitemap/product_sitemap_ar_sa.xml",
            "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_ar_sa.xml",
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                resp = self.session.get(sitemap_url, timeout=settings.REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
                
                # Try to decompress gzip, fall back to raw content
                content = resp.content
                if sitemap_url.endswith(".gz"):
                    try:
                        content = gzip.decompress(content)
                    except gzip.BadGzipFile:
                        pass  # Not actually gzipped
                
                soup = BeautifulSoup(content, "xml")
                urls_elem = soup.find_all("url")
                
                for url_elem in urls_elem:
                    loc = url_elem.find("loc")
                    if loc and loc.text:
                        url = loc.text.strip()
                        # Filter for property listing URLs
                        if "/property/" in url and ("/sale/" in url or "/rent/" in url):
                            urls.append(url)
                            
            except Exception as e:
                logger.warning(f"Failed to parse sitemap {sitemap_url}: {e}")
        
        # Deduplicate and limit
        urls = list(dict.fromkeys(urls))
        return urls[:self.max_pages]
    
    def _fetch_with_brotli(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL handling Brotli encoding"""
        try:
            resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None
            
            content = resp.content
            encoding = resp.headers.get("Content-Encoding", "").lower()
            
            if "br" in encoding and HAS_BROTLI:
                content = brotli.decompress(content)
            elif "gzip" in encoding:
                content = gzip.decompress(content)
            
            return BeautifulSoup(content, "lxml")
        except Exception as e:
            logger.debug(f"Brotli fetch failed for {url}: {e}")
            return None
    
    def parse_detail(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        """Extract property listing details"""
        try:
            data = {
                "title": "",
                "property_type": None,
                "transaction_type": None,
                "country": None,  # Don't hardcode - extract from page
                "city": None,
                "district": None,
                "price": None,
                "currency": None,  # Don't hardcode - extract from page
                "bedrooms": None,
                "bathrooms": None,
                "area": None,
                "area_unit": None,
                "area_sqm": None,
                "description": None,
                "amenities": [],
                "status": None,  # Don't hardcode - extract from page
                "image_url": None,
            }
            
            # Title
            title_elem = soup.select_one("h1.listing-title, h1.property-title, .property-header h1, h1, [data-testid='listing-title']")
            if title_elem:
                data["title"] = title_elem.get_text(strip=True)
            
            # Price
            price_elem = soup.select_one(".price, .property-price, [itemprop='price'], [class*='price'], [data-testid='price']")
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                match = re.search(r"([\d,]+\.?\d*)\s*(SAR|USD|AED)?", price_text, re.IGNORECASE)
                if match:
                    data["price"] = float(match.group(1).replace(",", ""))
                    if match.group(2):
                        data["currency"] = match.group(2).upper()
            
            # Transaction type (Sale/Rent)
            trans_elem = soup.select_one(".transaction-type, .listing-type, [itemprop='businessFunction'], [class*='type'], [data-testid='transaction-type']")
            if trans_elem:
                text = trans_elem.get_text(strip=True).lower()
                if "rent" in text or "leasing" in text or "lease" in text:
                    data["transaction_type"] = "Rent"
                elif "sale" in text or "buy" in text or "purchase" in text:
                    data["transaction_type"] = "Sale"
            
            # Also infer from URL
            if "/sale/" in url:
                data["transaction_type"] = "Sale"
            elif "/rent/" in url:
                data["transaction_type"] = "Rent"
            
            # Location
            loc_elem = soup.select_one(".location, .property-location, [itemprop='address'], .address, [class*='location'], [data-testid='location']")
            if loc_elem:
                loc_text = loc_elem.get_text(strip=True)
                parts = [p.strip() for p in loc_text.split(",")]
                if len(parts) >= 2:
                    data["district"] = parts[0]
                    data["city"] = parts[1]
                elif len(parts) == 1:
                    data["city"] = parts[0]
            
            # Property details - bedrooms, bathrooms, area
            details = soup.select(".property-details li, .features li, .specs li, [class*='detail'] li, [class*='feature'] li, [data-testid*='detail']")
            for detail in details:
                text = detail.get_text(strip=True).lower()
                if "bed" in text:
                    match = re.search(r"(\d+)", text)
                    if match:
                        data["bedrooms"] = int(match.group(1))
                elif "bath" in text:
                    match = re.search(r"(\d+)", text)
                    if match:
                        data["bathrooms"] = int(match.group(1))
                elif "area" in text or "sqm" in text or "m²" in text or "sq ft" in text or "sqft" in text:
                    match = re.search(r"([\d,]+\.?\d*)", text)
                    if match:
                        data["area"] = float(match.group(1).replace(",", ""))
                        # Determine unit from text
                        if "sq ft" in text or "sqft" in text:
                            data["area_unit"] = "sqft"
                            data["area_sqm"] = round(data["area"] * 0.092903, 2)
                        else:
                            data["area_unit"] = "sqm"
                            data["area_sqm"] = data["area"]
            
            # Property type
            type_elem = soup.select_one(".property-type, [itemprop='propertyType'], [class*='type'], [data-testid='property-type']")
            if type_elem:
                data["property_type"] = type_elem.get_text(strip=True)
            elif "apartment" in data["title"].lower():
                data["property_type"] = "Apartment"
            elif "villa" in data["title"].lower():
                data["property_type"] = "Villa"
            elif "land" in data["title"].lower() or "plot" in data["title"].lower():
                data["property_type"] = "Land"
            
            # Description
            desc_elem = soup.select_one(".description, .property-description, [itemprop='description'], [class*='description'], [data-testid='description']")
            if desc_elem:
                data["description"] = desc_elem.get_text(strip=True)[:5000]
            
            # Amenities
            amenity_elems = soup.select(".amenities li, .features li, .amenities-list li, [class*='amenit'] li, [data-testid*='amenity']")
            data["amenities"] = [a.get_text(strip=True) for a in amenity_elems if a.get_text(strip=True)]
            
            # Images
            img_elem = soup.select_one(".property-gallery img, .listing-images img, [itemprop='image'], [class*='image'] img, [data-testid='image']")
            if img_elem:
                src = img_elem.get("src") or img_elem.get("data-src")
                if src:
                    data["image_url"] = urljoin(self.base_url, src)
            
            # Try to extract from JSON-LD structured data
            jsonld_objects = soup.select("script[type='application/ld+json']")
            for script in jsonld_objects:
                if script.string:
                    try:
                        ld_data = json.loads(script.string)
                        self._extract_from_json_ld(ld_data, data)
                    except:
                        pass
            
            # Extract JSON-LD data as supplement
            jsonld_data = self._extract_jsonld(soup)
            for key, value in jsonld_data.items():
                if value and not data.get(key):
                    data[key] = value
            
            if not data["title"]:
                logger.warning(f"No title found for {url}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Parse error for {url}: {e}")
            return None
    
    def _extract_from_json_ld(self, ld_data: Any, data: Dict) -> None:
        """Extract additional fields from JSON-LD structured data"""
        if isinstance(ld_data, list):
            for item in ld_data:
                self._extract_from_json_ld(item, data)
            return
        
        if not isinstance(ld_data, dict):
            return
        
        # Handle RealEstateListing schema
        if ld_data.get("@type") in ["RealEstateListing", "Product"]:
            if "name" in ld_data and not data["title"]:
                data["title"] = ld_data["name"]
            if "description" in ld_data and not data["description"]:
                data["description"] = ld_data["description"]
            if "offers" in ld_data:
                offers = ld_data["offers"]
                if isinstance(offers, dict):
                    if "price" in offers and not data["price"]:
                        data["price"] = float(offers["price"])
                    if "priceCurrency" in offers:
                        data["currency"] = offers["priceCurrency"]
            if "address" in ld_data:
                addr = ld_data["address"]
                if isinstance(addr, dict):
                    if "addressLocality" in addr and not data["city"]:
                        data["city"] = addr["addressLocality"]
                    if "addressRegion" in addr and not data["district"]:
                        data["district"] = addr["addressRegion"]
                    if "addressCountry" in addr and not data["country"]:
                        data["country"] = addr["addressCountry"]