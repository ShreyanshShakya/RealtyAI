"""Tests for scraper modules."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

from app.scraper.darglobal import DarGlobalScraper
from app.scraper.wasalt import WasaltScraper
from app.data.schema import PropertyRecord


# Sample HTML fixtures
DARGLOBAL_PROJECT_HTML = """
<html>
<body>
    <h1 class="project-title">DarGlobal Luxury Residences</h1>
    <div class="project-location">Riyadh, Saudi Arabia</div>
    <div class="project-description">Luxury residential development with premium amenities.</div>
    <ul class="amenities">
        <li>Swimming Pool</li>
        <li>Gym</li>
        <li>24/7 Security</li>
    </ul>
    <span class="project-status">Under Construction</span>
    <div class="price">2,500,000 SAR</div>
    <img class="project-image" src="/images/project1.jpg" />
</body>
</html>
"""

WASALT_LISTING_HTML = """
<html>
<body>
    <h1 class="listing-title">3BR Apartment in Al Rimal</h1>
    <div class="price">750,000 SAR</div>
    <span class="transaction-type">For Sale</span>
    <div class="location">Al Rimal, Riyadh</div>
    <ul class="property-details">
        <li>Bedrooms: 3</li>
        <li>Bathrooms: 2</li>
        <li>Area: 150 sqm</li>
    </ul>
    <div class="description">Spacious apartment with modern finishes.</div>
    <ul class="amenities">
        <li>Parking</li>
        <li>Elevator</li>
    </ul>
    <img class="property-gallery" src="/images/listing1.jpg" />
</body>
</html>
"""

DARGLOBAL_JSONLD_HTML = """
<html>
<body>
    <script type="application/ld+json">
    {
        "@type": "RealEstateListing",
        "name": "JSON-LD Project",
        "description": "From JSON-LD",
        "offers": {
            "@type": "Offer",
            "price": "1500000",
            "priceCurrency": "SAR"
        },
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "Jeddah",
            "addressCountry": "Saudi Arabia"
        }
    }
    </script>
</body>
</html>
"""


class TestDarGlobalScraper:
    @pytest.fixture
    def scraper(self):
        return DarGlobalScraper()
    
    @patch.object(DarGlobalScraper, 'fetch')
    def test_parse_detail_basic(self, mock_fetch, scraper):
        mock_fetch.return_value = BeautifulSoup(DARGLOBAL_PROJECT_HTML, "lxml")
        result = scraper.parse_detail(mock_fetch.return_value, "https://darglobal.co.uk/projects/test")
        
        assert result is not None
        assert result["title"] == "DarGlobal Luxury Residences"
        assert result["city"] == "Riyadh"
        assert result["country"] == "Saudi Arabia"
        assert "Swimming Pool" in result["amenities"]
        assert result["status"] == "Under Construction"
        assert result["price"] == 2500000
        assert result["currency"] == "SAR"
    
    @patch.object(DarGlobalScraper, 'fetch')
    def test_parse_detail_with_jsonld(self, mock_fetch, scraper):
        mock_fetch.return_value = BeautifulSoup(DARGLOBAL_JSONLD_HTML, "lxml")
        result = scraper.parse_detail(mock_fetch.return_value, "https://darglobal.co.uk/projects/jsonld")
        
        assert result is not None
        assert result["title"] == "JSON-LD Project"
        assert result["city"] == "Jeddah"
        assert result["country"] == "Saudi Arabia"
        assert result["price"] == 1500000
        assert result["currency"] == "SAR"


class TestWasaltScraper:
    @pytest.fixture
    def scraper(self):
        return WasaltScraper()
    
    @patch.object(WasaltScraper, 'fetch')
    def test_parse_detail_basic(self, mock_fetch, scraper):
        mock_fetch.return_value = BeautifulSoup(WASALT_LISTING_HTML, "lxml")
        result = scraper.parse_detail(mock_fetch.return_value, "https://wasalt.sa/en/property/test")
        
        assert result is not None
        assert result["title"] == "3BR Apartment in Al Rimal"
        assert result["city"] == "Riyadh"
        assert result["district"] == "Al Rimal"
        assert result["price"] == 750000
        assert result["currency"] == "SAR"
        assert result["bedrooms"] == 3
        assert result["bathrooms"] == 2
        assert result["area"] == 150
        assert result["transaction_type"] == "Sale"
    
    @patch.object(WasaltScraper, 'fetch')
    def test_parse_detail_missing_fields(self, mock_fetch, scraper):
        """Test that missing fields are None, not defaults"""
        minimal_html = """
        <html><body>
            <h1 class="listing-title">Minimal Listing</h1>
        </body></html>
        """
        mock_fetch.return_value = BeautifulSoup(minimal_html, "lxml")
        result = scraper.parse_detail(mock_fetch.return_value, "https://wasalt.sa/en/property/minimal")
        
        assert result is not None
        assert result["title"] == "Minimal Listing"
        assert result["price"] is None
        assert result["currency"] is None
        assert result["country"] is None
        assert result["status"] is None


class TestPropertyRecord:
    def test_valid_record(self):
        record = PropertyRecord(
            source="DarGlobal",
            title="Test Project",
            source_url="https://example.com/project",
            city="Riyadh",
            price=1000000,
            currency="SAR"
        )
        assert record.source == "DarGlobal"
        assert record.city == "Riyadh"
    
    def test_to_document_text(self):
        record = PropertyRecord(
            source="Wasalt",
            title="Test Apartment",
            source_url="https://example.com/apt",
            property_type="Apartment",
            city="Riyadh",
            price=750000,
            currency="SAR",
            bedrooms=3,
            amenities=["Parking", "Pool"]
        )
        text = record.to_document_text()
        assert "Wasalt" in text
        assert "Test Apartment" in text
        assert "750,000 SAR" in text
        assert "Parking, Pool" in text
    
    def test_to_metadata(self):
        record = PropertyRecord(
            source="DarGlobal",
            title="Test",
            source_url="https://example.com",
            city="Riyadh",
            price=1000000
        )
        meta = record.to_metadata()
        assert meta["source"] == "DarGlobal"
        assert meta["city"] == "Riyadh"
        assert meta["price"] == 1000000
    
    def test_data_quality_score(self):
        # Complete record
        complete = PropertyRecord(
            source="DarGlobal", title="Test", source_url="https://example.com",
            property_type="Apartment", city="Riyadh", price=1000000, currency="SAR",
            bedrooms=3, bathrooms=2, area=150, description="Nice", amenities=["Pool"],
            status="Available", image_url="https://img.jpg"
        )
        assert complete.data_quality_score() == 1.0
        
        # Minimal record
        minimal = PropertyRecord(
            source="DarGlobal", title="Test", source_url="https://example.com"
        )
        assert minimal.data_quality_score() < 0.3
    
    def test_invalid_source(self):
        with pytest.raises(ValueError):
            PropertyRecord(source="Invalid", title="Test", source_url="https://example.com")


class TestURLCanonicalization:
    def test_canonicalize_removes_tracking(self):
        from app.utils.url import canonicalize_url
        url = "https://example.com/property/123?utm_source=google&utm_medium=cpc&ref=facebook"
        result = canonicalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "ref" not in result
        assert "/property/123" in result
    
    def test_canonicalize_keeps_identifiers(self):
        from app.utils.url import canonicalize_url
        url = "https://example.com/property/123?id=456&listing_id=789&utm_source=google"
        result = canonicalize_url(url)
        assert "id=456" in result
        assert "listing_id=789" in result
        assert "utm_source" not in result
    
    def test_canonicalize_normalizes_slashes(self):
        from app.utils.url import canonicalize_url
        url = "https://example.com//property///123//"
        result = canonicalize_url(url)
        assert result == "https://example.com/property/123"
    
    def test_is_same_domain(self):
        from app.utils.url import is_same_domain
        assert is_same_domain("https://darglobal.co.uk/projects/1", "darglobal.co.uk")
        assert is_same_domain("https://www.darglobal.co.uk/projects/1", "darglobal.co.uk")
        assert not is_same_domain("https://evil-darglobal.co.uk/projects/1", "darglobal.co.uk")
        assert not is_same_domain("https://wasalt.sa/property/1", "darglobal.co.uk")


class TestJSONLDExtraction:
    def test_extract_real_estate_listing(self):
        from app.data.jsonld import extract_all_jsonld, extract_property_data
        soup = BeautifulSoup(DARGLOBAL_JSONLD_HTML, "lxml")
        jsonld_objects = extract_all_jsonld(soup)
        property_data = extract_property_data(jsonld_objects)
        
        assert property_data["title"] == "JSON-LD Project"
        assert property_data["description"] == "From JSON-LD"
        assert property_data["price"] == 1500000
        assert property_data["currency"] == "SAR"
        assert property_data["address"]["city"] == "Jeddah"
        assert property_data["address"]["country"] == "Saudi Arabia"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])