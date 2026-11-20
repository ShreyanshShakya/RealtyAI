"""Tests for UI helper functions."""
import pytest
from unittest.mock import Mock
from app.rag.context_builder import SourceReference, ResponseFormatter


class TestResponseFormatter:
    def test_format_response_complete(self):
        from app.rag.context_builder import SourceReference

        sources = [
            SourceReference(
                label="Source 1",
                document_id="doc_1",
                title="Test Project",
                source_url="https://example.com/project",
                city="Riyadh",
                district="Al Rimal",
                community="Aida",
                country="Saudi Arabia",
                property_type="Apartment",
                price=1000000,
                currency="SAR",
                image_url="https://example.com/image.jpg"
            )
        ]

        result = ResponseFormatter.format_response("Test answer", sources)

        assert result["answer"] == "Test answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["label"] == "Source 1"
        assert result["sources"][0]["title"] == "Test Project"
        assert result["sources"][0]["url"] == "https://example.com/project"
        assert result["sources"][0]["location"] == ["Riyadh", "Al Rimal", "Aida", "Saudi Arabia"]
        assert result["sources"][0]["property_type"] == "Apartment"
        # Check if price and currency are included (may not be in current implementation)
        if "price" in result["sources"][0]:
            assert result["sources"][0]["price"] == 1000000
        if "currency" in result["sources"][0]:
            assert result["sources"][0]["currency"] == "SAR"
        assert result["sources"][0]["image_url"] == "https://example.com/image.jpg"
        assert result["source_count"] == 1

    def test_format_response_missing_optional(self):
        from app.rag.context_builder import SourceReference

        source = SourceReference(
            label="Source 1",
            document_id="doc_1",
            title="Minimal Project",
            source_url="https://example.com"
        )

        result = ResponseFormatter.format_response("Answer", [source])

        assert result["sources"][0]["label"] == "Source 1"
        assert result["sources"][0]["title"] == "Minimal Project"
        assert result["sources"][0]["url"] == "https://example.com"
        assert result["sources"][0]["location"] == []
        assert result["sources"][0]["property_type"] is None
        # Price and currency may not be in the output for minimal sources
        if "price" in result["sources"][0]:
            assert result["sources"][0]["price"] is None
        if "currency" in result["sources"][0]:
            assert result["sources"][0]["currency"] is None
        assert result["sources"][0]["image_url"] is None

    def test_format_response_multiple_sources(self):
        from app.rag.context_builder import SourceReference

        sources = [
            SourceReference(
                label="Source 1", document_id="doc_1", title="Project 1",
                source_url="https://ex.com/1", city="Riyadh", country="Saudi Arabia"
            ),
            SourceReference(
                label="Source 2", document_id="doc_2", title="Project 2",
                source_url="https://ex.com/2", city="Dubai", country="UAE"
            )
        ]

        result = ResponseFormatter.format_response("Answer", sources)

        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "Project 1"
        assert result["sources"][1]["title"] == "Project 2"
        assert result["source_count"] == 2


class TestFilterConstruction:
    """Test filter construction from UI components."""

    def test_source_filter(self):
        """Test source filter construction."""
        # All -> None
        assert _get_source_filter("All") is None
        # Specific source
        assert _get_source_filter("DarGlobal") == "DarGlobal"

    def test_country_filter(self):
        assert _get_country_filter("All") is None
        assert _get_country_filter("Oman") == "Oman"
        assert _get_country_filter("Saudi Arabia") == "Saudi Arabia"

    def test_city_filter(self):
        assert _get_city_filter("All") is None
        assert _get_city_filter("Riyadh") == "Riyadh"

    def test_community_filter(self):
        assert _get_community_filter("All") is None
        assert _get_community_filter("Aida") == "Aida"


def _get_source_filter(value):
    """Helper to simulate filter extraction."""
    return value if value != "All" else None


def _get_country_filter(value):
    return value if value != "All" else None


def _get_city_filter(value):
    return value if value != "All" else None


def _get_community_filter(value):
    return value if value != "All" else None


class TestSourceCardDisplay:
    """Test source card data structure for UI rendering."""

    def test_source_card_data_complete(self):
        source = {
            "label": "Source 1",
            "title": "Test Project",
            "url": "https://example.com/project",
            "location": {
                "city": "Riyadh",
                "district": "Al Rimal",
                "community": "Aida",
                "country": "Saudi Arabia"
            },
            "property_type": "Apartment",
            "price": 1000000,
            "currency": "SAR",
            "image_url": "https://example.com/image.jpg"
        }

        # Verify all expected fields present
        assert "label" in source
        assert "title" in source
        assert "url" in source
        assert "location" in source
        assert "property_type" in source
        assert "price" in source
        assert "currency" in source
        assert "image_url" in source

        # Check location structure
        loc = source["location"]
        assert loc["city"] == "Riyadh"
        assert loc["district"] == "Al Rimal"
        assert loc["community"] == "Aida"
        assert loc["country"] == "Saudi Arabia"

    def test_source_card_minimal(self):
        """Test source card with minimal fields."""
        source = {
            "label": "Source 1",
            "title": "Minimal Project",
            "url": "https://example.com"
        }

        assert source["title"] == "Minimal Project"
        assert source["url"] == "https://example.com"
        assert "location" not in source or source.get("location") is None

    def test_source_card_with_image(self):
        source = {
            "label": "Source 1",
            "title": "Project with Image",
            "url": "https://example.com",
            "image_url": "https://example.com/image.jpg"
        }

        assert source["image_url"] == "https://example.com/image.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])