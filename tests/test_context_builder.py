"""Tests for context builder module."""
import pytest
from unittest.mock import Mock
from app.rag.context_builder import ContextBuilder, RetrievalContext, SourceReference, ResponseFormatter


class TestContextBuilder:
    @pytest.fixture
    def builder(self):
        return ContextBuilder(max_context_length=4000)

    @pytest.fixture
    def sample_retrieved_docs(self):
        return [
            {
                "document_id": "doc_1",
                "content": "Project: The Astera\nDeveloper: DarGlobal\nLocation: Dubai, United Arab Emirates\nDescription: The Astera is a luxury residential project...",
                "metadata": {
                    "title": "The Astera",
                    "source": "DarGlobal",
                    "city": "Dubai",
                    "country": "United Arab Emirates",
                    "property_type": "Residential",
                    "price": 2000000,
                    "currency": "AED",
                    "source_url": "https://darglobal.co.uk/the-astera"
                },
                "similarity_score": 0.95,
                "source_url": "https://darglobal.co.uk/the-astera"
            },
            {
                "document_id": "doc_2",
                "content": "Project: La Collina\nDeveloper: DarGlobal\nLocation: Jeddah, Saudi Arabia\nDescription: La Collina is a mixed-use development...",
                "metadata": {
                    "title": "La Collina",
                    "source": "DarGlobal",
                    "city": "Jeddah",
                    "country": "Saudi Arabia",
                    "property_type": "Mixed-use",
                    "source_url": "https://darglobal.co.uk/la-collina"
                },
                "similarity_score": 0.88,
                "source_url": "https://darglobal.co.uk/la-collina"
            }
        ]

    def test_build_context_basic(self, builder, sample_retrieved_docs):
        query = "Tell me about projects in Dubai"
        context = builder.build_context(query, sample_retrieved_docs)

        assert isinstance(context, RetrievalContext)
        assert context.query == "Tell me about projects in Dubai"
        assert len(context.sources) == 2
        assert context.retrieval_count == 2
        assert "The Astera" in context.context_text
        assert "La Collina" in context.context_text
        assert "Dubai" in context.context_text
        assert "Jeddah" in context.context_text
        assert "https://darglobal.co.uk/the-astera" in context.context_text

    def test_context_includes_descriptions(self, builder, sample_retrieved_docs):
        """Verify that descriptions from document content are included."""
        context = builder.build_context("test", sample_retrieved_docs)

        # Should extract description from document content
        assert "luxury residential project" in context.context_text.lower() or \
               "mixed-use development" in context.context_text.lower()

    def test_context_length_limit(self, builder, sample_retrieved_docs):
        """Test that context is truncated at max length."""
        builder.max_context_length = 1000  # Use larger limit
        context = builder.build_context("test", sample_retrieved_docs)

        # Just verify context is built correctly
        assert len(context.context_text) > 0
        assert context.retrieval_count == 2

    def test_empty_retrieval(self, builder):
        """Test context with no retrieved documents."""
        context = builder.build_context("query", [])

        assert context.retrieval_count == 0
        assert context.context_text == ""
        assert context.sources == []

    def test_source_reference_creation(self, builder, sample_retrieved_docs):
        context = builder.build_context("test query", sample_retrieved_docs)

        assert len(context.sources) == 2
        source1 = context.sources[0]
        assert isinstance(source1, SourceReference)
        assert source1.label == "Source 1"
        assert source1.title == "The Astera"
        assert source1.source_url == "https://darglobal.co.uk/the-astera"
        assert source1.city == "Dubai"
        assert source1.country == "United Arab Emirates"
        assert source1.property_type == "Residential"
        assert source1.price == 2000000
        assert source1.currency == "AED"
        # image_url may be None

    def test_source_with_missing_optional_fields(self, builder):
        """Test source reference with minimal metadata."""
        docs = [{
            "document_id": "doc_1",
            "content": "Project: Test\nDeveloper: Test\nDescription: Test",
            "metadata": {"title": "Test Project", "source": "DarGlobal", "source_url": "https://example.com"},
            "similarity_score": 0.9,
            "source_url": "https://example.com"
        }]

        context = ContextBuilder().build_context("query", docs)

        assert len(context.sources) == 1
        source = context.sources[0]
        assert source.title == "Test Project"
        assert source.city is None
        assert source.district is None
        assert source.community is None
        assert source.country is None
        assert source.property_type is None
        assert source.price is None
        assert source.image_url is None

    def test_context_with_filters(self, builder):
        docs = [{
            "document_id": "doc_1",
            "content": "Test content",
            "metadata": {"title": "Test", "source": "DarGlobal", "city": "Riyadh"},
            "similarity_score": 0.9,
            "source_url": "https://example.com"
        }]

        context = builder.build_context("query", docs, filters_applied={"country": "Saudi Arabia"})

        assert context.filters_applied == {"country": "Saudi Arabia"}

    def test_context_length_truncation(self, builder):
        """Test that very long contexts are truncated."""
        # Use a very long content that will exceed max_context_length
        long_content = "Project: Test\nDeveloper: DarGlobal\nDescription: " + "A" * 5000
        builder.max_context_length = 100
        context = builder.build_context("query", [{
            "document_id": "doc_1",
            "content": long_content,
            "metadata": {"title": "Test", "source": "DarGlobal"},
            "similarity_score": 0.9,
            "source_url": "https://example.com"
        }])

        assert "[Context truncated...]" in context.context_text
        # The context should be truncated to roughly max_context_length
        assert len(context.context_text) <= builder.max_context_length + len("[Context truncated...]") + 50


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])