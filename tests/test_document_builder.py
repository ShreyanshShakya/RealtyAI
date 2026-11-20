"""Tests for document builder module."""
import pytest
import json
from unittest.mock import Mock, patch
from app.rag.document_builder import DocumentBuilder, RagDocument, load_rag_ready_records
from app.data.schema import PropertyRecord


class TestDocumentBuilder:
    @pytest.fixture
    def builder(self):
        return DocumentBuilder()
    
    @pytest.fixture
    def sample_record(self):
        return PropertyRecord(
            source="DarGlobal",
            title="Test Project",
            source_url="https://darglobal.co.uk/projects/test",
            city="Riyadh",
            country="Saudi Arabia",
            country_code="SA",
            description="A luxury residential project in Riyadh.",
            property_type="Residential",
            price=1000000,
            currency="SAR",
            bedrooms=3,
            bathrooms=2,
            area=150,
            area_unit="sqm",
            amenities=["Pool", "Gym", "Security"],
            status="Under Construction",
            image_url="https://example.com/image.jpg",
            district="Al Rimal",
            community="Aida"
        )
    
    def test_build_document(self, builder, sample_record):
        doc = builder._build_document(sample_record)
        
        assert isinstance(doc, RagDocument)
        assert doc.document_id is not None
        assert len(doc.document_id) == 32  # SHA256 truncated to 32 chars
        assert "Project: Test Project" in doc.content
        assert "Developer: DarGlobal" in doc.content
        assert "A luxury residential project in Riyadh." in doc.content
        assert "Location: Riyadh, Al Rimal, Aida, Saudi Arabia" in doc.content
        assert "Property Type: Residential" in doc.content
        assert "Price: 1,000,000 SAR" in doc.content
        assert "Bedrooms: 3" in doc.content
        assert "Bathrooms: 2" in doc.content
        assert "Area: 150 sqm" in doc.content
        assert "Amenities: Pool, Gym, Security" in doc.content
        assert "Status: Under Construction" in doc.content
        assert "Source: DarGlobal" in doc.content
        assert "Original URL: https://darglobal.co.uk/projects/test" in doc.content
    
    def test_build_documents_multiple(self, builder, sample_record):
        records = [sample_record, sample_record]
        docs = builder.build_documents(records)
        
        assert len(docs) == 2
        # IDs should be same for same URL
        assert docs[0].document_id == docs[1].document_id
    
    def test_document_id_deterministic(self, builder, sample_record):
        doc1 = builder._build_document(sample_record)
        doc2 = builder._build_document(sample_record)
        
        assert doc1.document_id == doc2.document_id
    
    def test_missing_optional_fields(self, builder):
        """Test handling of missing optional fields."""
        record = PropertyRecord(
            source="DarGlobal",
            title="Minimal Project",
            source_url="https://example.com/project"
        )
        doc = builder._build_document(record)
        
        assert doc.content is not None
        assert "Project: Minimal Project" in doc.content
        # Optional fields should be None or empty
        assert doc.metadata.get("city") is None
        assert doc.metadata.get("country") is None
        assert doc.metadata.get("property_type") is None


class TestLoadRagReadyRecords:
    def test_load_valid_jsonl(self, tmp_path):
        """Test loading records from JSONL file."""
        test_file = tmp_path / "test_records.jsonl"
        test_data = [
            {
                "source": "DarGlobal",
                "title": "Project 1",
                "source_url": "https://example.com/1",
                "city": "Riyadh",
                "country": "Saudi Arabia"
            },
            {
                "source": "DarGlobal",
                "title": "Project 2",
                "source_url": "https://example.com/2",
                "city": "Dubai",
                "country": "UAE"
            }
        ]
        
        with open(test_file, "w") as f:
            for record in test_data:
                f.write(json.dumps(record) + "\n")
        
        records = load_rag_ready_records(str(test_file))
        assert len(records) == 2
        assert records[0].title == "Project 1"
        assert records[1].title == "Project 2"
    
    def test_load_empty_file(self, tmp_path):
        test_file = tmp_path / "empty.jsonl"
        test_file.write_text("")
        
        records = load_rag_ready_records(str(test_file))
        assert len(records) == 0
    
    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_rag_ready_records("/nonexistent/path.jsonl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])