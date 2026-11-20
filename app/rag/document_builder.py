"""Document preparation for RAG ingestion.

Converts verified project records into documents suitable for semantic retrieval.
"""
import hashlib
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from app.data.schema import PropertyRecord


@dataclass
class RagDocument:
    """A document ready for embedding and indexing."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    source_record: PropertyRecord


class DocumentBuilder:
    """Builds RAG documents from validated property records."""
    
    def __init__(self):
        self.logger = logger.bind(component="DocumentBuilder")
    
    def build_documents(self, records: List[PropertyRecord]) -> List[RagDocument]:
        """Convert validated records to RAG documents."""
        documents = []
        
        for record in records:
            doc = self._build_document(record)
            documents.append(doc)
        
        self.logger.info(f"Built {len(documents)} RAG documents")
        return documents
    
    def _build_document(self, record: PropertyRecord) -> RagDocument:
        """Build a single RAG document from a property record."""
        # Generate deterministic document ID from source URL
        document_id = self._generate_document_id(record.source_url)
        
        # Build searchable content
        content = self._build_content(record)
        
        # Build metadata for ChromaDB
        metadata = self._build_metadata(record)
        
        return RagDocument(
            document_id=document_id,
            content=content,
            metadata=metadata,
            source_record=record
        )
    
    def _generate_document_id(self, source_url: str) -> str:
        """Generate deterministic document ID from canonical source URL."""
        # Use SHA256 of canonical URL for deterministic ID
        return hashlib.sha256(source_url.encode()).hexdigest()[:32]
    
    def _build_content(self, record: PropertyRecord) -> str:
        """Build searchable text content for embedding."""
        parts = []
        
        # Title
        parts.append(f"Project: {record.title}")
        
        # Developer/Source
        parts.append(f"Developer: {record.source}")
        
        # Description
        if record.description:
            parts.append(f"\nProject Description:\n{record.description}")
        
        # Location
        location_parts = []
        if record.city:
            location_parts.append(record.city)
        if record.district:
            location_parts.append(record.district)
        if record.community:
            location_parts.append(record.community)
        if record.country:
            location_parts.append(record.country)
        
        if location_parts:
            parts.append(f"\nLocation: {', '.join(location_parts)}")
        
        # Property type
        if record.property_type:
            parts.append(f"\nProperty Type: {record.property_type}")
        
        # Transaction type
        if record.transaction_type:
            parts.append(f"Transaction Type: {record.transaction_type}")
        
        # Price
        if record.price is not None and record.currency:
            parts.append(f"\nPrice: {record.price:,.0f} {record.currency}")
        
        # Bedrooms/Bathrooms
        if record.bedrooms is not None:
            parts.append(f"Bedrooms: {record.bedrooms}")
        if record.bathrooms is not None:
            parts.append(f"Bathrooms: {record.bathrooms}")
        
        # Area
        if record.area is not None:
            area_str = f"{record.area:,.0f}"
            if record.area_unit:
                area_str += f" {record.area_unit}"
            else:
                area_str += " sqm"
            parts.append(f"Area: {area_str}")
        
        # Amenities
        if record.amenities:
            parts.append(f"\nAmenities: {', '.join(record.amenities)}")
        
        # Status
        if record.status:
            parts.append(f"Status: {record.status}")
        
        # Source
        parts.append(f"\nSource: {record.source}")
        parts.append(f"Original URL: {record.source_url}")
        
        return "\n".join(parts)
    
    def _build_metadata(self, record: PropertyRecord) -> Dict[str, Any]:
        """Build ChromaDB-compatible metadata."""
        metadata = {
            "source": record.source,
            "source_url": record.source_url,
            "title": record.title,
            "record_type": getattr(record, 'record_type', 'project'),
        }
        
        # Add optional fields only if present
        if record.country:
            metadata["country"] = record.country
        if record.country_code:
            metadata["country_code"] = record.country_code
        if record.city:
            metadata["city"] = record.city
        if record.district:
            metadata["district"] = record.district
        if record.community:
            metadata["community"] = record.community
        if record.property_type:
            metadata["property_type"] = record.property_type
        if record.transaction_type:
            metadata["transaction_type"] = record.transaction_type
        if record.price is not None:
            metadata["price"] = record.price
        if record.currency:
            metadata["currency"] = record.currency
        if record.bedrooms is not None:
            metadata["bedrooms"] = record.bedrooms
        if record.bathrooms is not None:
            metadata["bathrooms"] = record.bathrooms
        if record.area is not None:
            metadata["area"] = record.area
        if record.area_unit:
            metadata["area_unit"] = record.area_unit
        if record.area_sqm is not None:
            metadata["area_sqm"] = record.area_sqm
        if record.image_url:
            metadata["image_url"] = record.image_url
        if record.status:
            metadata["status"] = record.status
        if record.scraped_at:
            metadata["scraped_at"] = record.scraped_at.isoformat()
        
        # Handle alternate URLs
        alternate_urls = getattr(record, 'alternate_source_urls', [])
        if alternate_urls:
            metadata["alternate_source_urls"] = json.dumps(alternate_urls)
        
        # Handle merged_from
        merged_from = getattr(record, 'merged_from', [])
        if merged_from:
            metadata["merged_from"] = json.dumps(merged_from)
        
        # Quality info
        if hasattr(record, 'quality_score'):
            metadata["quality_score"] = record.quality_score
        if hasattr(record, 'quality_issues') and record.quality_issues:
            metadata["quality_issues"] = json.dumps(record.quality_issues)
        
        return metadata


def load_rag_ready_records(filepath: str) -> List[PropertyRecord]:
    """Load RAG-ready records from JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(PropertyRecord(**data))
    return records