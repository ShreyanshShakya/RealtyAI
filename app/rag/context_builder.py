"""Context builder for RAG pipeline.

Converts retrieved documents into a controlled context for the LLM.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class SourceReference:
    """Reference to a source document."""
    label: str  # e.g., "Source 1"
    document_id: str
    title: str
    source_url: str
    city: Optional[str] = None
    district: Optional[str] = None
    community: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    property_type: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class RetrievalContext:
    """Context built from retrieved documents for LLM."""
    query: str
    sources: List[SourceReference]
    context_text: str
    retrieval_count: int
    filters_applied: Dict[str, Any]


class ContextBuilder:
    """Builds LLM context from retrieved documents."""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.logger = logger.bind(component="ContextBuilder")
    
    def build_context(self, 
                      query: str,
                      retrieved_docs: List[Dict[str, Any]],
                      filters_applied: Optional[Dict[str, Any]] = None) -> RetrievalContext:
        """Build context from retrieved documents."""
        sources = []
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {})
            label = f"Source {i + 1}"
            
            # Create source reference
            source_ref = SourceReference(
                label=label,
                document_id=doc.get("document_id", ""),
                title=metadata.get("title", "Unknown"),
                source_url=doc.get("source_url", metadata.get("source_url", "")),
                city=metadata.get("city"),
                district=metadata.get("district"),
                community=metadata.get("community"),
                country=metadata.get("country"),
                country_code=metadata.get("country_code"),
                property_type=metadata.get("property_type"),
                price=metadata.get("price"),
                currency=metadata.get("currency"),
                image_url=metadata.get("image_url")
            )
            sources.append(source_ref)
            
            # Build context text for this source
            # Use the actual document content (which contains the full description)
            doc_content = doc.get("content", "")
            source_text = self._format_source_text(label, metadata, doc_content)
            context_parts.append(source_text)
        
        # Combine context parts
        context_text = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context_text) > self.max_context_length:
            context_text = context_text[:self.max_context_length] + "\n[Context truncated...]"
        
        self.logger.info(f"Built context with {len(sources)} sources, length: {len(context_text)}")
        
        return RetrievalContext(
            query=query,
            sources=sources,
            context_text=context_text,
            retrieval_count=len(sources),
            filters_applied=filters_applied or {}
        )
    
    def _format_source_text(self, label: str, metadata: Dict[str, Any], doc_content: str = "") -> str:
        """Format a single source for context."""
        parts = [f"[{label}]"]
        
        # Title
        title = metadata.get("title", "Unknown")
        parts.append(f"Project: {title}")
        
        # Source
        source = metadata.get("source", "Unknown")
        parts.append(f"Developer: {source}")
        
        # Location
        location_parts = []
        if metadata.get("city"):
            location_parts.append(metadata["city"])
        if metadata.get("district"):
            location_parts.append(metadata["district"])
        if metadata.get("community"):
            location_parts.append(metadata["community"])
        if metadata.get("country"):
            location_parts.append(metadata["country"])
        
        if location_parts:
            parts.append(f"Location: {', '.join(location_parts)}")
        
        # Property type
        if metadata.get("property_type"):
            parts.append(f"Property Type: {metadata['property_type']}")
        
        # Transaction type
        if metadata.get("transaction_type"):
            parts.append(f"Transaction: {metadata['transaction_type']}")
        
        # Price
        if metadata.get("price") is not None and metadata.get("currency"):
            parts.append(f"Price: {metadata['price']:,.0f} {metadata['currency']}")
        
        # Description - use actual document content if available, fallback to metadata
        if doc_content and "Description:" in doc_content:
            # Extract description from document content
            desc_start = doc_content.find("Description:")
            if desc_start >= 0:
                desc = doc_content[desc_start + len("Description:"):].strip()
                # Find next section or end
                next_section = len(desc)
                for marker in ["\nSource:", "\nDeveloper:", "\nLocation:", "\nProperty Type:", "\nTransaction:", "\nPrice:", "\nBedrooms:", "\nBathrooms:", "\nArea:", "\nAmenities:", "\nStatus:"]:
                    idx = desc.find(marker)
                    if idx >= 0 and idx < next_section:
                        next_section = idx
                desc = desc[:next_section].strip()
                if len(desc) > 1000:
                    desc = desc[:1000] + "..."
                parts.append(f"Description: {desc}")
        elif metadata.get("description"):
            desc = metadata["description"]
            # Truncate very long descriptions
            if len(desc) > 1000:
                desc = desc[:1000] + "..."
            parts.append(f"Description: {desc}")
        
        # Source URL
        source_url = metadata.get("source_url", "")
        if source_url:
            parts.append(f"Source URL: {source_url}")
        
        return "\n".join(parts)


class ResponseFormatter:
    """Format the final response with citations."""
    
    @staticmethod
    def format_response(answer: str, sources: List[SourceReference]) -> Dict[str, Any]:
        """Format the final response with source citations."""
        # Build source list for display
        source_cards = []
        for source in sources:
            card = {
                "label": source.label,
                "title": source.title,
                "url": source.source_url,
                "location": [],
                "property_type": source.property_type,
                "image_url": source.image_url
            }
            
            # Build location string
            if source.city:
                card["location"].append(source.city)
            if source.district:
                card["location"].append(source.district)
            if source.community:
                card["location"].append(source.community)
            if source.country:
                card["location"].append(source.country)
            
            source_cards.append(card)
        
        return {
            "answer": answer,
            "sources": source_cards,
            "source_count": len(sources)
        }
    
    @staticmethod
    def add_inline_citations(answer: str, sources: List[SourceReference]) -> str:
        """Add inline citation markers to answer text.
        
        This is a simple implementation - in practice, the LLM should be
        instructed to add citations in its response.
        """
        # For now, return answer as-is. The LLM should be prompted to include citations.
        return answer