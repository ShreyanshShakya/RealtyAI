"""Unified property data schema."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class PropertyRecord(BaseModel):
    """Unified property schema for both sources"""
    source: str = Field(..., description="DarGlobal or Wasalt")
    title: str = Field(..., min_length=1)
    property_type: Optional[str] = None
    transaction_type: Optional[str] = None  # Sale, Rent, N/A
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    community: Optional[str] = None
    price: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = None
    bedrooms: Optional[int] = Field(None, ge=0)
    bathrooms: Optional[int] = Field(None, ge=0)
    area: Optional[float] = Field(None, ge=0)
    area_unit: Optional[str] = None
    area_sqm: Optional[float] = None
    description: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    status: Optional[str] = None
    source_url: str
    image_url: Optional[str] = None
    
    # Additional metadata for RAG
    record_type: Optional[str] = None
    alternate_source_urls: List[str] = Field(default_factory=list)
    merged_from: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = None
    quality_issues: List[str] = Field(default_factory=list)
    merged_from: List[str] = Field(default_factory=list)
    
    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return v.upper()
        return v
    
    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ["DarGlobal", "Wasalt"]:
            raise ValueError("Source must be DarGlobal or Wasalt")
        return v
    
    def to_document_text(self) -> str:
        """Convert to searchable document text for embedding"""
        lines = [
            f"Source: {self.source}",
            f"Property: {self.title}",
        ]
        
        if self.property_type:
            lines.append(f"Type: {self.property_type}")
        if self.transaction_type:
            lines.append(f"Transaction: {self.transaction_type}")
        if self.city:
            loc = self.city
            if self.district:
                loc = f"{self.district}, {self.city}"
            if self.country:
                loc = f"{loc}, {self.country}"
            lines.append(f"Location: {loc}")
        if self.price is not None and self.currency:
            lines.append(f"Price: {self.price:,.0f} {self.currency}")
        if self.bedrooms is not None:
            lines.append(f"Bedrooms: {self.bedrooms}")
        if self.bathrooms is not None:
            lines.append(f"Bathrooms: {self.bathrooms}")
        if self.area is not None:
            lines.append(f"Area: {self.area:,.0f} sqm")
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.amenities:
            lines.append(f"Amenities: {', '.join(self.amenities)}")
        if self.status:
            lines.append(f"Status: {self.status}")
        lines.append(f"Source URL: {self.source_url}")
        
        return "\n".join(lines)
    
    def to_metadata(self) -> dict:
        """Extract metadata for ChromaDB filtering"""
        return {
            "source": self.source,
            "property_type": self.property_type or "",
            "transaction_type": self.transaction_type or "",
            "country": self.country or "",
            "country_code": self.country_code or "",
            "city": self.city or "",
            "district": self.district or "",
            "community": self.community or "",
            "price": self.price if self.price is not None else -1,
            "currency": self.currency or "",
            "bedrooms": self.bedrooms if self.bedrooms is not None else -1,
            "bathrooms": self.bathrooms if self.bathrooms is not None else -1,
            "area": self.area if self.area is not None else -1,
            "status": self.status or "",
            "source_url": self.source_url,
            "image_url": self.image_url or "",
        }
    
    def data_quality_score(self) -> float:
        """Calculate data quality score (0-1) based on field completeness"""
        fields = [
            self.title,
            self.property_type,
            self.city,
            self.price,
            self.currency,
            self.bedrooms,
            self.bathrooms,
            self.area,
            self.description,
            self.amenities,
            self.status,
            self.image_url,
        ]
        filled = sum(1 for f in fields if f not in (None, "", [], -1))
        return filled / len(fields)