#!/usr/bin/env python
"""DarGlobal data cleaning and classification for RAG ingestion.

Processes validated records through:
1. Classification: project | insight | crypto | duplicate
2. Deduplication with alternate URL preservation
3. Country code normalization (ISO alpha-2 + full name)
4. City/district/community normalization
5. Quality segmentation: RAG-ready / low-quality / invalid / non-project
6. Output: 5 JSONL files + quality report
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

settings = get_settings()


@dataclass
class ClassifiedRecord:
    """Record with classification and normalized fields"""
    original: Dict[str, Any]
    classification: str  # project, insight, crypto, duplicate
    title: str
    source_url: str
    city: Optional[str] = None
    district: Optional[str] = None
    community: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    description: Optional[str] = None
    property_type: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    area: Optional[float] = None
    area_unit: Optional[str] = None
    area_sqm: Optional[float] = None
    amenities: List[str] = None
    status: Optional[str] = None
    image_url: Optional[str] = None
    alternate_source_urls: List[str] = None
    record_type: str = "project"
    quality_score: float = 0.0
    quality_issues: List[str] = None
    merged_from: List[str] = None  # URLs of merged duplicates


# Country normalization map
COUNTRY_MAP = {
    "OM": ("OM", "Oman"),
    "Oman": ("OM", "Oman"),
    "AE": ("AE", "United Arab Emirates"),
    "UAE": ("AE", "United Arab Emirates"),
    "United Arab Emirates": ("AE", "United Arab Emirates"),
    "QA": ("QA", "Qatar"),
    "Qatar": ("QA", "Qatar"),
    "SA": ("SA", "Saudi Arabia"),
    "KSA": ("SA", "Saudi Arabia"),
    "Saudi Arabia": ("SA", "Saudi Arabia"),
    "GB": ("GB", "United Kingdom"),
    "UK": ("GB", "United Kingdom"),
    "England": ("GB", "United Kingdom"),
    "London": ("GB", "United Kingdom"),
    "ES": ("ES", "Spain"),
    "Spain": ("ES", "Spain"),
}


def normalize_country(country: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (country_code, country_name) from various inputs"""
    if not country:
        return None, None
    country = country.strip()
    if country in COUNTRY_MAP:
        return COUNTRY_MAP[country]
    # Try case-insensitive
    for k, v in COUNTRY_MAP.items():
        if k.lower() == country.lower():
            return v
    # Unknown - return as-is but log
    logger.warning(f"Unknown country: {country}")
    return country.upper()[:2], country


def normalize_location(city: Optional[str], district: Optional[str], 
                       country: Optional[str], description: Optional[str],
                       source_url: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Normalize city, district, community, country from various sources"""
    community = None
    desc_lower = description.lower() if description else ""
    url_lower = source_url.lower()
    
    # Extract community from URL or description
    known_communities = ["aida", "rayana", "amaya", "trump plaza", "one-of-one", 
                         "the asteria", "the mulliner", "la collina"]
    for comm in known_communities:
        if comm in url_lower or comm in desc_lower:
            community = comm.title()
            if community == "One-Of-One":
                community = "One of One"
            elif community == "Trump Plaza":
                community = "Trump Plaza"
            break
    
    # Fix specific known locations
    if city:
        city_lower = city.lower().strip()
        # Fix "149 Old Park Lane" -> London/Mayfair
        if "149 old park lane" in city_lower or "149 old park lane" in (district or "").lower():
            city = "London"
            district = "Mayfair"
            community = "The Mulliner"
        # Fix Muscat district duplication
        elif city_lower == "muscat" and district and district.lower() == "muscat":
            district = None
        # Fix "Doha" when it's just a reference for Simaisma
        elif city_lower == "doha" and "simaisma" in desc_lower:
            city = "Simaisma"
            district = None
        # Fix "Benahavís" 
        elif "benahav" in city_lower:
            city = "Benahavís"
            district = None
        # Fix "Al Marjan Island"
        elif "al marjan" in city_lower or "al marjan" in (district or "").lower():
            city = "Al Marjan Island"
            district = None
        # Fix redundant district when it equals city
        elif district and district.lower() == city.lower().strip():
            district = None
        # Fix Benahavís district duplication
        elif "benahav" in city.lower() and district and "benahav" in district.lower():
            district = None
        # Fix Doha district duplication
        elif city.lower() == "doha" and district and district.lower() == "doha":
            district = None
        # Fix Dubai district duplication
        elif city.lower() == "dubai" and district and district.lower() == "dubai":
            district = None
        # Fix Jeddah district duplication
        elif city.lower() == "jeddah" and district and district.lower() == "jeddah":
            district = None
        # Fix Al Marjan Island
        elif "al marjan" in city.lower() or "al marjan" in (district or "").lower():
            city = "Al Marjan Island"
            district = None
    
    # Country inference from URL/description when missing
    if not country:
        # Specific Qatar projects - check BEFORE Rayana
        if "trump-international-qatar" in source_url.lower() or "simaisma" in (description or "").lower():
            country = "Qatar"
        # Rayana projects -> Saudi Arabia
        elif "rayana" in source_url.lower() or "rayana" in (description or "").lower():
            country = "Saudi Arabia"
        # Trump Plaza -> UAE (Dubai)
        elif "trump plaza" in source_url.lower() or "trump plaza" in (description or "").lower():
            country = "United Arab Emirates"
        # La Collina -> Saudi Arabia (Jeddah)
        elif "la-collina" in source_url.lower() or "la collina" in (description or "").lower():
            country = "Saudi Arabia"
        # The Astera -> UAE (Al Marjan Island)
        elif "the-astera" in source_url.lower() or "the asteria" in (description or "").lower():
            country = "United Arab Emirates"
        # Trump Tower Jeddah -> Saudi Arabia
        elif "trump-tower-jeddah" in source_url.lower() or "trump tower jeddah" in (description or "").lower():
            country = "Saudi Arabia"
        # Trump Plaza projects -> UAE (Dubai)
        elif "trump-plaza" in source_url.lower() or "trump executive" in (description or "").lower() or "trump park" in (description or "").lower():
            country = "United Arab Emirates"
        # Padel Living -> Saudi Arabia (Jeddah)
        elif "padel-living" in source_url.lower() or "padel living" in (description or "").lower():
            country = "Saudi Arabia"
    
    # Normalize country
    country_code, country_name = normalize_country(country)
    
    return city, district, community, country_code, country_name


def classify_record(record: Dict[str, Any]) -> str:
    """Classify record as project, insight, or crypto"""
    url = record.get("source_url", "").lower()
    title = record.get("title", "").lower()
    desc = record.get("description", "").lower() if record.get("description") else ""
    
    # Insight articles
    if "/insights/" in url:
        return "insight"
    
    # Crypto/investment pages
    if "/invest-crypto" in url or "/buy-property-with-crypto" in url or "crypto" in title:
        return "crypto"
    
    # Articles with minimal content (the one-of-one sub-pages)
    if "/one-of-one/" in url and len(title) < 50 and not desc:
        return "insight"  # These are article-like
    
    return "project"


def extract_location_from_description(desc: str, url: str) -> Dict[str, Optional[str]]:
    """Extract location hints from description text"""
    result = {"city": None, "district": None, "community": None}
    if not desc:
        return result
    
    desc_lower = desc.lower()
    
    # Community extraction
    if "aida" in desc_lower and "rayana" not in desc_lower:
        result["community"] = "Aida"
    elif "rayana" in desc_lower:
        result["community"] = "Rayana"
    elif "amaya" in desc_lower:
        result["community"] = "Amaya"
    elif "trump plaza" in desc_lower:
        result["community"] = "Trump Plaza"
    elif "one of one" in desc_lower:
        result["community"] = "One of One"
    elif "the asteria" in desc_lower:
        result["community"] = "The Astera"
    elif "la collina" in desc_lower:
        result["community"] = "La Collina"
    elif "the mulliner" in desc_lower:
        result["community"] = "The Mulliner"
    
    # City/district hints
    locations = {
        "muscat": ("Muscat", None),
        "jeddah": ("Jeddah", None),
        "dubai": ("Dubai", None),
        "riyadh": ("Riyadh", None),
        "doha": ("Doha", None),
        "benahav": ("Benahavís", None),
        "london": ("London", "Mayfair"),
        "al marjan": ("Al Marjan Island", None),
        "qetaifan": ("Doha", "Qetaifan Island"),
        "wadi safar": ("Riyadh", "Wadi Safar"),
        "diriyah": ("Riyadh", "Diriyah"),
    }
    
    for key, (city, district) in locations.items():
        if key in desc_lower and not result["city"]:
            result["city"] = city
            result["district"] = district
    
    return result


def process_record(record: Dict[str, Any]) -> ClassifiedRecord:
    """Process a single raw record into classified/normalized form"""
    classification = classify_record(record)
    
    # Extract basic fields
    title = record.get("title", "").strip()
    source_url = record.get("source_url", "").strip()
    description = record.get("description")
    if description:
        description = description.strip()
    
    # Extract location
    city = record.get("city")
    district = record.get("district")
    country = record.get("country")
    
    # Try to extract from description if missing
    if not city or not district or not country:
        loc_hints = extract_location_from_description(description or "", source_url)
        if not city and loc_hints["city"]:
            city = loc_hints["city"]
        if not district and loc_hints["district"]:
            district = loc_hints["district"]
        if not country and loc_hints.get("country"):
            country = loc_hints["country"]
    
    # Normalize location
    norm_city, norm_district, community, country_code, country_name = normalize_location(
        city, district, country, description, source_url
    )
    
    # Override community if found in description
    if not community:
        loc_hints = extract_location_from_description(description or "", source_url)
        community = loc_hints.get("community")
    
    # Calculate quality score
    quality_issues = []
    score = 0
    fields_checked = [
        ("title", title),
        ("source_url", source_url),
        ("city", norm_city),
        ("country", country_name),
        ("description", description),
        ("property_type", record.get("property_type")),
        ("price", record.get("price")),
        ("bedrooms", record.get("bedrooms")),
        ("area", record.get("area")),
        ("amenities", record.get("amenities")),
        ("status", record.get("status")),
        ("image_url", record.get("image_url")),
    ]
    
    for field_name, value in fields_checked:
        if value:
            score += 1
        else:
            quality_issues.append(f"missing_{field_name}")
    
    quality_score = score / len(fields_checked)
    
    # Check for meaningful description
    has_meaningful_desc = bool(description and len(description) >= 80)
    if not has_meaningful_desc:
        quality_issues.append("description_too_short_or_missing")
    
    return ClassifiedRecord(
        original=record,
        classification=classification,
        title=title,
        source_url=source_url,
        city=norm_city,
        district=norm_district,
        community=community,
        country=country_name,
        country_code=country_code,
        description=description,
        property_type=record.get("property_type"),
        price=record.get("price"),
        currency=record.get("currency"),
        bedrooms=record.get("bedrooms"),
        bathrooms=record.get("bathrooms"),
        area=record.get("area"),
        area_unit=record.get("area_unit"),
        area_sqm=record.get("area_sqm"),
        amenities=record.get("amenities", []),
        status=record.get("status"),
        image_url=record.get("image_url"),
        alternate_source_urls=[],
        record_type="project",
        quality_score=quality_score,
        quality_issues=quality_issues,
        merged_from=[]
    )


def deduplicate_records(records: List[ClassifiedRecord]) -> List[ClassifiedRecord]:
    """Merge duplicate records based on title similarity and URL patterns"""
    # Group by normalized title
    title_groups = defaultdict(list)
    for rec in records:
        norm_title = rec.title.lower().strip()
        # Remove common suffixes for comparison
        norm_title = norm_title.replace("trump international hotel & tower", "").replace("trump international hotel and tower", "").strip()
        norm_title = norm_title.replace("trump international hotel", "").strip()
        title_groups[norm_title].append(rec)
    
    merged = []
    seen_urls = set()
    
    for norm_title, group in title_groups.items():
        if len(group) == 1:
            rec = group[0]
            if rec.source_url not in seen_urls:
                merged.append(rec)
                seen_urls.add(rec.source_url)
        else:
            # Merge duplicates - keep the one with most complete data
            group.sort(key=lambda r: r.quality_score, reverse=True)
            primary = group[0]
            alternates = [r.source_url for r in group[1:]]
            primary.alternate_source_urls = alternates
            primary.merged_from = alternates
            
            # Merge non-conflicting fields from duplicates
            for dup in group[1:]:
                for field in ["city", "district", "country", "country_code", "community",
                             "description", "property_type", "price", "currency",
                             "bedrooms", "bathrooms", "area", "amenities", "status", "image_url"]:
                    if not getattr(primary, field) and getattr(dup, field):
                        setattr(primary, field, getattr(dup, field))
            
            # Merge quality issues
            all_issues = set(primary.quality_issues)
            for dup in group[1:]:
                all_issues.update(dup.quality_issues)
            primary.quality_issues = list(all_issues)
            
            merged.append(primary)
            seen_urls.add(primary.source_url)
            for alt in alternates:
                seen_urls.add(alt)
    
    return merged


def segment_records(records: List[ClassifiedRecord]) -> Tuple[
    List[ClassifiedRecord],  # rag_ready
    List[ClassifiedRecord],  # low_quality
    List[ClassifiedRecord],  # invalid
    List[ClassifiedRecord]   # non_project
]:
    """Segment records into quality categories"""
    rag_ready = []
    low_quality = []
    invalid = []
    non_project = []
    
    for rec in records:
        # Non-project content
        if rec.classification in ("insight", "crypto"):
            non_project.append(rec)
            continue
        
        # Invalid: missing required fields
        if not rec.title or not rec.source_url:
            rec.quality_issues.append("missing_required_field")
            invalid.append(rec)
            continue
        
        # RAG-ready: meaningful description + project classification
        has_meaningful_desc = bool(rec.description and len(rec.description) >= 80)
        has_location = bool(rec.city or rec.country)
        
        if rec.classification == "project" and has_meaningful_desc:
            rag_ready.append(rec)
        elif rec.classification == "project":
            low_quality.append(rec)
        else:
            invalid.append(rec)
    
    return rag_ready, low_quality, invalid, non_project


def record_to_rag_ready(rec: ClassifiedRecord) -> Dict[str, Any]:
    """Convert to RAG-ready record format"""
    return {
        "source": rec.original.get("source", "DarGlobal"),
        "source_url": rec.source_url,
        "alternate_source_urls": rec.alternate_source_urls,
        "title": rec.title,
        "record_type": rec.record_type,
        "property_type": rec.property_type,
        "description": rec.description,
        "city": rec.city,
        "district": rec.district,
        "community": rec.community,
        "country": rec.country,
        "country_code": rec.country_code,
        "price": rec.price,
        "currency": rec.currency,
        "bedrooms": rec.bedrooms,
        "bathrooms": rec.bathrooms,
        "area": rec.area,
        "area_unit": rec.area_unit,
        "area_sqm": rec.area_sqm,
        "amenities": rec.amenities,
        "status": rec.status,
        "image_url": rec.image_url,
        "quality_score": round(rec.quality_score, 3),
        "quality_issues": rec.quality_issues,
        "merged_from": rec.merged_from,
    }


def record_to_output(rec: ClassifiedRecord, include_quality: bool = True) -> Dict[str, Any]:
    """Convert to output format"""
    out = {
        "source": rec.original.get("source", "DarGlobal"),
        "source_url": rec.source_url,
        "title": rec.title,
        "property_type": rec.property_type,
        "transaction_type": rec.original.get("transaction_type"),
        "city": rec.city,
        "district": rec.district,
        "community": rec.community,
        "country": rec.country,
        "country_code": rec.country_code,
        "price": rec.price,
        "currency": rec.currency,
        "bedrooms": rec.bedrooms,
        "bathrooms": rec.bathrooms,
        "area": rec.area,
        "area_unit": rec.area_unit,
        "area_sqm": rec.area_sqm,
        "description": rec.description,
        "amenities": rec.amenities,
        "status": rec.status,
        "image_url": rec.image_url,
    }
    if include_quality:
        out["quality_score"] = round(rec.quality_score, 3)
        out["quality_issues"] = rec.quality_issues
    return out


def generate_quality_report(
    rag_ready: List[ClassifiedRecord],
    low_quality: List[ClassifiedRecord],
    invalid: List[ClassifiedRecord],
    non_project: List[ClassifiedRecord],
    original_count: int
) -> Dict[str, Any]:
    """Generate comprehensive quality report"""
    
    def field_completeness(records: List[ClassifiedRecord]) -> Dict[str, float]:
        if not records:
            return {}
        fields = ["title", "source_url", "city", "district", "community", "country", 
                  "country_code", "description", "property_type", "price", "currency",
                  "bedrooms", "bathrooms", "area", "amenities", "status", "image_url"]
        completeness = {}
        for field in fields:
            count = sum(1 for r in records if getattr(r, field))
            completeness[field] = round(count / len(records) * 100, 1)
        return completeness
    
    duplicates = [r for r in rag_ready if r.alternate_source_urls]
    
    return {
        "summary": {
            "total_original_records": original_count,
            "rag_ready": len(rag_ready),
            "low_quality": len(low_quality),
            "invalid": len(invalid),
            "non_project_excluded": len(non_project),
            "duplicates_merged": len(duplicates),
        },
        "field_completeness": {
            "rag_ready": field_completeness(rag_ready),
            "low_quality": field_completeness(low_quality),
        },
        "duplicates": [
            {
                "primary_url": r.source_url,
                "alternate_urls": r.alternate_source_urls,
                "title": r.title,
            }
            for r in duplicates
        ],
        "non_project_categories": {
            "insights": len([r for r in non_project if r.classification == "insight"]),
            "crypto": len([r for r in non_project if r.classification == "crypto"]),
        },
        "quality_distribution": {
            "high": len([r for r in rag_ready if r.quality_score >= 0.7]),
            "medium": len([r for r in rag_ready if 0.4 <= r.quality_score < 0.7]),
            "low": len([r for r in rag_ready if r.quality_score < 0.4]),
        },
        "warnings": [
            "Price missing from all records",
            "Property type missing from most records",
            "Description length varies significantly",
            "Some locations inferred from description text",
        ]
    }


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    # Paths
    raw_file = Path("dataset/raw/darglobal_raw.jsonl")
    validated_file = Path("dataset/processed/darglobal_validated.jsonl")
    
    output_dir = Path("dataset")
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    rejected_dir = output_dir / "rejected"
    excluded_dir = output_dir / "excluded"
    reports_dir = Path("reports")
    
    for d in [raw_dir, processed_dir, rejected_dir, excluded_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load validated records (or raw if validated doesn't exist)
    input_file = validated_file if validated_file.exists() else raw_file
    logger.info(f"Loading records from {input_file}")
    
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    logger.info(f"Loaded {len(records)} records")
    
    # Process each record
    classified = [process_record(r) for r in records]
    
    # Deduplicate
    deduplicated = deduplicate_records(classified)
    logger.info(f"After deduplication: {len(deduplicated)} records")
    
    # Segment
    rag_ready, low_quality, invalid, non_project = segment_records(deduplicated)
    
    logger.info(f"Segmentation: RAG-ready={len(rag_ready)}, low-quality={len(low_quality)}, invalid={len(invalid)}, non-project={len(non_project)}")
    
    # Write output files
    # 1. RAG-ready
    rag_file = processed_dir / "darglobal_rag_ready.jsonl"
    with open(rag_file, "w", encoding="utf-8") as f:
        for rec in rag_ready:
            f.write(json.dumps(record_to_rag_ready(rec), ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(rag_ready)} RAG-ready records to {rag_file}")
    
    # 2. Low quality
    lq_file = rejected_dir / "darglobal_low_quality.jsonl"
    with open(lq_file, "w", encoding="utf-8") as f:
        for rec in low_quality:
            f.write(json.dumps(record_to_output(rec), ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(low_quality)} low-quality records to {lq_file}")
    
    # 3. Invalid
    inv_file = rejected_dir / "darglobal_invalid.jsonl"
    with open(inv_file, "w", encoding="utf-8") as f:
        for rec in invalid:
            f.write(json.dumps(record_to_output(rec), ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(invalid)} invalid records to {inv_file}")
    
    # 4. Non-project (excluded)
    np_file = excluded_dir / "darglobal_non_project.jsonl"
    with open(np_file, "w", encoding="utf-8") as f:
        for rec in non_project:
            f.write(json.dumps(record_to_output(rec), ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(non_project)} non-project records to {np_file}")
    
    # 5. Quality report
    report = generate_quality_report(rag_ready, low_quality, invalid, non_project, len(records))
    report_file = reports_dir / "data_quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved quality report to {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DARGLOBAL DATA CLEANING COMPLETE")
    print("="*60)
    print(f"Original records:      {len(records)}")
    print(f"RAG-ready:             {len(rag_ready)}")
    print(f"Low quality:           {len(low_quality)}")
    print(f"Invalid:               {len(invalid)}")
    print(f"Non-project (excluded):{len(non_project)}")
    print(f"Duplicates merged:     {len([r for r in rag_ready if r.alternate_source_urls])}")
    print(f"\nOutput files:")
    print(f"  {rag_file}")
    print(f"  {lq_file}")
    print(f"  {inv_file}")
    print(f"  {np_file}")
    print(f"  {report_file}")
    print("="*60)


if __name__ == "__main__":
    main()