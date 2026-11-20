#!/usr/bin/env python
"""Validate RAG-ready dataset for consistency and completeness."""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationFinding:
    """Single validation finding"""
    record_index: int
    title: str
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    field: Optional[str] = None


def validate_record(record: Dict[str, Any], index: int) -> List[ValidationFinding]:
    """Validate a single record and return findings"""
    findings = []
    title = record.get("title", "")
    
    # Check for missing project title
    if not title:
        findings.append(ValidationFinding(
            record_index=index,
            title="MISSING_TITLE",
            severity="error",
            category="missing_fields",
            message="Project title is missing or empty",
            field="title"
        ))
    
    # Check for missing source URL
    if not record.get("source_url"):
        findings.append(ValidationFinding(
            record_index=index,
            title=title or "UNKNOWN",
            severity="error",
            category="missing_fields",
            message="Source URL is missing",
            field="source_url"
        ))
    
    # Check for missing description
    if not record.get("description"):
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="warning",
            category="missing_fields",
            message="Description is missing",
            field="description"
        ))
    elif len(record.get("description", "")) < 80:
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="info",
            category="description_quality",
            message=f"Description is very short ({len(record.get('description', ''))} chars)",
            field="description"
        ))
    
    # Check country consistency
    country = record.get("country")
    country_code = record.get("country_code")
    city = record.get("city")
    
    # Valid country codes
    valid_country_codes = {"OM", "SA", "AE", "QA", "GB", "ES"}
    if country_code and country_code not in valid_country_codes:
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="error",
            category="invalid_country",
            message=f"Invalid country code: {country_code}",
            field="country_code"
        ))
    
    # Check country name consistency with country code
    country_map = {
        "OM": "Oman",
        "SA": "Saudi Arabia",
        "AE": "United Arab Emirates",
        "QA": "Qatar",
        "GB": "United Kingdom",
        "ES": "Spain",
    }
    if country_code and country and country_code in country_map:
        expected_country = country_map[country_code]
        if country != expected_country:
            findings.append(ValidationFinding(
                record_index=index,
                title=title,
                severity="error",
                category="country_inconsistency",
                message=f"Country name '{country}' doesn't match country code '{country_code}' (expected '{expected_country}')",
                field="country"
            ))
    
    # Check city/country consistency
    city_country_map = {
        "Muscat": "Oman",
        "Jeddah": "Saudi Arabia",
        "Dubai": "United Arab Emirates",
        "Riyadh": "Saudi Arabia",
        "Doha": "Qatar",
        "London": "United Kingdom",
        "Benahavís": "Spain",
        "Al Marjan Island": "United Arab Emirates",
        "Simaisma": "Qatar",
    }
    if city and city in city_country_map:
        expected_country = city_country_map[city]
        if country and country != expected_country:
            findings.append(ValidationFinding(
                record_index=index,
                title=title,
                severity="error",
                category="city_country_inconsistency",
                message=f"City '{city}' suggests country '{expected_country}' but record has '{country}'",
                field="country"
            ))
    
    # Check for missing metadata types
    if record.get("price") is not None and not isinstance(record.get("price"), (int, float)):
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="warning",
            category="invalid_metadata_type",
            message=f"Price should be numeric, got {type(record.get('price')).__name__}",
            field="price"
        ))
    
    if record.get("bedrooms") is not None and not isinstance(record.get("bedrooms"), int):
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="warning",
            category="invalid_metadata_type",
            message=f"Bedrooms should be integer, got {type(record.get('bedrooms')).__name__}",
            field="bedrooms"
        ))
    
    if record.get("area") is not None and not isinstance(record.get("area"), (int, float)):
        findings.append(ValidationFinding(
            record_index=index,
            title=title,
            severity="warning",
            category="invalid_metadata_type",
            message=f"Area should be numeric, got {type(record.get('area')).__name__}",
            field="area"
        ))
    
    return findings


def check_duplicates(records: List[Dict[str, Any]]) -> List[ValidationFinding]:
    """Check for duplicate records"""
    findings = []
    
    # Check duplicate canonical URLs
    url_map = defaultdict(list)
    for i, record in enumerate(records):
        url = record.get("source_url", "")
        if url:
            url_map[url].append(i)
    
    for url, indices in url_map.items():
        if len(indices) > 1:
            for idx in indices:
                findings.append(ValidationFinding(
                    record_index=idx,
                    title=records[idx].get("title", ""),
                    severity="error",
                    category="duplicate_url",
                    message=f"Duplicate source URL: {url} (also at indices {indices})",
                    field="source_url"
                ))
    
    # Check duplicate project IDs (if available)
    # Note: No explicit project_id field in current schema
    
    return findings


def main():
    rag_file = Path("dataset/processed/darglobal_rag_ready.jsonl")
    
    if not rag_file.exists():
        print(f"Error: File not found: {rag_file}")
        sys.exit(1)
    
    print(f"Loading records from {rag_file}...")
    records = []
    with open(rag_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records")
    
    # Validate each record
    all_findings = []
    for i, record in enumerate(records):
        findings = validate_record(record, i)
        all_findings.extend(findings)
    
    # Check duplicates
    dup_findings = check_duplicates(records)
    all_findings.extend(dup_findings)
    
    # Categorize findings
    errors = [f for f in all_findings if f.severity == "error"]
    warnings = [f for f in all_findings if f.severity == "warning"]
    infos = [f for f in all_findings if f.severity == "info"]
    
    # Count records with warnings/errors
    records_with_errors = set(f.record_index for f in errors)
    records_with_warnings = set(f.record_index for f in warnings)
    
    # Generate report
    report = {
        "total_records": len(records),
        "valid_records": len(records) - len(records_with_errors),
        "records_with_warnings": len(records_with_warnings),
        "records_with_errors": len(records_with_errors),
        "individual_findings": [
            {
                "record_index": f.record_index,
                "title": f.title,
                "severity": f.severity,
                "category": f.category,
                "message": f.message,
                "field": f.field,
            }
            for f in all_findings
        ],
        "summary_by_category": {}
    }
    
    # Summary by category
    for finding in all_findings:
        cat = finding.category
        if cat not in report["summary_by_category"]:
            report["summary_by_category"][cat] = {"error": 0, "warning": 0, "info": 0}
        report["summary_by_category"][cat][finding.severity] += 1
    
    # Save report
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "phase2_4_data_validation.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    print(f"Total records:           {len(records)}")
    print(f"Valid records:           {len(records) - len(records_with_errors)}")
    print(f"Records with warnings:   {len(records_with_warnings)}")
    print(f"Records with errors:     {len(records_with_errors)}")
    print(f"Total findings:          {len(all_findings)}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Info:     {len(infos)}")
    print("\nFindings by category:")
    for cat, counts in sorted(report["summary_by_category"].items()):
        print(f"  {cat}: {counts}")
    
    print(f"\nReport saved to: {report_file}")
    
    # Print individual errors
    if errors:
        print("\nERRORS:")
        for f in errors:
            print(f"  [{f.record_index}] {f.title}: {f.message}")
    
    if warnings:
        print("\nWARNINGS:")
        for f in warnings[:10]:  # Show first 10
            print(f"  [{f.record_index}] {f.title}: {f.message}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)