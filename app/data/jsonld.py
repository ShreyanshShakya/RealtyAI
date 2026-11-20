import json
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from loguru import logger

SCHEMA_TYPES = {
    "RealEstateListing",
    "Product",
    "Offer",
    "Place",
    "PostalAddress",
    "Residence",
    "Apartment",
    "SingleFamilyResidence",
    "House",
    "Property",
    "ResidentialProperty",
    "CommercialProperty",
}

def extract_all_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract all JSON-LD script tags from page"""
    scripts = soup.find_all("script", type="application/ld+json")
    results = []
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string.strip())
            results.extend(_flatten_jsonld(data))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON-LD parse error: {e}")
    return results

def _flatten_jsonld(data: Any) -> List[Dict[str, Any]]:
    """Flatten JSON-LD data handling arrays, @graph, and nested structures"""
    if isinstance(data, list):
        items = []
        for item in data:
            items.extend(_flatten_jsonld(item))
        return items
    if not isinstance(data, dict):
        return []
    
    # Handle @graph
    if "@graph" in data:
        items = []
        for item in data["@graph"]:
            items.extend(_flatten_jsonld(item))
        return items
    
    return [data]

def extract_property_data(jsonld_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract property-relevant data from JSON-LD objects"""
    result = {
        "title": None,
        "description": None,
        "price": None,
        "currency": None,
        "property_type": None,
        "transaction_type": None,
        "city": None,
        "country": None,
        "district": None,
        "address": {},
        "image_url": None,
        "availability": None,
    }
    
    for obj in jsonld_objects:
        obj_type = obj.get("@type", "")
        if isinstance(obj_type, list):
            obj_type = obj_type[0] if obj_type else ""
        
        # Check if this object is relevant to property data
        type_str = str(obj_type)
        if not any(t in type_str for t in SCHEMA_TYPES):
            continue
        
        # Title/Name
        if not result["title"] and obj.get("name"):
            result["title"] = obj["name"]
        
        # Description
        if not result["description"] and obj.get("description"):
            result["description"] = obj["description"]
        
        # Image
        if not result["image_url"]:
            img = obj.get("image")
            if isinstance(img, str):
                result["image_url"] = img
            elif isinstance(img, list) and img:
                result["image_url"] = img[0] if isinstance(img[0], str) else img[0].get("url")
            elif isinstance(img, dict):
                result["image_url"] = img.get("url")
        
        # Offers -> price, currency, availability, transaction type
        offers = obj.get("offers")
        if offers:
            offer_list = offers if isinstance(offers, list) else [offers]
            for offer in offer_list:
                if not result["price"] and offer.get("price"):
                    try:
                        result["price"] = float(offer["price"])
                    except (ValueError, TypeError):
                        pass
                if not result["currency"] and offer.get("priceCurrency"):
                    result["currency"] = offer["priceCurrency"]
                if not result["transaction_type"] and offer.get("@type") == "Offer":
                    bf = offer.get("businessFunction", "")
                    if "Sell" in bf:
                        result["transaction_type"] = "Sale"
                    elif "Lease" in bf or "Rent" in bf:
                        result["transaction_type"] = "Rent"
                if not result["availability"] and offer.get("availability"):
                    avail = offer["availability"]
                    if isinstance(avail, str) and "InStock" in avail:
                        result["availability"] = "Available"
        
        # Address
        address = obj.get("address")
        if address and isinstance(address, dict):
            result["address"] = {
                "street": address.get("streetAddress"),
                "city": address.get("addressLocality"),
                "district": address.get("addressRegion"),
                "country": address.get("addressCountry"),
                "postal_code": address.get("postalCode"),
            }
            # Also set flat fields for easier merging
            if address.get("addressLocality") and not result["city"]:
                result["city"] = address["addressLocality"]
            if address.get("addressCountry") and not result["country"]:
                result["country"] = address["addressCountry"]
            if address.get("addressRegion") and not result["district"]:
                result["district"] = address["addressRegion"]
        
        # Property type from @type
        if not result["property_type"]:
            type_mapping = {
                "Apartment": "Apartment",
                "House": "House",
                "SingleFamilyResidence": "Villa",
                "Residence": "Residential",
                "Property": "Property",
                "ResidentialProperty": "Residential",
                "CommercialProperty": "Commercial",
            }
            for k, v in type_mapping.items():
                if k in str(obj_type):
                    result["property_type"] = v
                    break
    
    return result