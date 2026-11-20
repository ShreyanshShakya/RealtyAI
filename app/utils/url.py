from urllib.parse import urlparse, urljoin, parse_qsl, urlunparse
import re

# Parameters to strip (tracking/marketing)
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclid", "ref", "source", "origin",
    "_ga", "_gl", "mc_cid", "mc_eid"
}

# Parameters that ARE identifiers (keep these)
IDENTIFIER_PARAMS = {"id", "listing_id", "property_id", "project_id", "pid"}

def canonicalize_url(url: str, base_url: str = None) -> str:
    """Normalize URL for deduplication"""
    if base_url:
        url = urljoin(base_url, url)
    
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    
    # Normalize path: remove duplicate slashes
    path = re.sub(r'/+', '/', path)
    # Normalize trailing slash
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    
    # Filter query parameters
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = []
    for k, v in query_params:
        k_lower = k.lower()
        if k_lower in TRACKING_PARAMS:
            continue
        if k_lower in IDENTIFIER_PARAMS:
            filtered.append((k, v))
        elif not k_lower.startswith("utm_") and not k_lower.startswith("_"):
            filtered.append((k, v))
    
    query = "&".join(f"{k}={v}" for k, v in filtered)
    fragment = ""
    
    return urlunparse((scheme, netloc, path, "", query, fragment))

def is_same_domain(url: str, allowed_domain: str) -> bool:
    """Check if URL belongs to allowed domain (exact or subdomain)"""
    host = urlparse(url).hostname or ""
    allowed_domain = allowed_domain.lower()
    
    return (
        host == allowed_domain
        or host.endswith("." + allowed_domain)
    )