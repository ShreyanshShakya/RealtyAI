"""Hybrid retrieval for RealtyAI."""
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
from loguru import logger
import re

from app.config import get_settings
from app.rag.vector_store import VectorStore
from app.rag.embeddings import get_embedding_generator


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document_id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_url: str


class QueryParser:
    """Parse user queries to extract structured filters."""
    
    # Known locations from dataset
    KNOW_COUNTRIES = {
        "oman", "saudi arabia", "saudi", "uae", "united arab emirates", 
        "qatar", "united kingdom", "uk", "spain"
    }
    
    KNOWN_CITIES = {
        "muscat", "jeddah", "dubai", "riyadh", "doha", "london", 
        "benahavís", "al marjan island", "simaisma"
    }
    
    KNOWN_COMMUNITIES = {
        "aida", "rayana", "amaya", "the mulliner", 
        "trump plaza", "one of one"
    }
    
    # Property type keywords that should only trigger when explicitly requested
    PROPERTY_TYPE_KEYWORDS = {
        "apartment": "Apartment",
        "villa": "Villa",
        "villas": "Villa",
        "residence": "Residence",
        "residences": "Residence",
        "hotel": "Hotel",
        "tower": "Tower",
        "plot": "Land",
        "plots": "Land",
        "land": "Land",
        "commercial": "Commercial",
        "residential": "Residential",
        "mixed-use": "Mixed-use",
        "mixed use": "Mixed-use",
    }
    
    # Project name keywords that should NOT trigger property type detection
    PROJECT_NAME_EXCLUSIONS = {
        "hotel": {"trump international hotel", "hotel & tower", "hotel and tower", "international hotel"},
        "tower": {"trump tower", "tower jeddah", "international hotel & tower", "international hotel and tower"},
        "residence": {"trump mansions", "darghlobal mansions", "trump executive residences", "trump park residences", "marriott residences", "w residences", "trump furnished residences", "trump cliff villas", "trump golf villas"},
        "commercial": set(),
        "residential": set(),
        "mixed-use": set(),
    }
    
    # City to country mapping for location inference
    CITY_TO_COUNTRY = {
        "muscat": "Oman",
        "jeddah": "Saudi Arabia",
        "dubai": "United Arab Emirates",
        "riyadh": "Saudi Arabia",
        "doha": "Qatar",
        "london": "United Kingdom",
        "benahavís": "Spain",
        "al marjan island": "United Arab Emirates",
        "simaisma": "Qatar",
    }
    
    def __init__(self):
        self.logger = logger.bind(component="QueryParser")
    
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse query to extract structured filters.
        
        Returns canonical filter keys:
        - country: List[str] - country names (e.g., ["Oman", "Saudi Arabia"])
        - city: List[str] - city names (e.g., ["Riyadh", "Jeddah"])
        - community: List[str] - community names (e.g., ["Aida", "Rayana"])
        - source: str - "DarGlobal" or "Wasalt"
        - property_type: str or List[str] - "Apartment", "Villa", "Hotel", "Tower", "Land", etc.
        """
        query_lower = query.lower()
        filters = {}
        
        # Extract countries (support multiple) - canonical key: "country"
        countries = self._extract_countries(query_lower)
        if countries:
            filters["country"] = countries
        
        # Extract cities (support multiple) - canonical key: "city"
        cities = self._extract_cities(query.lower())
        if cities:
            filters["city"] = cities
            # Infer country from city if no explicit country was found
            if "country" not in filters:
                inferred_countries = []
                for city in cities:
                    city_key = city.lower()
                    if city_key in self.CITY_TO_COUNTRY:
                        inferred = self.CITY_TO_COUNTRY[city_key]
                        if inferred not in inferred_countries:
                            inferred_countries.append(inferred)
                if inferred_countries:
                    filters["country"] = inferred_countries
        
        # Extract communities (support multiple) - canonical key: "community"
        communities = self._extract_communities(query.lower())
        if communities:
            filters["community"] = communities
        
        # Extract source - canonical key: "source"
        if "darglobal" in query.lower():
            filters["source"] = "DarGlobal"
        elif "wasalt" in query.lower():
            filters["source"] = "Wasalt"
        
        # Extract property type - canonical key: "property_type"
        property_types = self._extract_property_types(query_lower)
        if property_types:
            if len(property_types) == 1:
                filters["property_type"] = property_types[0]
            else:
                filters["property_type"] = property_types
        
        self.logger.debug(f"Parsed filters: {filters}")
        return filters
    
    def _extract_countries(self, query_lower: str) -> List[str]:
        """Extract all countries from query."""
        country_mapping = {
            "oman": "Oman",
            "saudi arabia": "Saudi Arabia",
            "saudi": "Saudi Arabia",
            "uae": "United Arab Emirates",
            "united arab emirates": "United Arab Emirates",
            "qatar": "Qatar",
            "united kingdom": "United Kingdom",
            "uk": "United Kingdom",
            "spain": "Spain",
        }
        
        found = []
        for key, value in country_mapping.items():
            if self._word_in_query(key, query_lower):
                if value not in found:
                    found.append(value)
        return found
    
    def _extract_cities(self, query_lower: str) -> List[str]:
        """Extract all cities from query."""
        city_mapping = {
            "muscat": "Muscat",
            "jeddah": "Jeddah",
            "dubai": "Dubai",
            "riyadh": "Riyadh",
            "doha": "Doha",
            "london": "London",
            "benahavís": "Benahavís",
            "benahavis": "Benahavís",
            "al marjan island": "Al Marjan Island",
            "simaisma": "Simaisma",
        }
        
        found = []
        for key, value in city_mapping.items():
            if self._word_in_query(key, query_lower):
                if value not in found:
                    found.append(value)
        return found
    
    def _extract_communities(self, query_lower: str) -> List[str]:
        """Extract all communities from query."""
        community_mapping = {
            "aida": "Aida",
            "rayana": "Rayana",
            "amaya": "Amaya",
            "the mulliner": "The Mulliner",
            "trump plaza": "Trump Plaza",
            "one of one": "One of One",
        }
        
        found = []
        for key, value in community_mapping.items():
            if self._word_in_query(key, query_lower):
                if value not in found:
                    found.append(value)
        return found
    
    def _word_in_query(self, word: str, query_lower: str) -> bool:
        """Check if a word/phrase appears as a whole word/phrase in the query."""
        if " " in word:
            return word in query_lower
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, query_lower))
    
    def _extract_property_types(self, query_lower: str) -> List[str]:
        """Extract all property types from query."""
        explicit_type_phrases = [
            "property type", "type of property", "what type of", "kind of property",
            "looking for", "searching for", "find", "show me", "list",
            "apartments", "villas", "residences", "hotels", "towers", "plots", "lands",
            "commercial", "residential", "mixed-use", "mixed use"
        ]
        
        has_explicit_request = any(phrase in query_lower for phrase in explicit_type_phrases)
        
        found_types = []
        for key, value in self.PROPERTY_TYPE_KEYWORDS.items():
            if self._word_in_query(key, query_lower):
                is_false_positive = False
                if key in self.PROJECT_NAME_EXCLUSIONS:
                    for exclusion in self.PROJECT_NAME_EXCLUSIONS[key]:
                        if exclusion in query_lower:
                            is_false_positive = True
                            break
                
                if not is_false_positive and (has_explicit_request or self._is_likely_type_request(query_lower)):
                    if value not in found_types:
                        found_types.append(value)
        
        return found_types
    
    def _is_likely_type_request(self, query_lower: str) -> bool:
        """Heuristic to determine if query is asking for property type."""
        type_indicators = [
            "apartment", "villa", "villas", "residence", "residences", 
            "hotel", "tower", "plot", "plots", "land",
            "bedroom", "bedrooms", "bed", "bath",
            "commercial", "residential", "mixed-use", "mixed use"
        ]
        if any(indicator in query_lower for indicator in ["bedroom", "bathroom", "area", "sqm", "sq ft", "sqft"]):
            return True
        return False


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    min_results: int = 1
    similarity_threshold: float = 0.3


class HybridRetriever:
    """Hybrid retriever combining semantic search with metadata filtering."""
    
    def __init__(self, vector_store: VectorStore, config: Optional[RetrievalConfig] = None):
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.query_parser = QueryParser()
        self.embedding_generator = get_embedding_generator()
        self.logger = logger.bind(component="HybridRetriever")
    
    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid approach.
        
        Args:
            query: User query string
            filters: Optional explicit filters with canonical keys:
                - source: str
                - country: List[str] or str
                - city: List[str] or str
                - community: List[str] or str
                - property_type: str or List[str]
        
        Returns:
            List of retrieval result dictionaries
        """
        # Parse query for structured filters (canonical keys)
        query_filters = self.query_parser.parse(query)
        
        # Merge with explicit filters
        if filters:
            for key, value in filters.items():
                if value is not None:
                    canonical_key = self._normalize_filter_key(key)
                    if canonical_key in query_filters:
                        existing = query_filters[canonical_key]
                        if isinstance(existing, list) and isinstance(value, list):
                            for v in value:
                                if v not in existing:
                                    existing.append(v)
                        elif isinstance(existing, list):
                            if isinstance(value, list):
                                for v in value:
                                    if v not in existing:
                                        existing.append(v)
                            else:
                                if value not in existing:
                                    existing.append(value)
                        elif isinstance(value, list):
                            query_filters[canonical_key] = [existing] + [v for v in value if v != existing]
                        else:
                            if value != existing:
                                query_filters[canonical_key] = [existing, value]
                    else:
                        query_filters[canonical_key] = value if isinstance(value, list) else [value]
        
        # Pass canonical filters directly to vector_store
        self.logger.info(f"Retrieving with query: '{query}', canonical_filters: {query_filters}")
        
        results = self.vector_store.search(
            query=query,
            k=self.config.top_k,
            filters=query_filters if query_filters else None
        )
        
        # Filter by similarity threshold if set
        if self.config.similarity_threshold > 0:
            results = [r for r in results if r["similarity_score"] >= self.config.similarity_threshold]
        
        if len(results) < self.config.min_results:
            self.logger.warning(f"Only {len(results)} results found, minimum is {self.config.min_results}")
        
        self.logger.info(f"Retrieved {len(results)} results")
        return results
    
    def _normalize_filter_key(self, key: str) -> str:
        """Normalize filter key to canonical form."""
        key_mapping = {
            "countries": "country",
            "cities": "city",
            "communities": "community",
            "source": "source",
            "property_type": "property_type",
        }
        return key_mapping.get(key, key)

    @staticmethod
    def normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize filters to canonical form.
        
        Canonical representation:
        {
            "source": "DarGlobal",
            "country": ["Oman"],
            "city": ["Muscat"],
            "community": ["Aida"],
            "property_type": ["Residential"]
        }
        
        Rules:
        1. Accept singular and plural input aliases at the application boundary.
        2. Convert all inputs into canonical singular keys.
        3. Normalise scalar strings into lists where appropriate.
        4. Remove empty lists, None values and unselected filters.
        5. Deduplicate values.
        6. Preserve case-insensitive matching where appropriate.
        7. Do not silently discard valid filters.
        8. Ensure ChromaDB receives metadata keys that actually exist in indexed documents.
        
        Args:
            filters: Input filters with any key format (singular/plural)
            
        Returns:
            Normalized filters with canonical singular keys
        """
        if not filters:
            return {}
        
        # Canonical key mappings (plural -> singular)
        canonical_map = {
            "countries": "country",
            "cities": "city", 
            "communities": "community",
            "source": "source",
            "property_type": "property_type",
            "property_types": "property_type",
        }
        
        normalized = {}
        
        for key, value in filters.items():
            if value is None or value == "":
                continue
            
            # Get canonical key
            canonical_key = canonical_map.get(key, key)
            
            # Normalize value to list
            if isinstance(value, list):
                # Filter out empty/None values and deduplicate (case-insensitive for strings)
                values = []
                seen = set()
                for v in value:
                    if v is None or v == "":
                        continue
                    v_lower = v.lower() if isinstance(v, str) else str(v).lower()
                    if v_lower not in seen:
                        seen.add(v_lower)
                        values.append(v)
                
                if values:
                    # If only one value and it's a single-select field (source), keep as scalar
                    if canonical_key == "source" and len(values) == 1:
                        normalized[canonical_key] = values[0]
                    else:
                        normalized[canonical_key] = values
            else:
                # Scalar value - convert to list unless it's source
                if canonical_key == "source":
                    normalized[canonical_key] = value
                else:
                    normalized[canonical_key] = [value]
        
        return normalized
    
    def retrieve_with_structured_filters(self, 
                                           query: str,
                                           countries: Optional[List[str]] = None,
                                           cities: Optional[List[str]] = None,
                                           communities: Optional[List[str]] = None,
                                           property_type: Optional[Union[str, List[str]]] = None,
                                           property_types: Optional[List[str]] = None,
                                           source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve with explicit structured filters (supports multiple values).
        
        Uses metadata-only retrieval when filters are provided for exact matching.
        """
        filters = {}
        if countries:
            filters["country"] = countries
        if cities:
            filters["city"] = cities
        if communities:
            filters["community"] = communities
        if property_types:
            filters["property_type"] = property_types
        elif property_type:
            # Handle both string and list
            if isinstance(property_type, list):
                filters["property_type"] = property_type
            else:
                filters["property_type"] = property_type
        if source:
            filters["source"] = source
        
        # Use metadata-only retrieval for exact filter matching
        if filters:
            return self.retrieve_for_listing(
                question=query,
                filters=filters,
                top_k=50
            )
        else:
            # Fall back to semantic search
            return self.retrieve(query, filters)

    def retrieve_for_listing(self, 
                              question: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None,
                              countries: Optional[List[str]] = None,
                              cities: Optional[List[str]] = None,
                              communities: Optional[List[str]] = None,
                              property_type: Optional[Union[str, List[str]]] = None,
                              property_types: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for listing queries (metadata-only, no similarity threshold).
        
        Uses ChromaDB get() with where filter for exact metadata matching.
        Retrieves ALL matching records by default (no artificial limit).
        Supports pagination for large datasets.
        
        Args:
            question: Optional query string (not used for metadata-only retrieval)
            filters: Canonical filters dict (preferred)
            countries: List of countries (legacy, merged into filters)
            cities: List of cities (legacy, merged into filters)
            communities: List of communities (legacy, merged into filters)
            property_type: Single property type (legacy)
            property_types: List of property types (legacy)
            title: Specific title to filter by
            top_k: Maximum results (None = retrieve all matching records)
        """
        # Build combined filters (canonical keys)
        combined_filters = {}
        if filters:
            for k, v in filters.items():
                if v is not None:
                    combined_filters[k] = v
        if countries:
            combined_filters["country"] = countries
        if cities:
            combined_filters["city"] = cities
        if communities:
            combined_filters["community"] = communities
        if property_types:
            combined_filters["property_type"] = property_types
        elif property_type:
            combined_filters["property_type"] = property_type
        if title:
            combined_filters["title"] = title
        
        query = question if question else ""
        
        # For listing, use metadata-only retrieval with NO similarity threshold
        # and NO artificial limit (retrieve all matching records)
        if combined_filters:
            return self.vector_store.get_by_metadata(
                filters=combined_filters,
                limit=top_k
            )
        else:
            # Fall back to semantic with no threshold and high/no limit
            original_threshold = self.config.similarity_threshold
            original_top_k = self.config.top_k
            self.config.similarity_threshold = 0.0
            self.config.top_k = top_k if top_k is not None else 1000
            try:
                return self.retrieve(query, combined_filters)
            finally:
                self.config.similarity_threshold = original_threshold
                self.config.top_k = original_top_k


class NumericalFilter:
    """Handle numerical filters for price, bedrooms, area etc."""
    
    @staticmethod
    def apply_filters(results: List[Dict[str, Any]], 
                      min_price: Optional[float] = None,
                      max_price: Optional[float] = None,
                      min_bedrooms: Optional[int] = None,
                      max_bedrooms: Optional[int] = None,
                      min_area: Optional[float] = None,
                      max_area: Optional[float] = None) -> List[Dict[str, Any]]:
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            if min_price is not None or max_price is not None:
                price = metadata.get("price")
                if price is None:
                    include = False
                else:
                    if min_price is not None and price < min_price:
                        include = False
                    if max_price is not None and price > max_price:
                        include = False
            
            if min_bedrooms is not None or max_bedrooms is not None:
                bedrooms = metadata.get("bedrooms")
                if bedrooms is None:
                    include = False
                else:
                    if min_bedrooms is not None and bedrooms < min_bedrooms:
                        include = False
                    if max_bedrooms is not None and bedrooms > max_bedrooms:
                        include = False
            
            if min_area is not None or max_area is not None:
                area = metadata.get("area")
                if area is None:
                    include = False
                else:
                    if min_area is not None and area < min_area:
                        include = False
                    if max_area is not None and area > max_area:
                        include = False
            
            if include:
                filtered.append(result)
        
        return filtered