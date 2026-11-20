"""RAG Pipeline orchestration for RealtyAI."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from app.config import get_settings
from app.rag.document_builder import DocumentBuilder, load_rag_ready_records
from app.rag.vector_store import VectorStore
from app.rag.retriever import HybridRetriever, RetrievalConfig, QueryParser
from app.rag.context_builder import ContextBuilder, RetrievalContext
from app.llm.openrouter import OpenRouterClient, SYSTEM_PROMPT, LLMResponse


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_documents: List[Dict[str, Any]]
    filters_applied: Dict[str, Any]
    model: str
    retrieval_count: int
    context_length: int
    query_type: str = "general"  # general, listing, comparison, detail


class RAGPipeline:
    """Main RAG pipeline orchestration."""
    
    def __init__(self, 
                 retrieval_config: Optional[RetrievalConfig] = None,
                 openrouter_client: Optional = None):
        settings = get_settings()
        
        # Initialize components
        self.document_builder = DocumentBuilder()
        self.vector_store = VectorStore()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retriever = HybridRetriever(self.vector_store, self.retrieval_config)
        self.context_builder = ContextBuilder()
        self.openrouter_client = openrouter_client or OpenRouterClient()
        self.query_parser = QueryParser()
        
        self.logger = logger.bind(component="RAGPipeline")
    
    def initialize(self, rebuild_index: bool = False) -> int:
        """Initialize the pipeline by building/loading the index."""
        self.logger.info("Initializing RAG pipeline...")
        
        # Load RAG-ready records
        settings = get_settings()
        rag_ready_path = f"{settings.PROCESSED_DATA_DIR}/darglobal_rag_ready.jsonl"
        records = load_rag_ready_records(rag_ready_path)
        self.logger.info(f"Loaded {len(records)} RAG-ready records")
        
        # Build documents
        documents = self.document_builder.build_documents(records)
        
        # Build or load index
        if rebuild_index or self.vector_store.needs_rebuild("dataset/processed/darglobal_rag_ready.jsonl"):
            self.logger.info("Building new index...")
            indexed_count = self.vector_store.build_index(documents, rebuild=True)
        else:
            self.logger.info("Using existing index")
            indexed_count = self.vector_store.get_collection_stats()["document_count"]
        
        self.logger.success(f"RAG pipeline initialized with {indexed_count} indexed documents")
        return indexed_count
    
    def ask(self, question: str, filters: Optional[Dict[str, Any]] = None) -> RAGResponse:
        """Process a user question through the full RAG pipeline."""
        self.logger.info(f"Processing question: {question}")
        
        # Normalize filters to canonical form early
        from app.rag.retriever import HybridRetriever
        canonical_filters = HybridRetriever.normalize_filters(filters)
        self.logger.debug(f"Normalized filters: {canonical_filters}")
        
        # Detect query type for specialized handling
        query_type = self._detect_query_type(question)
        self.logger.debug(f"Detected query type: {query_type}")
        
        # Check for explicit Wasalt source request
        is_wasalt_query = (
            (canonical_filters and canonical_filters.get("source") == "Wasalt") or
            "wasalt" in question.lower()
        )
        
        # Step 1: Retrieve relevant documents (with query-type specific handling)
        if query_type == "listing":
            retrieved_docs = self._retrieve_for_listing(question, canonical_filters)
        elif query_type == "comparison":
            retrieved_docs = self._retrieve_for_comparison(question, canonical_filters)
        elif query_type == "detail":
            retrieved_docs = self._retrieve_for_detail(question, canonical_filters)
        else:
            retrieved_docs = self.retriever.retrieve(question, canonical_filters)
        
        # Step 2: Pre-generation validation - handle zero results
        if not retrieved_docs:
            return self._handle_zero_results(question, query_type, is_wasalt_query, canonical_filters)
        
        # Step 3: Build context
        context = self.context_builder.build_context(
            query=question,
            retrieved_docs=retrieved_docs,
            filters_applied=canonical_filters
        )
        
        # Step 4: Generate answer using LLM
        answer = self._generate_answer(question, context, query_type)
        
        # Step 5: Format sources
        sources = self._format_sources(context.sources)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_documents=[self._format_retrieved_doc(d) for d in retrieved_docs],
            filters_applied=context.filters_applied,
            model=self.openrouter_client.model,
            retrieval_count=context.retrieval_count,
            context_length=len(context.context_text),
            query_type=query_type
        )

    def _handle_zero_results(self, question: str, query_type: str, 
                              is_wasalt_query: bool, filters: Dict[str, Any]) -> RAGResponse:
        """Handle case when no relevant documents are retrieved.
        
        Does not call OpenRouter. Returns controlled response explaining
        the requested information is not available in the collected dataset.
        """
        self.logger.info(f"Zero results for query: {question}")
        
        if is_wasalt_query:
            # Explicit Wasalt request - explain Wasalt data unavailable
            answer = (
                "I couldn't find any Wasalt projects matching your criteria. "
                "Wasalt records have not yet been acquired in our dataset. "
                "The current dataset contains only DarGlobal projects. "
                "Please try searching for DarGlobal projects instead."
            )
        elif query_type == "listing":
            answer = (
                "I couldn't find any projects matching your criteria in the DarGlobal dataset. "
                "The dataset currently contains 29 verified DarGlobal projects across Oman, Saudi Arabia, "
                "United Arab Emirates, Qatar, United Kingdom, and Spain. "
                "Try broadening your location filters or checking the spelling of locations."
            )
        elif query_type == "comparison":
            answer = (
                "I couldn't find enough information to compare the requested projects. "
                "The dataset contains 29 verified DarGlobal projects. "
                "Make sure the project names are spelled correctly and exist in the dataset."
            )
        elif query_type == "detail":
            answer = (
                "I couldn't find any relevant information for the requested project in the DarGlobal dataset. "
                "The dataset contains 29 verified DarGlobal projects. "
                "Try rephrasing your question or asking about a specific project, location, or community."
            )
        else:
            answer = (
                "I couldn't find any relevant information in the DarGlobal dataset for your question. "
                "The dataset currently contains 29 verified DarGlobal projects. "
                "Try rephrasing your question or asking about a specific project, location, or community."
            )
        
        return RAGResponse(
            answer=answer,
            sources=[],
            retrieved_documents=[],
            filters_applied=filters,
            model=self.openrouter_client.model,
            retrieval_count=0,
            context_length=0,
            query_type=query_type
        )
    
    def _detect_query_type(self, question: str) -> str:
        """Detect the type of query for specialized handling."""
        question_lower = question.lower()
        
        # Comparison queries
        if any(phrase in question_lower for phrase in [
            "compare", "vs", "versus", "difference between", "which is better",
            "compare and contrast"
        ]):
            return "comparison"
        
        # Listing queries - expanded to include natural language listing requests
        if any(phrase in question_lower for phrase in [
            "list all", "show all", "what projects", "which projects",
            "projects in", "show projects", "all projects", "list projects",
            "give me all", "display all",
            "find projects", "find ", "search for", "looking for projects",
            "what residential", "what luxury", "what commercial", "what properties",
            "available projects", "available properties"
        ]):
            return "listing"
        
        # Detail queries (specific project)
        if any(phrase in question_lower for phrase in [
            "tell me about", "describe", "details of", "information about",
            "what is", "what's"
        ]):
            return "detail"
        
        return "general"
    
    def _retrieve_for_listing(self, question: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for listing queries with no similarity threshold."""
        # Extract location filters from query (canonical keys)
        query_filters = self.query_parser.parse(question)
        
        # Also extract multiple property types
        property_types = self.query_parser._extract_property_types(question.lower())
        
        # Merge query-derived filters with explicit sidebar filters
        # Explicit sidebar filters must remain constraints rather than being silently broadened by query-derived filters
        combined_filters = {}
        for key in ["country", "city", "community", "source", "property_type"]:
            values = []
            # Add query-derived filters
            if key in query_filters:
                val = query_filters[key]
                if isinstance(val, list):
                    values.extend(val)
                else:
                    values.append(val)
            # Add explicit sidebar filters (these take precedence/constrain)
            if filters and key in filters:
                val = filters[key]
                if isinstance(val, list):
                    values.extend(val)
                else:
                    values.append(val)
            # Deduplicate (case-insensitive for strings)
            if values:
                seen = set()
                deduped = []
                for v in values:
                    v_lower = v.lower() if isinstance(v, str) else str(v).lower()
                    if v_lower not in seen:
                        seen.add(v_lower)
                        deduped.append(v)
                combined_filters[key] = deduped
        
        # Add multiple property types if found from query
        if property_types:
            existing = combined_filters.get("property_type", [])
            for pt in property_types:
                if pt.lower() not in [x.lower() for x in existing]:
                    existing.append(pt)
            combined_filters["property_type"] = existing
        
        return self.retriever.retrieve_for_listing(
            question=question,
            filters=combined_filters,
            top_k=None  # Retrieve ALL matching records
        )
    
    def _retrieve_for_comparison(self, question: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for comparison queries - retrieve each named project independently."""
        # Extract project names from query
        project_names = self._extract_project_names(question)
        
        if len(project_names) >= 2:
            all_docs = []
            for project_name in project_names[:3]:  # Limit to 3 projects
                # Retrieve using targeted query WITHOUT incidental location filters
                # Use metadata-only retrieval with project title filter for exact match
                docs = self.retriever.retrieve_for_listing(
                    question=project_name,
                    title=project_name,
                    top_k=5
                )
                if docs:
                    all_docs.extend(docs[:1])  # Take top result for each
            
            # Deduplicate by document_id
            seen = set()
            unique_docs = []
            for doc in all_docs:
                doc_id = doc.get("document_id")
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            
            if len(unique_docs) >= 2:
                return unique_docs
        
        # Fallback to regular retrieval
        return self.retriever.retrieve(question, filters)
    
    def _extract_project_names(self, question: str) -> List[str]:
        """Extract potential project names from comparison query using dynamic resolution."""
        # Get all indexed titles from vector store
        indexed_titles = self.vector_store.get_all_titles()
        if not indexed_titles:
            self.logger.warning("No indexed titles available for project resolution")
            return []
        
        # Build normalized title map for matching
        # Key: normalized title (lowercase, punctuation removed), Value: canonical title
        title_map = {}
        for item in indexed_titles:
            canonical = item["title"]
            normalized = self._normalize_title(canonical)
            title_map[normalized] = canonical
            
            # Also add common abbreviations/partial matches
            # e.g., "the astera" -> "The Astera, Interiors by Aston Martin"
            words = self._normalize_title(canonical).split()
            # Add first significant words as potential matches
            if len(words) > 1:
                # First word if it's distinctive (not "the", "a", "an")
                if words[0] not in {"the", "a", "an"}:
                    title_map[words[0]] = canonical
                # First two words
                if len(words) >= 2:
                    title_map[" ".join(words[:2])] = canonical
                # First three words
                if len(words) >= 3:
                    title_map[" ".join(words[:3])] = canonical
        
        question_lower = question.lower()
        found = []
        matched_keys = set()
        
        # Sort keys by length descending to prefer longer matches
        sorted_keys = sorted(title_map.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            if key in question_lower and key not in matched_keys:
                # Check if this key is a substring of an already matched longer key
                is_substring = any(key in mk for mk in matched_keys if len(mk) > len(key))
                if not is_substring:
                    found.append(title_map[key])
                    matched_keys.add(key)
        
        self.logger.debug(f"Resolved project names: {found} from query: {question}")
        return found

    def _normalize_title(self, title: str) -> str:
        """Normalize title for matching: lowercase, remove punctuation."""
        import re
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', ' ', title.lower())
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _retrieve_for_detail(self, question: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve for detail queries - standard retrieval with slightly higher k."""
        original_top_k = self.retrieval_config.top_k
        self.retrieval_config.top_k = 8
        try:
            return self.retriever.retrieve(question, filters)
        finally:
            self.retrieval_config.top_k = original_top_k
    
    def _generate_answer(self, question: str, context: RetrievalContext, query_type: str) -> str:
        """Generate answer using LLM with context."""
        
        if query_type == "listing":
            # Check if user asks for property type classification not in dataset
            question_lower = question.lower()
            asks_property_classification = any(term in question_lower for term in [
                "luxury residential", "residential projects", "commercial projects",
                "luxury projects", "residential", "commercial", "villa", "apartment"
            ])
            
            user_prompt = f"""Question: {question}

Context from retrieved documents:
{context.context_text}

Instructions:
- Answer using ONLY the information in the context above.
- This is a LISTING query - provide a structured list of all matching projects.
- Format as a clear list with: Project name, Location, Type, and Source URL.
- Include all retrieved projects in your response.
- If the context doesn't contain enough information, say so explicitly.
- Cite sources using the format [Source 1], [Source 2], etc.
- Do not use external knowledge.
- If asked about specific details not in context (price, bedrooms, area), say the information is not available in the collected dataset.
"""
            # Add note about property classification limitations
            if asks_property_classification:
                user_prompt += "\nNOTE: The dataset does not contain a reliable property classification field (property_type is missing for most records). Do not filter or categorize projects as residential/commercial/luxury based on external knowledge. Present all matching projects with their available metadata only."
        elif query_type == "comparison":
            user_prompt = f"""Question: {question}

Context from retrieved documents:
{context.context_text}

Instructions:
- Answer using ONLY the information in the context above.
- This is a COMPARISON query - provide a side-by-side comparison of the projects.
- Structure your response with clear sections for each project.
- Compare on available attributes: location, type, price, description, etc.
- Explicitly state when information is missing for one or both projects.
- Cite sources using the format [Source 1], [Source 2], etc.
- Do not use external knowledge.
- If asked about specific details not in context (price, bedrooms, area), say the information is not available in the collected dataset.
"""
        elif query_type == "detail":
            user_prompt = f"""Question: {question}

Context from retrieved documents:
{context.context_text}

Instructions:
- Answer using ONLY the information in the context above.
- This is a DETAIL query - provide comprehensive information about the specific project.
- Include all available details: location, description, type, price, etc.
- If the context doesn't contain enough information, say so explicitly.
- Cite sources using the format [Source 1], [Source 2], etc.
- Do not use external knowledge.
- If asked about specific details not in context (price, bedrooms, area), say the information is not available in the collected dataset.
"""
        else:
            user_prompt = f"""Question: {question}

Context from retrieved documents:
{context.context_text}

Instructions:
- Answer using ONLY the information in the context above.
- If the context doesn't contain enough information, say so explicitly.
- Cite sources using the format [Source 1], [Source 2], etc.
- Do not use external knowledge.
- If asked about specific details not in context (price, bedrooms, area), say the information is not available in the collected dataset.
"""
        
        try:
            response = self.openrouter_client.generate_simple(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Provide a fallback response when LLM is not available
            if not self.openrouter_client.is_configured():
                # Check if the question asks about specific missing details
                question_lower = question.lower()
                missing_keywords = ["price", "bedroom", "bathroom", "area", "sqm", "sqft", "cost"]
                if any(kw in question_lower for kw in missing_keywords):
                    return ("The requested information (price, bedrooms, area, etc.) is not available "
                            "in the collected dataset. The dataset contains project descriptions and "
                            "locations but does not include pricing or detailed specifications.")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _format_sources(self, sources) -> List[Dict[str, Any]]:
        """Format sources for response."""
        return [
            {
                "label": s.label,
                "title": s.title,
                "url": s.source_url,
                "location": {
                    "city": s.city,
                    "district": s.district,
                    "community": s.community,
                    "country": s.country
                },
                "property_type": s.property_type,
                "price": s.price,
                "currency": s.currency,
                "image_url": s.image_url
            }
            for s in sources
        ]
    
    def _format_retrieved_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format retrieved document for response."""
        metadata = doc.get("metadata", {})
        return {
            "document_id": doc.get("document_id"),
            "title": metadata.get("title"),
            "source_url": doc.get("source_url", metadata.get("source_url")),
            "similarity_score": round(doc.get("similarity_score", 0), 4),
            "city": metadata.get("city"),
            "country": metadata.get("country"),
            "community": metadata.get("community"),
            "property_type": metadata.get("property_type"),
            "metadata": metadata  # Include full metadata for tests
        }
    
    def _no_results_response(self, question: str, query_type: str) -> RAGResponse:
        """Generate response when no results found."""
        if query_type == "listing":
            answer = (
                "I couldn't find any projects matching your criteria in the DarGlobal dataset. "
                "The dataset currently contains 29 verified DarGlobal projects. "
                "Try broadening your location filters or checking the spelling of locations."
            )
        elif query_type == "comparison":
            answer = (
                "I couldn't find enough information to compare the requested projects. "
                "The dataset contains 29 verified DarGlobal projects. "
                "Make sure the project names are spelled correctly and exist in the dataset."
            )
        else:
            answer = (
                "I couldn't find any relevant information in the DarGlobal dataset for your question. "
                "The dataset currently contains 29 verified DarGlobal projects. "
                "Try rephrasing your question or asking about a specific project, location, or community."
            )
        
        return RAGResponse(
            answer=answer,
            sources=[],
            retrieved_documents=[],
            filters_applied={},
            model=self.openrouter_client.model,
            retrieval_count=0,
            context_length=0,
            query_type=query_type
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.vector_store.get_collection_stats()
        stats["embedding_model"] = self.vector_store.embedding_model
        return stats


def get_rag_pipeline() -> RAGPipeline:
    """Get configured RAG pipeline instance."""
    return RAGPipeline()