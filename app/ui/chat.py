"""Streamlit chat UI for RealtyAI."""
import os
import streamlit as st
from typing import List, Dict, Any, Optional
import json
from loguru import logger

from app.rag.pipeline import RAGPipeline, get_rag_pipeline, RAGResponse
from app.rag.retriever import HybridRetriever


# Filter widget keys used in the sidebar
FILTER_WIDGET_KEYS = {
    "source_filter_widget",
    "country_filter_widget", 
    "city_filter_widget",
    "community_filter_widget"
}


def build_filters_from_sidebar() -> Dict[str, Any]:
    """Build canonical filters from Streamlit sidebar widget values.
    
    Reads from session_state widget keys and converts to canonical filter format:
    {
        "source": "DarGlobal" or None,
        "country": ["Oman", "Saudi Arabia"] or [],
        "city": ["Riyadh", "Jeddah"] or [],
        "community": ["Aida", "Rayana"] or []
    }
    
    This is the canonical filter format expected by the pipeline and retriever.
    """
    # Build raw filters from session state (with plural keys as used by widgets)
    raw_filters = {}
    
    # Source filter - single select
    if "source_filter_widget" in st.session_state:
        val = st.session_state.source_filter_widget
        if val != "All":
            raw_filters["source"] = val
    
    # Country filters - multi-select
    if "country_filter_widget" in st.session_state:
        val = st.session_state.country_filter_widget
        if val:
            raw_filters["countries"] = val
    
    # City filters - multi-select
    if "city_filter_widget" in st.session_state:
        val = st.session_state.city_filter_widget
        if val:
            raw_filters["cities"] = val
    
    # Community filters - multi-select
    if "community_filter_widget" in st.session_state:
        val = st.session_state.community_filter_widget
        if val:
            raw_filters["communities"] = val
    
    # Normalize to canonical form using HybridRetriever.normalize_filters
    return HybridRetriever.normalize_filters(raw_filters)


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "pipeline_initialized" not in st.session_state:
        st.session_state.pipeline_initialized = False
    # Initialize filter session state keys
    if "source_filter" not in st.session_state:
        st.session_state.source_filter = None
    if "country_filter" not in st.session_state:
        st.session_state.country_filter = "All"
    if "city_filter" not in st.session_state:
        st.session_state.city_filter = "All"
    if "community_filter" not in st.session_state:
        st.session_state.community_filter = "All"


@st.cache_resource(show_spinner=False)
def get_shared_pipeline() -> RAGPipeline:
    """Create one process-wide pipeline and reuse its model/database resources."""
    pipeline = get_rag_pipeline()
    pipeline.initialize()
    return pipeline


def get_pipeline() -> RAGPipeline:
    """Get the cached process-wide pipeline for this Streamlit session."""
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initializing shared RAG pipeline..."):
            st.session_state.rag_pipeline = get_shared_pipeline()
            st.session_state.pipeline_initialized = True
    return st.session_state.rag_pipeline


def display_source_card(source: Dict[str, Any], index: int):
    """Display a source card in the UI."""
    with st.expander(f"📄 {source['label']}: {source['title']}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Location
            location_parts = []
            loc = source.get("location", {})
            if loc.get("city"):
                location_parts.append(loc["city"])
            if loc.get("district"):
                location_parts.append(loc["district"])
            if loc.get("community"):
                location_parts.append(loc["community"])
            if loc.get("country"):
                location_parts.append(loc["country"])
            
            if location_parts:
                st.markdown(f"**📍 Location:** {', '.join(location_parts)}")
            
            # Property type
            if source.get("property_type"):
                st.markdown(f"**🏠 Property Type:** {source['property_type']}")
            
            # Price
            if source.get("price") is not None and source.get("currency"):
                st.markdown(f"**💰 Price:** {source['price']:,.0f} {source['currency']}")
            
            # Source URL
            if source.get("url"):
                st.markdown(f"**🔗 Source:** [{source['url']}]({source['url']})")
        
        with col2:
            # Image if available
            if source.get("image_url"):
                try:
                    st.image(source["image_url"], caption=source["title"], use_container_width=True)
                except Exception:
                    st.caption("Image unavailable")


def render_sidebar():
    """Render sidebar with filters and info."""
    with st.sidebar:
        st.header("🔧 Filters")
        
        # Check if OpenRouter is configured
        openrouter_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
        
        # Status indicator
        if st.session_state.pipeline_initialized:
            if os.environ.get("OPENROUTER_API_KEY"):
                st.success("🟢 RAG Mode — AI answers with retrieved sources")
            else:
                st.info("🔍 Retrieval Mode — Project search available (OpenRouter not configured)")
        else:
            st.info("🔄 Initializing pipeline...")
        
        st.divider()
        
        # Source filter (only DarGlobal available)
        source_filter = st.selectbox(
            "Source",
            ["All", "DarGlobal"],  # Wasalt not available
            index=0,
            key="source_filter_widget"
        )
        
        # Country filter - now supports multi-select for OR logic
        country_filter = st.multiselect(
            "Country",
            ["Oman", "Saudi Arabia", "United Arab Emirates", "Qatar", "United Kingdom", "Spain"],
            default=[],
            key="country_filter_widget",
            help="Select one or more countries (OR logic)"
        )
        
        # City filter - now supports multi-select for OR logic
        city_filter = st.multiselect(
            "City",
            ["Muscat", "Jeddah", "Dubai", "Riyadh", "Doha", "London", "Benahavís", "Al Marjan Island", "Simaisma"],
            default=[],
            key="city_filter_widget",
            help="Select one or more cities (OR logic)"
        )
        
        # Community filter - now supports multi-select for OR logic
        community_filter = st.multiselect(
            "Community",
            ["Aida", "Rayana", "Amaya", "The Astera", "The Mulliner", "La Collina", "Trump Plaza", "One of One"],
            default=[],
            key="community_filter_widget",
            help="Select one or more communities (OR logic)"
        )
        
        st.divider()
        
        # Stats
        if st.session_state.pipeline_initialized:
            pipeline = st.session_state.rag_pipeline
            stats = pipeline.get_stats()
            st.subheader("📊 Index Stats")
            st.metric("Indexed Documents", stats.get("document_count", 0))
            st.metric("Embedding Model", stats.get("embedding_model", "N/A"))
            st.metric("Embedding Dim", stats.get("embedding_dim", "N/A"))
        
        st.divider()
        
        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "Tell me about The Astera",
            "Show DarGlobal projects in Oman",
            "Which projects are located in Riyadh?",
            "Compare projects in Oman and Saudi Arabia",
            "What information is available about AIDA?",
            "Which projects have published price information?",
            "Tell me about Trump Tower Jeddah",
            "What projects are in the AIDA community?",
            "Show me projects in Dubai"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Convert filters to canonical format for pipeline
        raw_filters = {}
        # Source is single select
        if source_filter != "All":
            raw_filters["source"] = source_filter
        # Countries - list for OR logic
        if country_filter:
            raw_filters["countries"] = country_filter
        # Cities - list for OR logic
        if city_filter:
            raw_filters["cities"] = city_filter
        # Communities - list for OR logic
        if community_filter:
            raw_filters["communities"] = community_filter
        
        # Normalize to canonical form
        return HybridRetriever.normalize_filters(raw_filters)


def render_chat():
    """Render main chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if assistant message
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    st.markdown("---")
                    st.subheader("📚 Sources")
                    for source in message["sources"]:
                        display_source_card(source, 0)
    
    # Handle pending question from sidebar
    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question
        process_question(question)
    
    # Chat input
    if prompt := st.chat_input("Ask about DarGlobal projects..."):
        process_question(prompt)


def process_question(question: str):
    """Process user question through RAG pipeline."""
    if not st.session_state.pipeline_initialized:
        st.error("Pipeline not initialized. Please wait...")
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Get pipeline from session state (already initialized at startup)
    pipeline = st.session_state.rag_pipeline
    
    with st.chat_message("assistant"):
        with st.spinner("Searching projects and generating response..."):
            # Get sidebar filters from session state (set by render_sidebar)
            raw_filters = {}
            # Source filter
            if "source_filter_widget" in st.session_state:
                val = st.session_state.source_filter_widget
                if val != "All":
                    raw_filters["source"] = val
            
            # Country filters (multi-select)
            if "country_filter_widget" in st.session_state:
                val = st.session_state.country_filter_widget
                if val:
                    raw_filters["countries"] = val
            
            # City filters (multi-select)
            if "city_filter_widget" in st.session_state:
                val = st.session_state.city_filter_widget
                if val:
                    raw_filters["cities"] = val
            
            # Community filters (multi-select)
            if "community_filter_widget" in st.session_state:
                val = st.session_state.community_filter_widget
                if val:
                    raw_filters["communities"] = val
            
            # Normalize to canonical form
            filters = HybridRetriever.normalize_filters(raw_filters)
            
            # Process through RAG pipeline
            response = pipeline.ask(question, filters)
            
            # Display answer
            st.markdown(response.answer)
            
            # Display sources
            if response.sources:
                st.markdown("---")
                st.subheader("📚 Sources")
                for source in response.sources:
                    display_source_card(source, 0)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources
            })


def render_header():
    """Render app header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏠 RealtyAI")
        st.caption("AI-powered Real Estate Discovery Assistant")
    with col2:
        st.caption("Data: DarGlobal")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RealtyAI",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize RAG pipeline at startup (only once per session)
    if not st.session_state.pipeline_initialized:
        with st.spinner("Initializing RAG pipeline... Loading embedding model and ChromaDB index..."):
            try:
                pipeline = get_pipeline()
                st.session_state.rag_pipeline = pipeline
                st.session_state.pipeline_initialized = True
                st.toast("RAG pipeline initialized successfully", icon="✅")
            except Exception as e:
                logger.exception("Pipeline initialization failed")
                st.error(f"Failed to initialize RAG pipeline: {str(e)}")
                st.stop()
    
    # Render header
    render_header()
    
    # Render sidebar and get filters
    filters = render_sidebar()
    
    # Render chat
    render_chat()


if __name__ == "__main__":
    main()