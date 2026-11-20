"""Tests for multiple-project comparisons and location-based listings."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever, QueryParser
from app.rag.vector_store import VectorStore
from app.data.schema import PropertyRecord


class TestComparisonQueries:
    """Test comparison query handling."""
    
    @pytest.fixture
    def comparison_pipeline(self):
        """Pipeline with multiple projects for comparison."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="The Astera, Interiors by Aston Martin",
                source_url="https://darglobal.co.uk/the-astera",
                city="Dubai",
                country="United Arab Emirates",
                description="The Astera is a luxury residential project with premium amenities.",
                property_type="Residential",
                price=2000000,
                currency="AED",
                bedrooms=3,
                bathrooms=2,
                area=180,
                area_unit="sqm",
                amenities=["Pool", "Gym", "Security", "Parking"],
                status="Under Construction",
                image_url="https://example.com/astera.jpg",
                community="Aida",
                district="Al Rimal"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="La Collina",
                source_url="https://darglobal.co.uk/la-collina",
                city="Jeddah",
                country="Saudi Arabia",
                description="La Collina is a mixed-use development with residential and commercial.",
                property_type="Mixed-use",
                price=1500000,
                currency="SAR",
                bedrooms=2,
                bathrooms=2,
                area=120,
                area_unit="sqm",
                amenities=["Park", "Retail", "Security"],
                status="Planning",
                image_url="https://example.com/lacollina.jpg",
                community="Rayana",
                district="Al Salamah"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Trump Tower Jeddah",
                source_url="https://darglobal.co.uk/trump-tower-jeddah",
                city="Jeddah",
                country="Saudi Arabia",
                description="Trump Tower Jeddah is a luxury hotel and residence tower.",
                property_type="Hotel",
                price=3000000,
                currency="SAR",
                bedrooms=4,
                bathrooms=3,
                area=250,
                area_unit="sqm",
                amenities=["Spa", "Pool", "Concierge", "Security"],
                status="Under Construction",
                image_url="https://example.com/trumptower.jpg",
                community="Amaya",
                district="Al Hamra"
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_comparison"
            mock_settings.return_value.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value.PROCESSED_DATA_DIR = tmp
            mock_settings.return_value.OPENROUTER_API_KEY = ""
            mock_settings.return_value.OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
            mock_settings.return_value.OPENROUTER_FALLBACK_MODEL = "openrouter/free"
            mock_settings.return_value.OPENROUTER_SITE_URL = "https://example.com"
            mock_settings.return_value.OPENROUTER_APP_NAME = "RealtyAI"
            mock_settings.return_value.LLM_TEMPERATURE = 0.2
            mock_settings.return_value.LLM_MAX_TOKENS = 1200
            mock_settings.return_value.LLM_TIMEOUT = 60
            
            pipeline = RAGPipeline()
            from app.rag.document_builder import DocumentBuilder
            builder = DocumentBuilder()
            documents = builder.build_documents(records)
            
            vector_store = VectorStore(persist_dir=f"{tmp}/chroma")
            vector_store.build_index(documents, rebuild=True)
            
            pipeline.vector_store = vector_store
            pipeline.retriever = HybridRetriever(vector_store)
            
            # Mock the OpenRouter client to return comparison responses
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            
            def mock_generate_simple(system_prompt, user_prompt):
                from app.llm.openrouter import LLMResponse
                if "comparison" in user_prompt.lower() or "compare" in user_prompt.lower():
                    return LLMResponse(
                        content="Comparison of The Astera and La Collina:\n\n**The Astera** (Dubai, UAE): Luxury residential project, 3 bedrooms, 2 bathrooms, 180 sqm, 2,000,000 AED. [Source 1]\n\n**La Collina** (Jeddah, Saudi Arabia): Mixed-use development, 2 bedrooms, 2 bathrooms, 120 sqm, 1,500,000 SAR. [Source 2]\n\nKey differences: The Astera is residential-only in Dubai, while La Collina is mixed-use in Jeddah. The Astera is larger and more expensive.",
                        model="test-model"
                    )
                return LLMResponse(content="Test response", model="test-model")
            
            mock_openrouter.generate_simple = mock_generate_simple
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_explicit_comparison_two_projects(self, comparison_pipeline):
        """Test explicit comparison of two named projects."""
        response = comparison_pipeline.ask("Compare The Astera and La Collina")
        
        assert response.query_type == "comparison"
        assert response.retrieval_count >= 2
        
        # Should find both projects (using actual indexed title)
        titles = [doc.get("metadata", {}).get("title", "") for doc in response.retrieved_documents]
        assert "The Astera, Interiors by Aston Martin" in titles
        assert "La Collina" in titles
    
    def test_comparison_with_vs(self, comparison_pipeline):
        """Test comparison using 'vs' keyword."""
        response = comparison_pipeline.ask("The Astera, Interiors by Aston Martin vs La Collina")
        
        assert response.query_type == "comparison"
        assert response.retrieval_count >= 2
    
    def test_comparison_versus(self, comparison_pipeline):
        """Test comparison using 'versus' keyword."""
        response = comparison_pipeline.ask("The Astera, Interiors by Aston Martin versus Trump Tower Jeddah")
        
        assert response.query_type == "comparison"
        assert response.retrieval_count >= 2
    
    def test_comparison_three_projects(self, comparison_pipeline):
        """Test comparison of three projects."""
        response = comparison_pipeline.ask("Compare The Astera, Interiors by Aston Martin, La Collina, and Trump Tower Jeddah")
        
        assert response.query_type == "comparison"
        assert response.retrieval_count >= 3
    
    def test_comparison_by_location(self, comparison_pipeline):
        """Test comparison of projects in same location."""
        response = comparison_pipeline.ask("Compare projects in Jeddah")
        
        assert response.query_type in ["comparison", "listing"]
        # Should find La Collina and Trump Tower Jeddah
        titles = [doc.get("metadata", {}).get("title", "") for doc in response.retrieved_documents]
        assert "La Collina" in titles
        assert "Trump Tower Jeddah" in titles
    
    def test_comparison_response_structure(self, comparison_pipeline):
        """Test that comparison response has side-by-side structure."""
        response = comparison_pipeline.ask("Compare The Astera, Interiors by Aston Martin and La Collina")
        
        # Response should mention both projects (case-insensitive)
        answer_lower = response.answer.lower()
        assert "the astera" in answer_lower or "the asteria" in answer_lower
        assert "la collina" in answer_lower
        
        # Should have sources for both
        assert len(response.sources) >= 2


class TestListingQueries:
    """Test listing query handling."""
    
    @pytest.fixture
    def listing_pipeline(self):
        """Pipeline with multiple projects for listing."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Project 1",
                source_url="https://example.com/1",
                city="Riyadh",
                country="Saudi Arabia",
                description="Project in Riyadh",
                property_type="Residential",
                price=1000000,
                currency="SAR",
                community="Aida"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Project 2",
                source_url="https://example.com/2",
                city="Riyadh",
                country="Saudi Arabia",
                description="Another project in Riyadh",
                property_type="Commercial",
                price=2000000,
                currency="SAR",
                community="Rayana"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Project 3",
                source_url="https://example.com/3",
                city="Dubai",
                country="United Arab Emirates",
                description="Project in Dubai",
                property_type="Residential",
                price=1500000,
                currency="AED",
                community="Amaya"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Project 4",
                source_url="https://example.com/4",
                city="Muscat",
                country="Oman",
                description="Project in Muscat",
                property_type="Mixed-use",
                price=800000,
                currency="OMR",
                community="Aida"
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_listing"
            mock_settings.return_value.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value.PROCESSED_DATA_DIR = tmp
            mock_settings.return_value.OPENROUTER_API_KEY = ""
            mock_settings.return_value.OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
            mock_settings.return_value.OPENROUTER_FALLBACK_MODEL = "openrouter/free"
            mock_settings.return_value.OPENROUTER_SITE_URL = "https://example.com"
            mock_settings.return_value.OPENROUTER_APP_NAME = "RealtyAI"
            mock_settings.return_value.LLM_TEMPERATURE = 0.2
            mock_settings.return_value.LLM_MAX_TOKENS = 1200
            mock_settings.return_value.LLM_TIMEOUT = 60
            
            pipeline = RAGPipeline()
            from app.rag.document_builder import DocumentBuilder
            builder = DocumentBuilder()
            documents = builder.build_documents(records)
            
            vector_store = VectorStore(persist_dir=f"{tmp}/chroma")
            vector_store.build_index(documents, rebuild=True)
            
            pipeline.vector_store = vector_store
            pipeline.retriever = HybridRetriever(vector_store)
            
            # Mock OpenRouter to return listing responses
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            
            def mock_generate_simple(system_prompt, user_prompt):
                from app.llm.openrouter import LLMResponse
                if "listing" in user_prompt.lower() or "list" in user_prompt.lower():
                    return LLMResponse(
                        content="Here are the projects in Saudi Arabia:\n\n1. **Project 1** (Riyadh, Saudi Arabia) - Residential, 1,000,000 SAR - [Source 1]\n2. **Project 2** (Riyadh, Saudi Arabia) - Commercial, 2,000,000 SAR - [Source 2]\n\nTotal: 2 projects found.",
                        model="test-model"
                    )
                return LLMResponse(content="Test response", model="test-model")
            
            mock_openrouter.generate_simple = mock_generate_simple
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_list_all_projects_in_country(self, listing_pipeline):
        """Test listing all projects in a country."""
        response = listing_pipeline.ask("List all projects in Saudi Arabia")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 2
        
        # All results should be from Saudi Arabia
        for doc in response.retrieved_documents:
            assert doc.get("metadata", {}).get("country") == "Saudi Arabia"
    
    def test_list_all_projects_in_city(self, listing_pipeline):
        """Test listing all projects in a city."""
        response = listing_pipeline.ask("Show all projects in Riyadh")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 2
        
        for doc in response.retrieved_documents:
            assert doc.get("metadata", {}).get("city") == "Riyadh"
    
    def test_list_all_projects_in_community(self, listing_pipeline):
        """Test listing all projects in a community."""
        response = listing_pipeline.ask("What projects are in the Aida community?")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 2
        
        for doc in response.retrieved_documents:
            assert doc.get("metadata", {}).get("community") == "Aida"
    
    def test_list_all_projects(self, listing_pipeline):
        """Test listing all projects."""
        response = listing_pipeline.ask("List all projects")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 4
    
    def test_list_with_multiple_countries(self, listing_pipeline):
        """Test listing with multiple countries."""
        response = listing_pipeline.ask("Show projects in Oman and Saudi Arabia")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 3
        
        countries = {doc.get("metadata", {}).get("country") for doc in response.retrieved_documents}
        assert "Oman" in countries
        assert "Saudi Arabia" in countries
    
    def test_list_with_multiple_cities(self, listing_pipeline):
        """Test listing with multiple cities."""
        response = listing_pipeline.ask("Show projects in Riyadh and Dubai")
        
        assert response.query_type == "listing"
        assert response.retrieval_count >= 3
        
        cities = {doc.get("metadata", {}).get("city") for doc in response.retrieved_documents}
        assert "Riyadh" in cities
        assert "Dubai" in cities
    
    def test_list_response_format(self, listing_pipeline):
        """Test that listing response is structured as a list."""
        response = listing_pipeline.ask("List all projects in Saudi Arabia")
        
        # Should have multiple sources
        assert len(response.sources) >= 2
        
        # Each source should have required fields
        for source in response.sources:
            assert "label" in source
            assert "title" in source
            assert "url" in source
            assert "location" in source


class TestCombinedFilters:
    """Test queries with combined filters."""
    
    @pytest.fixture
    def filter_pipeline(self):
        """Pipeline with diverse projects for filter testing."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Residential Riyadh",
                source_url="https://example.com/1",
                city="Riyadh",
                country="Saudi Arabia",
                description="Residential project in Riyadh",
                property_type="Residential",
                price=1000000,
                currency="SAR",
                community="Aida"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Commercial Riyadh",
                source_url="https://example.com/2",
                city="Riyadh",
                country="Saudi Arabia",
                description="Commercial project in Riyadh",
                property_type="Commercial",
                price=2000000,
                currency="SAR",
                community="Rayana"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Mixed-use Dubai",
                source_url="https://example.com/3",
                city="Dubai",
                country="United Arab Emirates",
                description="Mixed-use project in Dubai",
                property_type="Mixed-use",
                price=1500000,
                currency="AED",
                community="Amaya"
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_filters"
            mock_settings.return_value.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value.PROCESSED_DATA_DIR = tmp
            mock_settings.return_value.OPENROUTER_API_KEY = ""
            mock_settings.return_value.OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
            mock_settings.return_value.OPENROUTER_FALLBACK_MODEL = "openrouter/free"
            mock_settings.return_value.OPENROUTER_SITE_URL = "https://example.com"
            mock_settings.return_value.OPENROUTER_APP_NAME = "RealtyAI"
            mock_settings.return_value.LLM_TEMPERATURE = 0.2
            mock_settings.return_value.LLM_MAX_TOKENS = 1200
            mock_settings.return_value.LLM_TIMEOUT = 60
            
            pipeline = RAGPipeline()
            from app.rag.document_builder import DocumentBuilder
            builder = DocumentBuilder()
            documents = builder.build_documents(records)
            
            vector_store = VectorStore(persist_dir=f"{tmp}/chroma")
            vector_store.build_index(documents, rebuild=True)
            
            pipeline.vector_store = vector_store
            pipeline.retriever = HybridRetriever(vector_store)
            
            # Mock OpenRouter
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            mock_openrouter.generate_simple = Mock(return_value=LLMResponse(content="Test response", model="test-model"))
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_filter_by_country_and_property_type(self, filter_pipeline):
        """Test filtering by country and property type."""
        response = filter_pipeline.ask("Show residential projects in Saudi Arabia")
        
        assert response.retrieval_count >= 1
        for doc in response.retrieved_documents:
            meta = doc.get("metadata", {})
            assert meta.get("country") == "Saudi Arabia"
            assert meta.get("property_type") == "Residential"
    
    def test_filter_by_city_and_community(self, filter_pipeline):
        """Test filtering by city and community."""
        response = filter_pipeline.ask("Show projects in Riyadh in the Aida community")
        
        assert response.retrieval_count >= 1
        for doc in response.retrieved_documents:
            meta = doc.get("metadata", {})
            assert meta.get("city") == "Riyadh"
            assert meta.get("community") == "Aida"
    
    def test_filter_by_property_type_only(self, filter_pipeline):
        """Test filtering by property type only."""
        response = filter_pipeline.ask("List all commercial projects")
        
        assert response.retrieval_count >= 1
        for doc in response.retrieved_documents:
            assert doc.get("metadata", {}).get("property_type") == "Commercial"
    
    def test_filter_multiple_property_types(self, filter_pipeline):
        """Test filtering by multiple property types."""
        response = filter_pipeline.ask("Show residential and commercial projects")
        
        assert response.retrieval_count >= 2
        types = {doc.get("metadata", {}).get("property_type") for doc in response.retrieved_documents}
        assert "Residential" in types
        assert "Commercial" in types


class TestStructuredRetrieval:
    """Test structured retrieval methods."""
    
    @pytest.fixture
    def retriever(self):
        """Create a retriever with test data."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Project 1",
                source_url="https://example.com/1",
                city="Riyadh",
                country="Saudi Arabia",
                description="Project in Riyadh",
                property_type="Residential"
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Project 2",
                source_url="https://example.com/2",
                city="Dubai",
                country="United Arab Emirates",
                description="Project in Dubai",
                property_type="Residential"  # Changed to Residential to match test filter
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_structured"
            mock_settings.return_value.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value.PROCESSED_DATA_DIR = tmp
            
            from app.rag.document_builder import DocumentBuilder
            builder = DocumentBuilder()
            documents = builder.build_documents(records)
            
            vector_store = VectorStore(persist_dir=f"{tmp}/chroma")
            vector_store.build_index(documents, rebuild=True)
            
            retriever = HybridRetriever(vector_store)
            yield retriever
    
    def test_retrieve_with_structured_filters_method(self, retriever):
        """Test retrieve_with_structured_filters method directly."""
        results = retriever.retrieve_with_structured_filters(
            "test query",
            countries=["Saudi Arabia", "UAE"],
            cities=["Riyadh", "Dubai"],
            property_type="Residential"
        )
        
        assert len(results) >= 1
        # Should filter by all criteria
        for doc in results:
            meta = doc.get("metadata", {})
            if meta.get("country"):
                assert meta["country"] in ["Saudi Arabia", "UAE"]
            if meta.get("city"):
                assert meta["city"] in ["Riyadh", "Dubai"]
            if meta.get("property_type"):
                assert meta["property_type"] == "Residential"
    
    def test_retrieve_for_listing_method(self, retriever):
        """Test retrieve_for_listing method."""
        results = retriever.retrieve_for_listing(
            "list projects",
            countries=["Saudi Arabia"],
            top_k=10
        )
        
        assert len(results) >= 1
        for doc in results:
            assert doc.get("metadata", {}).get("country") == "Saudi Arabia"
        # Should not apply similarity threshold
        for doc in results:
            # Scores can be any value since threshold is 0
            assert "similarity_score" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])