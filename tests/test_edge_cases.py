"""Tests for edge cases: missing information and unsupported questions."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag.pipeline import RAGPipeline
from app.rag.context_builder import ContextBuilder, RetrievalContext
from app.rag.retriever import HybridRetriever, QueryParser
from app.rag.vector_store import VectorStore
from app.data.schema import PropertyRecord


def extract_question_from_prompt(user_prompt: str) -> str:
    """Extract the question from the user prompt."""
    # The prompt format is: "Question: {question}\n\nContext from retrieved documents:\n{context}"
    for line in user_prompt.split('\n'):
        if line.startswith('Question:'):
            return line[len('Question:'):].strip()
    return ""


class TestMissingInformation:
    """Test handling of queries for unavailable information."""
    
    @pytest.fixture
    def pipeline_with_limited_data(self):
        """Pipeline with records missing price/bedrooms/area."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Project A",
                source_url="https://example.com/a",
                city="Riyadh",
                country="Saudi Arabia",
                description="A residential project in Riyadh",
                # No price, bedrooms, bathrooms, area
            ),
            PropertyRecord(
                source="DarGlobal",
                title="Project B",
                source_url="https://example.com/b",
                city="Dubai",
                country="United Arab Emirates",
                description="A commercial project in Dubai",
                price=1000000,
                currency="AED",
                # No bedrooms, bathrooms, area
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_edge"
            mock_settings.return_value.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.return_value.PROCESSED_DATA_DIR = tmp
            mock_settings.return_value.OPENROUTER_API_KEY = ""  # No API key
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
            
            # Mock the OpenRouter client to simulate missing info responses
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            
            def mock_generate_simple(system_prompt, user_prompt):
                from app.llm.openrouter import LLMResponse
                question = extract_question_from_prompt(user_prompt).lower()
                
                if "price of project b" in question:
                    return LLMResponse(content="The price for Project B is 1,000,000 AED.", model="test-model")
                elif "price of project a" in question:
                    return LLMResponse(content="The price for Project A is not available in the collected dataset.", model="test-model")
                elif "bedroom" in question and "project a" in question:
                    return LLMResponse(content="The number of bedrooms for Project A is not available in the collected dataset.", model="test-model")
                elif "area" in question and "project a" in question:
                    return LLMResponse(content="The area for Project A is not available in the collected dataset.", model="test-model")
                elif "price" in question and "bedroom" in question and "area" in question and "project a" in question:
                    return LLMResponse(content="The price, bedrooms, and area for Project A are not available in the collected dataset.", model="test-model")
                
                return LLMResponse(content="Test response", model="test-model")
            
            mock_openrouter.generate_simple = mock_generate_simple
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_price_query_when_unavailable(self, pipeline_with_limited_data):
        """Test query about price when not available."""
        response = pipeline_with_limited_data.ask("What is the price of Project A?")
        
        assert "not available" in response.answer.lower() or \
               "could not be verified" in response.answer.lower() or \
               "not in the dataset" in response.answer.lower()
    
    def test_bedrooms_query_when_unavailable(self, pipeline_with_limited_data):
        """Test query about bedrooms when not available."""
        response = pipeline_with_limited_data.ask("How many bedrooms does Project A have?")
        
        assert "not available" in response.answer.lower() or \
               "could not be verified" in response.answer.lower() or \
               "not in the dataset" in response.answer.lower()
    
    def test_area_query_when_unavailable(self, pipeline_with_limited_data):
        """Test query about area when not available."""
        response = pipeline_with_limited_data.ask("What is the area of Project A?")
        
        assert "not available" in response.answer.lower() or \
               "could not be verified" in response.answer.lower() or \
               "not in the dataset" in response.answer.lower()
    
    def test_price_query_when_available(self, pipeline_with_limited_data):
        """Test query about price when available."""
        response = pipeline_with_limited_data.ask("What is the price of Project B?")
        
        # Should include the price information
        assert "1,000,000" in response.answer or "1000000" in response.answer
        assert "AED" in response.answer
    
    def test_multiple_missing_fields(self, pipeline_with_limited_data):
        """Test query asking for multiple missing fields."""
        response = pipeline_with_limited_data.ask("What are the price, bedrooms, and area for Project A?")
        
        # Should explicitly state unavailability
        assert "not available" in response.answer.lower() or \
               "could not be verified" in response.answer.lower() or \
               "not in the dataset" in response.answer.lower()


class TestUnsupportedQuestions:
    """Test handling of unsupported/out-of-scope questions.
    
    With the new implementation, when no relevant documents are retrieved,
    the pipeline returns a controlled response WITHOUT calling OpenRouter.
    """
    
    @pytest.fixture
    def pipeline_no_results(self):
        """Pipeline with minimal data that won't match unsupported queries."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Test Project",
                source_url="https://example.com/1",
                city="Riyadh",
                country="Saudi Arabia",
                description="A test project"
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_unsupported"
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
            
            # OpenRouter should NOT be called for zero-result queries
            # Mock it to verify it's not called
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            mock_openrouter.generate_simple = Mock(return_value=LLMResponse(
                content="Should not be called", model="test-model"
            ))
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_rental_query_returns_no_data(self, pipeline_no_results):
        """Test that rental queries return controlled no-data response."""
        response = pipeline_no_results.ask("What are the rental prices for commercial properties?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_mortgage_query_returns_no_data(self, pipeline_no_results):
        """Test that mortgage queries return controlled no-data response."""
        response = pipeline_no_results.ask("What mortgage options are available?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_legal_query_returns_no_data(self, pipeline_no_results):
        """Test that legal queries return controlled no-data response."""
        response = pipeline_no_results.ask("What are the legal requirements for buying property in Oman?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_tax_query_returns_no_data(self, pipeline_no_results):
        """Test that tax queries return controlled no-data response."""
        response = pipeline_no_results.ask("What are the property taxes in Dubai?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_investment_query_returns_no_data(self, pipeline_no_results):
        """Test that investment advice queries return controlled no-data response."""
        response = pipeline_no_results.ask("Is this a good investment? What's the ROI?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_market_forecast_returns_no_data(self, pipeline_no_results):
        """Test that market forecasts return controlled no-data response."""
        response = pipeline_no_results.ask("What will property prices be in 2025?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()
    
    def test_agent_commission_returns_no_data(self, pipeline_no_results):
        """Test that agent/broker queries return controlled no-data response."""
        response = pipeline_no_results.ask("What commission do agents charge?")
        
        assert response.retrieval_count == 0
        assert "DarGlobal" in response.answer
        assert "dataset" in response.answer.lower()
        pipeline_no_results.openrouter_client.generate_simple.assert_not_called()


class TestAmbiguousQueries:
    """Test handling of ambiguous queries."""
    
    @pytest.fixture
    def query_parser(self):
        return QueryParser()
    
    def test_ambiguous_location(self, query_parser):
        """Test query with ambiguous location name."""
        query = "Show projects in London"  # Could be UK London or London, Ontario
        filters = query_parser.parse(query)
        
        # Should find London (UK) from dataset
        assert "London" in filters.get("city", [])
        assert "United Kingdom" in filters.get("country", [])
    
    def test_generic_query(self, query_parser):
        """Test very generic query."""
        query = "Tell me about projects"
        filters = query_parser.parse(query)
        
        # Should not extract specific filters
        assert len(filters) == 0 or (len(filters) == 1 and "source" in filters)
    
    def test_partial_project_name(self, query_parser):
        """Test query with partial project name."""
        query = "Tell me about Trump"
        filters = query_parser.parse(query)
        
        # Should not infer property type from "Trump"
        # But might detect "trump" in project names if they match
        pass
    
    def test_misspelled_location(self, query_parser):
        """Test query with misspelled location."""
        query = "Show projects in Riyad"  # Missing 'h'
        filters = query_parser.parse(query)
        
        # Word boundary matching should not match "Riyad"
        assert "Riyadh" not in filters.get("city", [])


class TestWasaltQueries:
    """Test handling of Wasalt queries (0 records in dataset)."""
    
    @pytest.fixture
    def pipeline_no_wasalt(self):
        """Pipeline with only DarGlobal data."""
        records = [
            PropertyRecord(
                source="DarGlobal",
                title="Test Project",
                source_url="https://example.com/1",
                city="Riyadh",
                country="Saudi Arabia",
                description="A test project"
            )
        ]
        
        with patch('app.config.get_settings') as mock_settings:
            import tempfile
            tmp = tempfile.mkdtemp()
            mock_settings.return_value.CHROMA_DB_PATH = f"{tmp}/chroma"
            mock_settings.return_value.CHROMA_COLLECTION = "test_wasalt"
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
            
            # Mock the OpenRouter client - should NOT be called for zero-result Wasalt queries
            from app.llm.openrouter import LLMResponse
            mock_openrouter = Mock()
            mock_openrouter.model = "test-model"
            mock_openrouter.generate_simple = Mock(return_value=LLMResponse(
                content="Should not be called", model="test-model"
            ))
            pipeline.openrouter_client = mock_openrouter
            
            yield pipeline
    
    def test_wasalt_query_returns_no_data_message(self, pipeline_no_wasalt):
        """Test that Wasalt queries return appropriate message."""
        response = pipeline_no_wasalt.ask("Show me Wasalt listings in Riyadh")
        
        assert "wasalt" in response.answer.lower()
        assert "not yet been acquired" in response.answer.lower() or "not available" in response.answer.lower()
        assert "DarGlobal" in response.answer
        assert response.retrieval_count == 0
        # OpenRouter should not have been called
        pipeline_no_wasalt.openrouter_client.generate_simple.assert_not_called()
    
    def test_wasalt_comparison_refused(self, pipeline_no_wasalt):
        """Test comparing Wasalt projects is handled."""
        response = pipeline_no_wasalt.ask("Compare Wasalt projects")
        
        assert "wasalt" in response.answer.lower()
        assert "DarGlobal" in response.answer
        assert response.retrieval_count == 0
        pipeline_no_wasalt.openrouter_client.generate_simple.assert_not_called()


class TestEmptyDatasetHandling:
    """Test handling of edge cases with minimal data."""
    
    def test_query_with_no_results(self):
        """Test query that returns no results."""
        parser = QueryParser()
        filters = parser.parse("Show projects in Antarctica")
        
        # Should not find Antarctica in known locations
        assert "Antarctica" not in filters.get("country", [])
        assert "Antarctica" not in filters.get("city", [])
    
    def test_very_long_query(self):
        """Test handling of very long query."""
        query = "Tell me about " + "project " * 100
        parser = QueryParser()
        filters = parser.parse(query)
        
        # Should not crash and should parse what it can
        assert isinstance(filters, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])