"""Tests for RAG pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag.pipeline import RAGPipeline, RAGResponse, get_rag_pipeline
from app.rag.document_builder import RagDocument
from app.data.schema import PropertyRecord
from app.llm.openrouter import LLMResponse


class TestRAGPipeline:
    @pytest.fixture
    def mock_pipeline_components(self):
        with patch('app.rag.pipeline.DocumentBuilder') as mock_doc_builder, \
             patch('app.rag.pipeline.VectorStore') as mock_vector_store, \
             patch('app.rag.pipeline.HybridRetriever') as mock_retriever, \
             patch('app.rag.pipeline.ContextBuilder') as mock_context_builder, \
             patch('app.rag.pipeline.OpenRouterClient') as mock_openrouter, \
             patch('app.rag.pipeline.load_rag_ready_records') as mock_load:

            # Setup mocks
            mock_doc_builder_instance = Mock()
            mock_doc_builder.return_value = mock_doc_builder_instance

            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            mock_retriever_instance = Mock()
            mock_retriever.return_value = mock_retriever_instance

            mock_context_builder_instance = Mock()
            mock_context_builder.return_value = mock_context_builder_instance

            mock_openrouter_instance = Mock()
            mock_openrouter_instance.model = "test-model"
            mock_openrouter.return_value = mock_openrouter_instance

            mock_load.return_value = [
                PropertyRecord(
                    source="DarGlobal",
                    title="Test Project",
                    source_url="https://example.com/1",
                    city="Riyadh",
                    country="Saudi Arabia",
                    description="Test description"
                )
            ]

            yield {
                "doc_builder": mock_doc_builder_instance,
                "vector_store": mock_vector_store_instance,
                "retriever": mock_retriever_instance,
                "context_builder": mock_context_builder_instance,
                "openrouter": mock_openrouter_instance,
                "load_records": mock_load
            }

    @pytest.fixture
    def pipeline(self, mock_pipeline_components):
        with patch('app.rag.pipeline.get_settings') as mock_settings:
            mock_settings.return_value.PROCESSED_DATA_DIR = "dataset/processed"
            pipeline = RAGPipeline()
            yield pipeline

    def test_initialize_builds_index(self, pipeline, mock_pipeline_components):
        """Test pipeline initialization builds index."""
        mock_pipeline_components["vector_store"].needs_rebuild.return_value = True
        mock_pipeline_components["vector_store"].build_index.return_value = 29

        count = pipeline.initialize(rebuild_index=True)

        assert count == 29
        mock_pipeline_components["doc_builder"].build_documents.assert_called_once()
        mock_pipeline_components["vector_store"].build_index.assert_called_once_with(
            mock_pipeline_components["doc_builder"].build_documents.return_value,
            rebuild=True
        )

    def test_initialize_uses_existing_index(self, pipeline, mock_pipeline_components):
        """Test that existing index is used when no rebuild needed."""
        mock_pipeline_components["vector_store"].needs_rebuild.return_value = False
        mock_pipeline_components["vector_store"].get_collection_stats.return_value = {
            "document_count": 29
        }

        count = pipeline.initialize(rebuild_index=False)

        assert count == 29
        mock_pipeline_components["vector_store"].build_index.assert_not_called()

    def test_ask_with_results(self, pipeline, mock_pipeline_components):
        """Test successful question answering with retrieval results."""
        # Setup mock retrieval results
        mock_pipeline_components["retriever"].retrieve.return_value = [
            {
                "document_id": "doc_1",
                "title": "Test Project",
                "content": "Test content",
                "metadata": {
                    "title": "Test Project",
                    "source": "DarGlobal",
                    "city": "Riyadh",
                    "country": "Saudi Arabia",
                    "source_url": "https://example.com"
                },
                "similarity_score": 0.9,
                "source_url": "https://example.com"
            }
        ]

        # Mock context builder
        mock_context = Mock()
        mock_context.sources = [
            Mock(
                label="Source 1",
                title="Test Project",
                source_url="https://example.com",
                city="Riyadh",
                country="Saudi Arabia",
                property_type="Residential",
                price=1000000,
                currency="SAR",
                image_url="https://example.com/img.jpg"
            )
        ]
        mock_context.context_text = "Context text"
        mock_context.retrieval_count = 1
        mock_context.filters_applied = {}
        mock_pipeline_components["context_builder"].build_context.return_value = mock_context

        # Mock OpenRouter response
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_pipeline_components["openrouter"].generate_simple.return_value = mock_response

        response = pipeline.ask("Test question")

        assert "Test answer" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0]["title"] == "Test Project"
        assert response.sources[0]["url"] == "https://example.com"
        assert response.retrieval_count == 1
        assert response.model == "test-model"

    def test_ask_no_results(self, pipeline, mock_pipeline_components):
        """Test response when no results found."""
        mock_pipeline_components["retriever"].retrieve.return_value = []
        
        # Mock OpenRouter to return a proper response
        mock_response = Mock()
        mock_response.content = "I couldn't find any relevant information in the DarGlobal dataset for your question. The dataset currently contains 29 verified DarGlobal projects. Try rephrasing your question or asking about a specific project, location, or community."
        mock_pipeline_components["openrouter"].generate_simple.return_value = mock_response
        
        # Mock context builder to return empty sources
        mock_context = Mock()
        mock_context.sources = []
        mock_context.context_text = ""
        mock_context.retrieval_count = 0
        mock_context.filters_applied = {}
        mock_pipeline_components["context_builder"].build_context.return_value = mock_context

        response = pipeline.ask("test query")

        assert "couldn't find any relevant information" in response.answer
        assert response.retrieval_count == 0

    def test_ask_passes_filters(self, pipeline, mock_pipeline_components):
        """Test that filters are passed to retriever."""
        mock_pipeline_components["retriever"].retrieve.return_value = []
        
        # Mock OpenRouter to return a proper response
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_pipeline_components["openrouter"].generate_simple.return_value = mock_response
        
        # Mock context builder to return empty sources
        mock_context = Mock()
        mock_context.sources = []
        mock_context.context_text = ""
        mock_context.retrieval_count = 0
        mock_context.filters_applied = {}
        mock_pipeline_components["context_builder"].build_context.return_value = mock_context

        filters = {"country": "Oman", "city": "Muscat"}
        pipeline.ask("test query", filters)

        mock_pipeline_components["retriever"].retrieve.assert_called_once()
        call_args = mock_pipeline_components["retriever"].retrieve.call_args
        called_filters = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("filters", {})
        # The query parser will convert these to lists
        assert "country" in called_filters or "countries" in called_filters

    def test_ask_no_results_response_format(self, pipeline):
        """Test format of no results response."""
        response = pipeline._no_results_response("test query", "general")

        assert "couldn't find any relevant information" in response.answer
        assert response.sources == []
        assert response.retrieved_documents == []
        assert response.retrieval_count == 0
        assert response.filters_applied == {}
        assert response.context_length == 0

    def test_get_stats(self, pipeline, mock_pipeline_components):
        """Test get_stats returns correct stats."""
        mock_pipeline_components["vector_store"].get_collection_stats.return_value = {
            "collection_name": "test",
            "document_count": 29,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384
        }
        # Also mock the embedding_model property on the vector_store
        mock_pipeline_components["vector_store"].embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        stats = pipeline.get_stats()

        assert stats["document_count"] == 29
        assert stats["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert stats["embedding_dim"] == 384
        assert stats["collection_name"] == "test"


class TestGetRAGPipeline:
    @patch('app.rag.pipeline.RAGPipeline')
    def test_singleton(self, mock_pipeline_class):
        mock_instance = Mock()
        mock_pipeline_class.return_value = mock_instance

        pipeline1 = get_rag_pipeline()
        pipeline2 = get_rag_pipeline()

        assert pipeline1 is pipeline2
        assert mock_pipeline_class.call_count >= 1


class TestRAGResponse:
    def test_rag_response_creation(self):
        response = RAGResponse(
            answer="Test answer",
            sources=[{"title": "Test"}],
            retrieved_documents=[{"title": "Doc"}],
            filters_applied={"country": "Oman"},
            model="test-model",
            retrieval_count=5,
            context_length=1000
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.retrieval_count == 5
        assert response.context_length == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])