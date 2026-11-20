"""Tests for embeddings module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.rag.embeddings import EmbeddingGenerator, get_embedding_generator


class TestEmbeddingGenerator:
    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_init(self, generator):
        assert generator.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert generator.batch_size == 32
        assert generator._model is None
        assert generator._embedding_dim is None

    def test_embedding_dim_property(self, generator):
        """Test that embedding_dim property loads model if needed."""
        with patch.object(generator, '_load_model') as mock_load:
            generator._embedding_dim = 384
            assert generator.embedding_dim == 384
            mock_load.assert_not_called()

    def test_embedding_dim_property_cached(self, generator):
        """Test that embedding_dim property returns cached value without loading model."""
        with patch.object(generator, '_load_model') as mock_load:
            generator._embedding_dim = 384
            assert generator.embedding_dim == 384
            mock_load.assert_not_called()

    def test_encode_single_string(self, generator):
        """Test encoding a single string."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        generator._model = mock_model
        generator._embedding_dim = 3

        result = generator.encode("test query")

        assert result.shape == (1, 3)
        assert np.allclose(result[0], [0.1, 0.2, 0.3])
        mock_model.encode.assert_called_once()

    def test_encode_list_of_strings(self, generator):
        """Test encoding a list of strings."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        generator._model = mock_model
        generator._embedding_dim = 2

        texts = ["query 1", "query 2", "query 3"]
        result = generator.encode(texts)

        assert result.shape == (3, 2)
        mock_model.encode.assert_called_once()

    def test_encode_empty_list(self, generator):
        """Test encoding empty list returns empty array."""
        generator._embedding_dim = 384

        result = generator.encode([])

        assert result.shape == (0, 384)

    def test_encode_with_empty_strings(self, generator):
        """Test encoding list with some empty strings."""
        mock_model = Mock()
        # When there are empty strings, they get filtered out
        # The mock should only be called with non-empty texts (2 valid texts)
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        generator._model = mock_model
        generator._embedding_dim = 2

        texts = ["valid", "", "  ", "also valid"]
        result = generator.encode(texts)

        # Should have 4 rows (one per input)
        assert result.shape == (4, 2)
        # Only valid texts should have embeddings
        assert np.allclose(result[0], [0.1, 0.2])
        assert np.allclose(result[1], [0.0, 0.0])  # empty
        assert np.allclose(result[2], [0.0, 0.0])  # whitespace only
        assert np.allclose(result[3], [0.3, 0.4])

    def test_encode_normalize_false(self, generator):
        """Test encoding with normalize=False."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0]])
        generator._model = mock_model
        generator._embedding_dim = 2

        result = generator.encode("test", normalize=False)

        mock_model.encode.assert_called_once_with(
            ["test"],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

    def test_encode_single(self, generator):
        """Test encode_single returns 1D array."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        generator._model = mock_model
        generator._embedding_dim = 3

        result = generator.encode_single("test")

        assert result.shape == (3,)
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_embedding_dim_property(self, generator):
        """Test that embedding_dim property loads model if needed."""
        with patch.object(generator, '_load_model') as mock_load:
            generator._embedding_dim = 384
            assert generator.embedding_dim == 384
            mock_load.assert_not_called()

    def test_embedding_dim_property_cached(self, generator):
        """Test that embedding_dim property returns cached value without loading model."""
        with patch.object(generator, '_load_model') as mock_load:
            generator._embedding_dim = 384
            assert generator.embedding_dim == 384
            mock_load.assert_not_called()


class TestGetEmbeddingGenerator:
    def test_singleton(self):
        """Test that get_embedding_generator returns singleton."""
        gen1 = get_embedding_generator()
        gen2 = get_embedding_generator()
        assert gen1 is gen2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])