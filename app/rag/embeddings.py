"""Local embedding generation using Sentence Transformers."""
import logging
from typing import List, Optional, Union
from functools import lru_cache
import numpy as np
from loguru import logger

from app.config import get_settings


class EmbeddingGenerator:
    """Generates embeddings using local Sentence Transformers model."""
    
    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._embedding_dim = None
        self.logger = logger.bind(component="EmbeddingGenerator")
    
    def _load_model(self):
        """Lazy load the Sentence Transformer model."""
        if self._model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get embedding dimension
                test_embedding = self._model.encode(["test"])
                self._embedding_dim = len(test_embedding[0])
                self.logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts to embed
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        self._load_model()
        
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        # Filter empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                self.logger.warning(f"Empty text at index {i}, skipping")
        
        if not valid_texts:
            return np.zeros((len(texts), self.embedding_dim))
        
        try:
            # Generate embeddings in batches
            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Ensure embeddings is 2D (n_texts, embedding_dim)
            embeddings = np.asarray(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            elif embeddings.ndim > 2:
                embeddings = embeddings.reshape(len(valid_texts), -1)
            
            # Create full result array with zeros for empty texts
            result = np.zeros((len(texts), self.embedding_dim))
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = embedding
            
            return result
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text. Returns 1D array."""
        result = self.encode(text, normalize=normalize)
        return result[0] if result.shape[0] > 0 else np.zeros(self.embedding_dim)


@lru_cache(maxsize=1)
def get_embedding_generator() -> EmbeddingGenerator:
    """Get cached embedding generator instance."""
    return EmbeddingGenerator()