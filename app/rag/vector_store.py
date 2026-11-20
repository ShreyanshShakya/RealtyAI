"""ChromaDB vector store for RealtyAI."""
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.rag.document_builder import RagDocument
from app.rag.embeddings import get_embedding_generator


@dataclass
class IndexManifest:
    """Manifest for index versioning and rebuild detection."""
    collection_name: str
    embedding_model: str
    embedding_dim: int
    dataset_file: str
    dataset_hash: str
    document_count: int
    build_timestamp: float
    schema_version: int = 1


class VectorStore:
    """ChromaDB vector store for RealtyAI project documents."""
    
    def __init__(self, 
                 collection_name: Optional[str] = None,
                 persist_dir: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        settings = get_settings()
        
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.persist_dir = persist_dir or settings.CHROMA_DB_PATH
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        
        self.embedding_generator = get_embedding_generator()
        self.logger = logger.bind(component="VectorStore")
        
        # Initialize ChromaDB
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Create client with persistent storage
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger.info(f"ChromaDB initialized: {self.collection_name} at {self.persist_dir}")
    
    def _compute_dataset_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of dataset file for change detection."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _load_manifest(self) -> Optional[IndexManifest]:
        """Load index manifest if exists."""
        manifest_path = Path(self.persist_dir) / f"{self.collection_name}_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                return IndexManifest(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load manifest: {e}")
        return None
    
    def _save_manifest(self, manifest: IndexManifest):
        """Save index manifest."""
        manifest_path = Path(self.persist_dir) / f"{self.collection_name}_manifest.json"
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest.__dict__, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save manifest: {e}")
    
    def needs_rebuild(self, dataset_file: str) -> bool:
        """Check if index needs rebuilding."""
        manifest = self._load_manifest()
        if manifest is None:
            return True
        
        # Check if dataset file has changed
        current_hash = self._compute_dataset_hash(dataset_file)
        if manifest.dataset_hash != current_hash:
            self.logger.info("Dataset file has changed, rebuild needed")
            return True
        
        # Check if embedding model changed
        if manifest.embedding_model != self.embedding_model:
            self.logger.info("Embedding model changed, rebuild needed")
            return True
        
        return False
    
    def _convert_canonical_to_chroma(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert canonical filters to ChromaDB where filter format.
        
        Canonical filters use lists for multiple values:
        - country: List[str] -> $or on "country" field (or $in for single field)
        - city: List[str] -> $or on "city" field
        - community: List[str] -> $or on "community" field
        - source: str -> exact match on "source" field
        - property_type: str -> exact match on "property_type" field
        
        Multiple filter groups are combined with $and.
        
        ChromaDB 0.5.5+ supports $in operator for multiple values in a single field.
        """
        if not filters:
            return None
        
        chroma_conditions = []
        
        for key, value in filters.items():
            if value is None or value == "":
                continue
            
            # Ensure value is a list for processing
            if not isinstance(value, list):
                value = [value]
            
            # Skip empty lists
            if len(value) == 0:
                continue
            
            # All canonical keys map directly to ChromaDB metadata fields
            if isinstance(value, list) and len(value) > 0:
                if len(value) == 1:
                    # Single value - exact match
                    chroma_conditions.append({key: value[0]})
                else:
                    # Multiple values - use $in operator for ChromaDB
                    chroma_conditions.append({key: {"$in": value}})
            elif value is not None and value != "":
                chroma_conditions.append({key: value})
        
        if not chroma_conditions:
            return None
        
        # Combine conditions with $and
        if len(chroma_conditions) > 1:
            return {"$and": chroma_conditions}
        
        # Single simple condition
        return chroma_conditions[0]
    
    def build_index(self, documents: List[RagDocument], rebuild: bool = False) -> int:
        """Build or update the vector index."""
        self.logger.info(f"Building index with {len(documents)} documents")
        
        if rebuild:
            self.logger.info("Rebuilding index from scratch...")
            self.reset_collection()
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.document_id)
            texts.append(doc.content)
            metadatas.append(doc.metadata)
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.encode(texts, show_progress=True)
        
        # Use upsert for incremental updates (insert or update)
        self.logger.info("Upserting documents to ChromaDB...")
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        # Remove stale documents (documents in index but not in current dataset)
        if not rebuild:
            current_ids = set(ids)
            existing_results = self.collection.get(ids=None, include=[])
            existing_ids = set(existing_results["ids"]) if existing_results["ids"] else set()
            stale_ids = existing_ids - current_ids
            if stale_ids:
                self.logger.info(f"Removing {len(stale_ids)} stale documents from index")
                self.collection.delete(ids=list(stale_ids))
        
        # Create and save manifest
        settings = get_settings()
        dataset_file = getattr(settings, 'PROCESSED_DATA_DIR', 'dataset/processed') + '/darglobal_rag_ready.jsonl'
        dataset_hash = self._compute_dataset_hash(dataset_file) if Path(dataset_file).exists() else ""
        
        manifest = IndexManifest(
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_generator.embedding_dim,
            dataset_file=dataset_file,
            dataset_hash=dataset_hash,
            document_count=len(documents),
            build_timestamp=time.time()
        )
        self._save_manifest(manifest)
        
        self.logger.success(f"Index built successfully with {len(documents)} documents")
        return len(documents)
    
    def search(self, 
               query: str, 
               k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            filters: Canonical filters (will be converted to ChromaDB format)
        """
        # Generate query embedding (1D array)
        query_embedding = self.embedding_generator.encode_single(query)
        
        # Convert canonical filters to ChromaDB format
        where_filter = self._convert_canonical_to_chroma(filters) if filters else None
        
        # Search - ChromaDB expects query_embeddings as list of lists
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        results_list = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                results_list.append({
                    "document_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "source_url": results["metadatas"][0][i].get("source_url", "")
                })
        
        return results_list
    
    def get_by_metadata(self, 
                        filters: Dict[str, Any], 
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents by metadata filters only (no semantic search).
        
        Used for listing queries where we want all matching documents.
        Supports pagination to retrieve all matching records across pages.
        
        Args:
            filters: Canonical filters
            limit: Maximum number of results (None = retrieve all matching records)
        """
        where_filter = self._convert_canonical_to_chroma(filters) if filters else None
        
        all_results = []
        batch_size = 1000  # ChromaDB max batch size
        offset = 0
        
        while True:
            current_limit = batch_size if limit is None else min(batch_size, limit - len(all_results))
            if current_limit <= 0:
                break
            
            # Use get() with where filter for metadata-only retrieval
            results = self.collection.get(
                where=where_filter,
                limit=current_limit,
                offset=offset,
                include=["documents", "metadatas"]
            )
            
            if not results["ids"]:
                break
            
            for i in range(len(results["ids"])):
                all_results.append({
                    "document_id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "similarity_score": 1.0,  # Metadata match = perfect relevance
                    "source_url": results["metadatas"][i].get("source_url", "")
                })
            
            # Check if we got fewer results than requested (last page)
            if len(results["ids"]) < current_limit:
                break
            
            offset += len(results["ids"])
            
            # If limit is specified and we've reached it, stop
            if limit is not None and len(all_results) >= limit:
                break
        
        # Preserve deterministic ordering by document_id
        all_results.sort(key=lambda x: x["document_id"])
        
        return all_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_generator.embedding_dim,
            "persist_dir": self.persist_dir
        }
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID."""
        results = self.collection.get(ids=[document_id], include=["documents", "metadatas"])
        if results["ids"]:
            return {
                "document_id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def get_all_titles(self) -> List[Dict[str, str]]:
        """Get all indexed document titles with their IDs.
        
        Returns:
            List of dicts with 'title', 'document_id', and optionally 'source_url'
        """
        results = self.collection.get(include=["metadatas"])
        titles = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                title = metadata.get("title", "")
                if title:
                    titles.append({
                        "title": title,
                        "document_id": results["ids"][i],
                        "source_url": metadata.get("source_url", "")
                    })
        return titles
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            return False
    
    def reset_collection(self):
        """Reset (delete and recreate) the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger.warning(f"Collection {self.collection_name} reset")