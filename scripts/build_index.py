#!/usr/bin/env python
"""Build or rebuild the ChromaDB index from RAG-ready dataset."""
import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.rag.document_builder import DocumentBuilder, load_rag_ready_records
from app.rag.vector_store import VectorStore


def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add("logs/build_index_{time}.log", rotation="10 MB", level="DEBUG")


def main():
    parser = argparse.ArgumentParser(description="Build or rebuild ChromaDB index for RealtyAI")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--source", choices=["darglobal", "wasalt", "both"], default="darglobal", help="Data source to index")
    parser.add_argument("--output", default="dataset/chroma_db", help="ChromaDB persist directory")
    parser.add_argument("--collection", default="realtyai_projects", help="Collection name")
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("Starting index build process")
    logger.info(f"Source: {args.source}, Rebuild: {args.rebuild}")
    
    settings = get_settings()
    
    # Load RAG-ready records
    if args.source in ["darglobal", "both"]:
        rag_ready_file = Path(settings.PROCESSED_DATA_DIR) / "darglobal_rag_ready.jsonl"
        if not rag_ready_file.exists():
            logger.error(f"RAG-ready file not found: {rag_ready_file}")
            logger.info("Run validation first: python scripts/validate.py --source darglobal")
            return 1
        
        logger.info(f"Loading records from {rag_ready_file}")
        records = load_rag_ready_records(str(rag_ready_file))
        logger.info(f"Loaded {len(records)} RAG-ready records")
    else:
        logger.warning("Wasalt data not available yet")
        return 0
    
    if not records:
        logger.error("No records to index")
        return 1
    
    # Build documents
    logger.info("Building documents...")
    doc_builder = DocumentBuilder()
    documents = doc_builder.build_documents(records)
    logger.info(f"Built {len(documents)} documents")
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name=args.collection,
        persist_dir=args.output
    )
    
    # Check if rebuild needed
    if not args.rebuild and not vector_store.needs_rebuild(str(rag_ready_file)):
        logger.info("Index is up to date. Use --rebuild to force rebuild.")
        stats = vector_store.get_collection_stats()
        logger.info(f"Existing index: {stats['document_count']} documents")
        return 0
    
    # Build index
    logger.info("Building index...")
    indexed_count = vector_store.build_index(documents, rebuild=args.rebuild)
    
    # Print statistics
    stats = vector_store.get_collection_stats()
    logger.success("Index build complete!")
    logger.info(f"Collection: {stats['collection_name']}")
    logger.info(f"Documents indexed: {stats['document_count']}")
    logger.info(f"Embedding model: {stats['embedding_model']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Persist directory: {stats['persist_dir']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())