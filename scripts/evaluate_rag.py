#!/usr/bin/env python
"""Evaluate RAG pipeline against evaluation queries."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.rag.pipeline import RAGPipeline, get_rag_pipeline
from app.config import get_settings


def load_evaluation_queries(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation queries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("evaluation_queries", [])


def evaluate_query(pipeline: RAGPipeline, query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single query."""
    query = query_data["query"]
    expected_projects = query_data.get("expected_projects", [])
    should_refuse = query_data.get("should_refuse", False)
    
    # Get RAG response
    response = pipeline.ask(query)
    
    # Check if response refuses (for missing info queries)
    answer = response.answer.lower()
    refused = any(phrase in answer for phrase in [
        "not available", "not in the dataset", "does not contain", 
        "cannot", "unable to", "not found", "not verified",
        "outside the scope", "not in the collected"
    ])
    
    # Check if expected projects were retrieved
    retrieved_titles = [doc.get("title", "") for doc in response.retrieved_documents]
    retrieved_matches = sum(1 for exp in expected_projects if any(exp.lower() in rt.lower() for rt in retrieved_titles))
    retrieval_recall = retrieved_matches / len(expected_projects) if expected_projects else 1.0
    
    # Check if answer cites sources
    has_citations = "source" in response.answer.lower() or "[" in response.answer
    
    # Evaluate
    result = {
        "query_id": query_data["id"],
        "query": query,
        "type": query_data["type"],
        "should_refuse": should_refuse,
        "refused": refused,
        "refusal_correct": refused == should_refuse,
        "retrieval_recall": round(retrieval_recall, 2),
        "retrieved_count": response.retrieval_count,
        "has_citations": has_citations,
        "answer_length": len(response.answer),
        "model": response.model,
        "retrieved_titles": retrieved_titles,
        "expected_titles": expected_projects,
        "answer_preview": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer
    }
    
    return result


def run_evaluation(pipeline: RAGPipeline, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run evaluation on all queries."""
    results = []
    
    for query_data in queries:
        logger.info(f"Evaluating query {query_data['id']}: {query_data['query'][:50]}...")
        result = evaluate_query(pipeline, query_data)
        results.append(result)
        
        # Log summary
        status = "✓" if result["refusal_correct"] else "✗"
        logger.info(f"  {status} Refusal correct: {result['refusal_correct']}, Recall: {result['retrieval_recall']:.2f}")
    
    # Calculate aggregate metrics
    total = len(results)
    refusal_correct = sum(1 for r in results if r["refusal_correct"])
    avg_recall = sum(r["retrieval_recall"] for r in results) / total if total > 0 else 0
    citation_rate = sum(1 for r in results if r["has_citations"]) / total if total > 0 else 0
    
    summary = {
        "total_queries": total,
        "refusal_accuracy": round(refusal_correct / total, 2) if total > 0 else 0,
        "avg_retrieval_recall": round(avg_recall, 2),
        "citation_rate": round(citation_rate, 2),
        "avg_retrieved_count": round(sum(r["retrieved_count"] for r in results) / total, 1) if total > 0 else 0
    }
    
    return {
        "summary": summary,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--queries", default="tests/evaluation/rag_queries.json", help="Evaluation queries file")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index before evaluation")
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    logger.info("Starting RAG evaluation")
    
    # Initialize pipeline
    pipeline = get_rag_pipeline()
    pipeline.initialize(rebuild_index=args.rebuild)
    
    # Load queries
    queries = load_evaluation_queries(args.queries)
    logger.info(f"Loaded {len(queries)} evaluation queries")
    
    # Run evaluation
    evaluation_results = run_evaluation(pipeline, queries)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries:     {evaluation_results['summary']['total_queries']}")
    print(f"Refusal accuracy:  {evaluation_results['summary']['refusal_accuracy']:.2%}")
    print(f"Avg retrieval recall: {evaluation_results['summary']['avg_retrieval_recall']:.2%}")
    print(f"Citation rate:     {evaluation_results['summary']['citation_rate']:.2%}")
    print(f"Avg retrieved:     {evaluation_results['summary']['avg_retrieved_count']}")
    print("="*60)
    
    # Save detailed results
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Evaluation results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())