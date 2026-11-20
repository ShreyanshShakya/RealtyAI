#!/usr/bin/env python
"""Production integration tests using the actual 29-record dataset."""
import sys
sys.path.insert(0, '.')
from app.rag.pipeline import RAGPipeline

def main():
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    results = []
    
    # Test 1: Project comparison
    print("Test 1: Compare The Astera and Trump Tower Jeddah")
    response = pipeline.ask("Compare The Astera and Trump Tower Jeddah")
    titles = [d.get('metadata', {}).get('title', '') for d in response.retrieved_documents]
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    print(f"  Titles: {titles}")
    test1_pass = "The Astera, Interiors by Aston Martin" in titles and "Trump Tower Jeddah" in titles
    print(f"  PASS: {test1_pass}")
    results.append({"test": "project_comparison", "pass": test1_pass, "titles": titles})
    print()
    
    # Test 2: Sidebar filter propagation
    print("Test 2: Sidebar filter - Country: Saudi Arabia, City: Riyadh")
    response = pipeline.ask("Show projects", filters={"country": ["Saudi Arabia"], "city": ["Riyadh"]})
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    all_match = True
    for d in response.retrieved_documents:
        meta = d.get('metadata', {})
        city = meta.get("city")
        country = meta.get("country")
        match = (city == "Riyadh" and country == "Saudi Arabia")
        all_match = all_match and match
        print(f"  - {meta.get('title')}: {city}, {country} {'OK' if match else 'MISMATCH'}")
    print(f"  PASS: {all_match}")
    results.append({"test": "sidebar_filter_propagation", "pass": all_match})
    print()
    
    # Test 3: Complete Oman listing
    print("Test 3: Show all DarGlobal projects in Oman")
    response = pipeline.ask("Show all DarGlobal projects in Oman")
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    oman_count = 0
    for d in response.retrieved_documents:
        meta = d.get('metadata', {})
        if meta.get("country") == "Oman":
            oman_count += 1
        print(f"  - {meta.get('title')}: {meta.get('city')}, {meta.get('country')}")
    # Should return all Oman projects (not just first 20)
    test3_pass = oman_count > 0 and response.retrieval_count == oman_count
    print(f"  PASS: {test3_pass} (all {oman_count} Oman projects returned)")
    results.append({"test": "complete_oman_listing", "pass": test3_pass, "count": oman_count})
    print()
    
    # Test 4: Missing price
    print("Test 4: What is the price of The Astera?")
    response = pipeline.ask("What is the price of The Astera?")
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    has_unavailable = "not available" in response.answer.lower() or "not in the dataset" in response.answer.lower() or "unavailable" in response.answer.lower()
    print(f"  Answer: {response.answer[:300]}...")
    print(f"  PASS: {has_unavailable}")
    results.append({"test": "missing_price", "pass": has_unavailable})
    print()
    
    # Test 5: Wasalt
    print("Test 5: Show me Wasalt listings in Riyadh")
    response = pipeline.ask("Show me Wasalt listings in Riyadh")
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    has_wasalt_msg = "wasalt" in response.answer.lower() and ("not available" in response.answer.lower() or "not yet been acquired" in response.answer.lower())
    print(f"  Answer: {response.answer}")
    print(f"  PASS: {has_wasalt_msg}")
    results.append({"test": "wasalt_unavailable", "pass": has_wasalt_msg})
    print()
    
    # Test 6: Unsupported query
    print("Test 6: Give me the stock-market forecast for next year")
    response = pipeline.ask("Give me the stock-market forecast for next year")
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    has_refusal = "stock" not in response.answer.lower() or "not available" in response.answer.lower() or "dataset" in response.answer.lower()
    print(f"  Answer: {response.answer}")
    print(f"  PASS: {has_refusal}")
    results.append({"test": "unsupported_query", "pass": has_refusal})
    print()
    
    # Test 7: Source attribution
    print("Test 7: Tell me about The Astera")
    response = pipeline.ask("Tell me about The Astera")
    print(f"  Query type: {response.query_type}")
    print(f"  Retrieved: {response.retrieval_count} docs")
    has_source = False
    for s in response.sources:
        print(f"  Source: {s['title']} - {s['url']}")
        if "astera" in s['title'].lower():
            has_source = True
    print(f"  PASS: {has_source}")
    results.append({"test": "source_attribution", "pass": has_source})
    print()
    
    # Summary
    print("="*60)
    print("PRODUCTION INTEGRATION TEST SUMMARY")
    print("="*60)
    all_pass = all(r["pass"] for r in results)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {status}: {r['test']}")
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    # Save results
    import json
    from pathlib import Path
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "phase2_4_production_integration.json", "w") as f:
        json.dump({
            "overall_pass": all_pass,
            "tests": results
        }, f, indent=2)
    print(f"\nResults saved to reports/phase2_4_production_integration.json")

if __name__ == "__main__":
    main()