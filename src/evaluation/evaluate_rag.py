import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from src.rag import RAGPipeline
from src.main import ChainOfThoughtRAG
from src.utils import get_best_device, get_model
from src.splade import SPLADEEncoder


class RAGEvaluator:

    def __init__(self, relevant_doc_ids: set):
        self.relevant_doc_ids = relevant_doc_ids
        self.results = []
    
    def evaluate_single_query(self, result: Dict, query_id: str = None) -> Dict:
        """
        Evaluate a single query result.
        """
        evaluation = {
            "query_id": query_id,
            "original_query": result["original_query"],
            "metrics": {},
        }
        
        # METRIC 1: Quality Score of Final Answer
        final_quality = result.get("final_quality", 0)
        evaluation["metrics"]["final_quality_score"] = final_quality
        
        # METRIC 2: Retrieved Relevance
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            relevant_retrieved = sum(
                1 for doc in retrieved_docs 
                if doc["doc_id"] in self.relevant_doc_ids
            )
            relevance_ratio = relevant_retrieved / len(retrieved_docs)
        else:
            relevance_ratio = 0.0
        
        evaluation["metrics"]["retrieved_relevance_ratio"] = relevance_ratio
        evaluation["metrics"]["relevant_docs_retrieved"] = (
            sum(1 for doc in retrieved_docs if doc["doc_id"] in self.relevant_doc_ids)
        )
        evaluation["metrics"]["total_docs_retrieved"] = len(retrieved_docs)
        
        # METRIC 3: Answer Consistency (if alternatives were generated)
        cot = result.get("chain_of_thought", {})
        comparison = cot.get("comparison")
        
        if comparison:
            answer1_quality = comparison.get("answer1_quality", 0)
            answer2_quality = comparison.get("answer2_quality", 0)
            quality_diff = abs(answer1_quality - answer2_quality)
            evaluation["metrics"]["answer_consistency"] = quality_diff
            evaluation["metrics"]["alternative_generated"] = True
            evaluation["metrics"]["comparison_made"] = True
        else:
            evaluation["metrics"]["answer_consistency"] = 0
            evaluation["metrics"]["alternative_generated"] = False
            evaluation["metrics"]["comparison_made"] = False
        
        # METRIC 4: Retrieval Coverage
        query_variants = cot.get("query_variants", [result["original_query"]])
        evaluation["metrics"]["num_query_variants"] = len(query_variants)
        
        # METRIC 5: Reasoning depth (number of CoT steps performed)
        cot_steps = sum(1 for k in cot.keys() if k != "query_variants")
        evaluation["metrics"]["cot_steps_performed"] = cot_steps
        
        self.results.append(evaluation)
        return evaluation
    
    def evaluate_batch(self, results_list: List[Dict], 
                      query_ids: List[str] = None) -> Dict:
        """
        Evaluate multiple query results.
        """
        if query_ids is None:
            query_ids = [None] * len(results_list)
        
        for result, qid in zip(results_list, query_ids):
            self.evaluate_single_query(result, qid)
        
        return self.get_summary_stats()
    
    def get_summary_stats(self) -> Dict:
        """
        Compute summary statistics across all evaluated queries.
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        quality_scores = [r["metrics"]["final_quality_score"] for r in self.results]
        relevance_ratios = [r["metrics"]["retrieved_relevance_ratio"] for r in self.results]
        consistency_scores = [r["metrics"]["answer_consistency"] for r in self.results]
        cot_steps = [r["metrics"]["cot_steps_performed"] for r in self.results]
        
        alternatives_generated = sum(
            1 for r in self.results if r["metrics"]["alternative_generated"]
        )
        
        summary = {
            "total_queries_evaluated": len(self.results),
            "quality_score": {
                "mean": statistics.mean(quality_scores),
                "median": statistics.median(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores),
                "stdev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            },
            "retrieval_relevance": {
                "mean_ratio": statistics.mean(relevance_ratios),
                "median_ratio": statistics.median(relevance_ratios),
                "min_ratio": min(relevance_ratios),
                "max_ratio": max(relevance_ratios),
            },
            "answer_consistency": {
                "mean_diff": statistics.mean(consistency_scores),
                "median_diff": statistics.median(consistency_scores),
                "max_diff": max(consistency_scores),
            },
            "pipeline_efficiency": {
                "mean_cot_steps": statistics.mean(cot_steps),
                "alternatives_generated": alternatives_generated,
                "alternatives_percentage": (alternatives_generated / len(self.results)) * 100,
            },
            "detailed_results": self.results,
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Pretty print summary statistics."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nTotal Queries Evaluated: {summary['total_queries_evaluated']}")
        
        print("\n--- QUALITY SCORES ---")
        qs = summary["quality_score"]
        print(f"  Mean: {qs['mean']:.2f}/5")
        print(f"  Median: {qs['median']:.2f}/5")
        print(f"  Range: {qs['min']:.0f}-{qs['max']:.0f}")
        print(f"  Std Dev: {qs['stdev']:.2f}")
        
        print("\n--- RETRIEVAL RELEVANCE ---")
        rr = summary["retrieval_relevance"]
        print(f"  Mean Ratio: {rr['mean_ratio']:.2%}")
        print(f"  Median Ratio: {rr['median_ratio']:.2%}")
        print(f"  Range: {rr['min_ratio']:.2%}-{rr['max_ratio']:.2%}")
        
        print("\n--- ANSWER CONSISTENCY ---")
        ac = summary["answer_consistency"]
        print(f"  Mean Quality Diff (initial vs alt): {ac['mean_diff']:.2f}")
        print(f"  Max Diff: {ac['max_diff']:.0f}")
        
        print("\n--- PIPELINE EFFICIENCY ---")
        pe = summary["pipeline_efficiency"]
        print(f"  Mean CoT Steps: {pe['mean_cot_steps']:.1f}")
        print(f"  Alternatives Generated: {pe['alternatives_generated']}/{summary['total_queries_evaluated']} ({pe['alternatives_percentage']:.1f}%)")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath: str):
        """Save detailed results to JSON file."""
        summary = self.get_summary_stats()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {filepath}")


def evaluate_on_test_set(rag: RAGPipeline, cot_rag, 
                         queries_df, qrels_df, 
                         num_queries: int = 5,
                         use_rewrites: bool = True,
                         use_alternatives: bool = True):
    """
    Evaluate pipeline on a subset of test queries.
    """
    from src.main import ChainOfThoughtRAG  # Import here to avoid circular deps
    
    # Get relevant doc IDs
    relevant_doc_ids = set(qrels_df["docno"].tolist())
    
    evaluator = RAGEvaluator(relevant_doc_ids)
    
    query_ids = queries_df["qid"].tolist()[:num_queries]
    test_queries = queries_df[queries_df["qid"].isin(query_ids)]
    
    print(f"\nEvaluating {len(test_queries)} queries...")
    print("="*80)
    
    results = []
    for idx, row in test_queries.iterrows():
        qid = row["qid"]
        query_text = row.get("query") or row.get("title") or row.get("text")
        
        print(f"\n[{idx+1}/{len(test_queries)}] Query {qid}: {query_text}")
        
        try:
            result = cot_rag.process_query(
                query_text,
                use_rewrites=use_rewrites,
                use_alternatives=use_alternatives,
                verbose=False 
            )
            results.append(result)
            
            eval_result = evaluator.evaluate_single_query(result, qid)
            print(f"  Quality: {eval_result['metrics']['final_quality_score']}/5 | "
                  f"Relevance: {eval_result['metrics']['retrieved_relevance_ratio']:.0%}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    summary = evaluator.get_summary_stats()
    evaluator.print_summary(summary)
    
    return evaluator, results


if __name__ == "__main__":
    import pyterrier as pt
    
    print("Loading dataset and models...")
    
    dataset = pt.get_dataset("irds:lotte/technology/dev/search")
    index_path = Path("./outputs/practical-04/index_lotte").absolute()
    index_ref = str(index_path)
    index = pt.IndexFactory.of(index_ref, memory={"meta": True})
    meta_index = index.getMetaIndex()
    
    device = get_best_device()
    model, tokenizer = get_model(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device
    )
    
    qrels_df = dataset.get_qrels()
    queries_df = dataset.get_topics()
    query_ids = set(queries_df["qid"].tolist())
    qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()
    
    relevant_doc_ids = set(qrels_df["docno"].tolist())
    doc_id_list = list(relevant_doc_ids)
    
    splade_encoder = SPLADEEncoder("naver/splade-cocondenser-ensembledistil")
    splade_encoder = splade_encoder.to(device)
    
    rag = RAGPipeline(
        retriever=splade_encoder,
        device=device,
        generator=model,
        generator_tokenizer=tokenizer,
        pt_index=meta_index,
        doc_ids=doc_id_list,
        top_k=3,
    )
    
    cot_rag = ChainOfThoughtRAG(rag, model, tokenizer, device)
    
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    evaluator, results = evaluate_on_test_set(
        rag, cot_rag,
        queries_df, qrels_df,
        num_queries=50,
        use_rewrites=True,
        use_alternatives=True
    )
    
    # Save results
    output_path = Path("./evaluation_results.json")
    evaluator.save_results(str(output_path))
    print(f"\nDetailed results saved to {output_path}")