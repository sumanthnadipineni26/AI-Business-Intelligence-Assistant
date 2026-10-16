"""
Benchmark runner script for isolated RAG evaluation.
Runs all 6 experimental conditions under identical conditions.
Exports results to benchmark_results.json.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List

# Ensure tests root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.evaluation.document_indexer import DatasetCorpus
from tests.evaluation.rag_baseline import ProductionRAGBaselineAdapter
from tests.evaluation.retrieval_adapters import (
    BM25RetrieverAdapter,
    DenseFAISSRetrieverAdapter,
    HybridRRFAdapter,
    CrossEncoderRerankAdapter,
    BGEDenseAdapter,
    GroundedPipelineRunner
)
from tests.evaluation.metrics_calculator import BenchmarkMetricsAggregator


def load_benchmark_suite() -> List[Dict[str, Any]]:
    suite_path = os.path.join(os.path.dirname(__file__), "benchmark_suite.json")
    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_all_benchmarks():
    print("=" * 80)
    print("STARTING ISOLATED RAG ACCURACY BENCHMARK EVALUATION")
    print("=" * 80)

    corpus = DatasetCorpus()
    suite = load_benchmark_suite()
    print(f"Loaded {len(suite)} verified benchmark queries across 3 datasets.")

    results_by_experiment: Dict[str, Dict[str, Any]] = {}
    detailed_outputs: Dict[str, List[Dict[str, Any]]] = {}

    # ---------------------------------------------------------
    # 1. Baseline Experiment (Production Rule-Based Matcher)
    # ---------------------------------------------------------
    print("\n--- Running Condition 1: Production Rule-Based Baseline ---")
    baseline = ProductionRAGBaselineAdapter(corpus)
    baseline_outputs = []
    for q in suite:
        res = baseline.query(q["query"], q["dataset_key"])
        res["query_id"] = q["id"]
        baseline_outputs.append(res)
    
    baseline_metrics = BenchmarkMetricsAggregator.aggregate(baseline_outputs, suite, k=5)
    results_by_experiment["1_Baseline_Rule_Matcher"] = {
        "status": "COMPLETED",
        "description": "Production regex/rule-based matcher with static aggregations",
        "metrics": baseline_metrics
    }
    detailed_outputs["1_Baseline_Rule_Matcher"] = baseline_outputs
    print(f"Baseline Completed. Numerical Correctness: {baseline_metrics['numerical_correctness']*100:.1f}%, Abstention: {baseline_metrics['abstention_accuracy']*100:.1f}%")

    # ---------------------------------------------------------
    # 2. Condition 2: Pure-Python BM25 Lexical Search
    # ---------------------------------------------------------
    print("\n--- Running Condition 2: Pure-Python BM25 Retrieval ---")
    bm25_adapter = BM25RetrieverAdapter(corpus)
    runner_bm25 = GroundedPipelineRunner(bm25_adapter, corpus)
    bm25_outputs = []
    for q in suite:
        res = runner_bm25.answer_query(q, top_k=5)
        bm25_outputs.append(res)
    
    bm25_metrics = BenchmarkMetricsAggregator.aggregate(bm25_outputs, suite, k=5)
    results_by_experiment["2_Lexical_BM25"] = {
        "status": "COMPLETED",
        "description": "Pure-Python BM25Okapi lexical retrieval (k1=1.5, b=0.75)",
        "metrics": bm25_metrics
    }
    detailed_outputs["2_Lexical_BM25"] = bm25_outputs
    print(f"BM25 Completed. P@5: {bm25_metrics['precision_at_5']:.4f}, Recall@5: {bm25_metrics['recall_at_5']:.4f}, MRR: {bm25_metrics['mrr']:.4f}")

    # ---------------------------------------------------------
    # 3. Condition 3: Dense Semantic Search (all-MiniLM-L6-v2 + FAISS)
    # ---------------------------------------------------------
    print("\n--- Running Condition 3: Dense Semantic Retrieval (all-MiniLM-L6-v2 + FAISS) ---")
    dense_adapter = DenseFAISSRetrieverAdapter(corpus, model_name="all-MiniLM-L6-v2")
    if dense_adapter.status == "READY":
        runner_dense = GroundedPipelineRunner(dense_adapter, corpus)
        dense_outputs = []
        for q in suite:
            res = runner_dense.answer_query(q, top_k=5)
            dense_outputs.append(res)
        
        dense_metrics = BenchmarkMetricsAggregator.aggregate(dense_outputs, suite, k=5)
        results_by_experiment["3_Dense_MiniLM_FAISS"] = {
            "status": "COMPLETED",
            "description": "Sentence-BERT all-MiniLM-L6-v2 (384-d) with FAISS IndexFlatIP",
            "metrics": dense_metrics
        }
        detailed_outputs["3_Dense_MiniLM_FAISS"] = dense_outputs
        print(f"Dense Completed. P@5: {dense_metrics['precision_at_5']:.4f}, Recall@5: {dense_metrics['recall_at_5']:.4f}, MRR: {dense_metrics['mrr']:.4f}")
    else:
        results_by_experiment["3_Dense_MiniLM_FAISS"] = {
            "status": "BLOCKED",
            "description": "Sentence-BERT all-MiniLM-L6-v2 + FAISS",
            "error": dense_adapter.error_message
        }
        print(f"Dense BLOCKED: {dense_adapter.error_message}")

    # ---------------------------------------------------------
    # 4. Condition 4: Hybrid RRF (Dense + BM25)
    # ---------------------------------------------------------
    print("\n--- Running Condition 4: Hybrid RRF Retrieval (Dense + BM25) ---")
    if dense_adapter.status == "READY":
        hybrid_adapter = HybridRRFAdapter(dense_adapter, bm25_adapter, rrf_k=60)
        runner_hybrid = GroundedPipelineRunner(hybrid_adapter, corpus)
        hybrid_outputs = []
        for q in suite:
            res = runner_hybrid.answer_query(q, top_k=5)
            hybrid_outputs.append(res)

        hybrid_metrics = BenchmarkMetricsAggregator.aggregate(hybrid_outputs, suite, k=5)
        results_by_experiment["4_Hybrid_RRF"] = {
            "status": "COMPLETED",
            "description": "Reciprocal Rank Fusion (k=60) combining all-MiniLM-L6-v2 and BM25",
            "metrics": hybrid_metrics
        }
        detailed_outputs["4_Hybrid_RRF"] = hybrid_outputs
        print(f"Hybrid RRF Completed. P@5: {hybrid_metrics['precision_at_5']:.4f}, Recall@5: {hybrid_metrics['recall_at_5']:.4f}, MRR: {hybrid_metrics['mrr']:.4f}")
    else:
        results_by_experiment["4_Hybrid_RRF"] = {
            "status": "BLOCKED",
            "description": "Hybrid RRF (Dense + BM25)",
            "error": "Dense retriever was not available"
        }

    # ---------------------------------------------------------
    # 5. Condition 5: Cross-Encoder Reranking
    # ---------------------------------------------------------
    print("\n--- Running Condition 5: Hybrid RRF + Cross-Encoder Reranker ---")
    if dense_adapter.status == "READY":
        cross_adapter = CrossEncoderRerankAdapter(hybrid_adapter, corpus, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        if cross_adapter.status == "READY":
            runner_cross = GroundedPipelineRunner(cross_adapter, corpus)
            cross_outputs = []
            for q in suite:
                res = runner_cross.answer_query(q, top_k=5)
                cross_outputs.append(res)

            cross_metrics = BenchmarkMetricsAggregator.aggregate(cross_outputs, suite, k=5)
            results_by_experiment["5_CrossEncoder_Rerank"] = {
                "status": "COMPLETED",
                "description": "Hybrid RRF top-15 reranked by cross-encoder/ms-marco-MiniLM-L-6-v2",
                "metrics": cross_metrics
            }
            detailed_outputs["5_CrossEncoder_Rerank"] = cross_outputs
            print(f"Cross-Encoder Completed. P@5: {cross_metrics['precision_at_5']:.4f}, Recall@5: {cross_metrics['recall_at_5']:.4f}, MRR: {cross_metrics['mrr']:.4f}")
        else:
            results_by_experiment["5_CrossEncoder_Rerank"] = {
                "status": "BLOCKED",
                "description": "Cross-Encoder Reranker",
                "error": cross_adapter.error_message
            }
            print(f"Cross-Encoder BLOCKED: {cross_adapter.error_message}")
    else:
        results_by_experiment["5_CrossEncoder_Rerank"] = {
            "status": "BLOCKED",
            "description": "Cross-Encoder Reranker",
            "error": "Dense retriever was not available"
        }

    # ---------------------------------------------------------
    # 6. Condition 6: BGE Dense Embeddings (BAAI/bge-small-en-v1.5)
    # ---------------------------------------------------------
    print("\n--- Running Condition 6: BGE Embeddings (BAAI/bge-small-en-v1.5 + FAISS) ---")
    bge_adapter = BGEDenseAdapter(corpus, model_name="BAAI/bge-small-en-v1.5")
    if bge_adapter.status == "READY":
        runner_bge = GroundedPipelineRunner(bge_adapter, corpus)
        bge_outputs = []
        for q in suite:
            res = runner_bge.answer_query(q, top_k=5)
            bge_outputs.append(res)

        bge_metrics = BenchmarkMetricsAggregator.aggregate(bge_outputs, suite, k=5)
        results_by_experiment["6_BGE_Dense_FAISS"] = {
            "status": "COMPLETED",
            "description": "BAAI/bge-small-en-v1.5 (384-d) with FAISS IndexFlatIP",
            "metrics": bge_metrics
        }
        detailed_outputs["6_BGE_Dense_FAISS"] = bge_outputs
        print(f"BGE Completed. P@5: {bge_metrics['precision_at_5']:.4f}, Recall@5: {bge_metrics['recall_at_5']:.4f}, MRR: {bge_metrics['mrr']:.4f}")
    else:
        results_by_experiment["6_BGE_Dense_FAISS"] = {
            "status": "BLOCKED",
            "description": "BGE Dense Embeddings + FAISS",
            "error": bge_adapter.error_message
        }
        print(f"BGE BLOCKED: {bge_adapter.error_message}")

    # ---------------------------------------------------------
    # Output and Save Summary
    # ---------------------------------------------------------
    out_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(suite),
            "summary_metrics": results_by_experiment,
            "detailed_outputs": detailed_outputs
        }, f, indent=2)

    print("\n" + "=" * 105)
    print("BENCHMARK RESULTS SUMMARY TABLE (CORRECTED EVALUATION)")
    print("=" * 105)
    header = f"{'Experiment':<25} | {'Status':<9} | {'P@5':<6} | {'BoundP':<6} | {'R-Prec':<6} | {'R@5':<6} | {'MRR':<6} | {'EvidGrd':<7} | {'NumAcc':<7} | {'Abstain':<7} | {'Latency':<8}"
    print(header)
    print("-" * len(header))
    for exp_name, data in results_by_experiment.items():
        if data["status"] == "COMPLETED":
            m = data["metrics"]
            p5 = f"{m['precision_at_5']:.4f}"
            bp5 = f"{m.get('bounded_precision_at_5', 0.0):.4f}"
            rprec = f"{m.get('r_precision', 0.0):.4f}"
            r5 = f"{m['recall_at_5']:.4f}"
            mrr = f"{m['mrr']:.4f}"
            egrd = f"{m.get('evidence_grounding', 0.0)*100:.1f}%"
            num = f"{m['numerical_correctness']*100:.1f}%"
            abs_acc = f"{m['abstention_accuracy']*100:.1f}%"
            lat = f"{m['avg_latency_ms']:.1f}ms"
            print(f"{exp_name:<25} | {data['status']:<9} | {p5:<6} | {bp5:<6} | {rprec:<6} | {r5:<6} | {mrr:<6} | {egrd:<7} | {num:<7} | {abs_acc:<7} | {lat:<8}")
        else:
            print(f"{exp_name:<25} | {data['status']:<9} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<7} | {'N/A':<7} | {'N/A':<7} | {'N/A':<8}")

    print("\nBenchmark results saved to:", out_file)
    return results_by_experiment


if __name__ == "__main__":
    run_all_benchmarks()
