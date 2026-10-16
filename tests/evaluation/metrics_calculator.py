"""
Evaluation metrics calculator for isolated RAG benchmark.
Computes:
- Precision@K (Standard)
- Bounded Precision@K (Normalized by min(K, |Relevant|))
- R-Precision (Precision at rank R = |Relevant|)
- Recall@K
- Mean Reciprocal Rank (MRR)
- Evidence Grounding Rate (checks whether retrieved document chunks contain required ground-truth evidence)
- Numerical Correctness Rate (analytical query accuracy using exact dataset calculations)
- Abstention Accuracy Rate (unsupported query refusal accuracy)
- Average Latency (ms)
"""
import re
from typing import List, Dict, Any, Optional


def compute_retrieval_metrics(retrieved_indices: List[int], relevant_indices: List[int], k: int = 5) -> Dict[str, float]:
    """
    Computes standard and normalized retrieval metrics:
    - Precision@K: hits / K
    - Bounded Precision@K: hits / min(K, |Relevant|)
    - R-Precision: hits in top-R / R, where R = |Relevant|
    - Recall@K: hits / |Relevant|
    - MRR: reciprocal rank of first relevant retrieved document
    """
    if not relevant_indices:
        return {
            "precision_at_k": 0.0,
            "bounded_precision_at_k": 0.0,
            "r_precision": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0
        }

    top_k = retrieved_indices[:k]
    relevant_set = set(relevant_indices)
    retrieved_set = set(top_k)

    intersection = len(retrieved_set.intersection(relevant_set))
    precision = intersection / k if k > 0 else 0.0
    
    # Bounded Precision@K: normalizes by maximum possible hits achievable in top-K
    denom_bounded = min(k, len(relevant_set))
    bounded_precision = (intersection / denom_bounded) if denom_bounded > 0 else 0.0

    # R-Precision: precision at rank R, where R is total number of relevant documents
    r_val = len(relevant_set)
    top_r = retrieved_indices[:r_val]
    r_hits = len(set(top_r).intersection(relevant_set))
    r_precision = r_hits / r_val if r_val > 0 else 0.0

    recall = intersection / len(relevant_set) if len(relevant_set) > 0 else 0.0

    # Reciprocal Rank of first relevant item
    rr = 0.0
    for rank_idx, doc_idx in enumerate(top_k):
        if doc_idx in relevant_set:
            rr = 1.0 / (rank_idx + 1)
            break

    return {
        "precision_at_k": precision,
        "bounded_precision_at_k": bounded_precision,
        "r_precision": r_precision,
        "recall_at_k": recall,
        "mrr": rr
    }


def extract_numbers_from_text(text: str) -> List[float]:
    """Extracts numeric values from string, stripping $, commas, and percentages."""
    cleaned = re.sub(r"[$,]", "", text)
    matches = re.findall(r"[-+]?\d*\.?\d+", cleaned)
    nums = []
    for m in matches:
        try:
            val = float(m)
            nums.append(val)
        except ValueError:
            pass
    return nums


def evaluate_numerical_correctness(answer_text: str, expected_val: Optional[float], tolerance: float = 0.01) -> bool:
    """
    Checks if expected numerical value is present within absolute or relative tolerance.
    """
    if expected_val is None:
        return True
    
    nums = extract_numbers_from_text(answer_text)
    if not nums:
        return False

    for n in nums:
        # Check absolute or relative difference
        if abs(n - expected_val) <= tolerance or (expected_val != 0 and abs(n - expected_val) / abs(expected_val) <= tolerance):
            return True
    return False


def evaluate_evidence_grounding(retrieved_texts: List[str], expected_entities: List[str]) -> bool:
    """
    Evaluates whether the actual retrieved documents contain the required evidence entities.
    Strictly checks the retrieved corpus chunks, preventing hallucination or hardcoded answers.
    """
    if not expected_entities:
        return True
    if not retrieved_texts:
        return False

    combined_retrieved = " ".join(retrieved_texts).lower()
    for ent in expected_entities:
        clean_ent = ent.replace(",", "").lower()
        if clean_ent not in combined_retrieved and ent.lower() not in combined_retrieved:
            return False
    return True


def evaluate_abstention(answer_text: str, abstention_expected: bool) -> bool:
    """
    Validates if an unsupported query was properly declined without hallucination.
    """
    if not abstention_expected:
        return True
    
    refusal_keywords = [
        "does not contain", "not available", "insufficient information",
        "no information", "cannot answer", "not found"
    ]
    ans_lower = answer_text.lower()
    return any(kw in ans_lower for kw in refusal_keywords)


class BenchmarkMetricsAggregator:
    @staticmethod
    def aggregate(results: List[Dict[str, Any]], benchmark_suite: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        """
        Aggregates metrics across the benchmark suite.
        Separates retrieval quality metrics from final answer quality metrics.
        """
        retrieval_precisions = []
        retrieval_bounded_precisions = []
        retrieval_r_precisions = []
        retrieval_recalls = []
        retrieval_mrrs = []

        analytical_correct = []
        evidence_grounded = []
        abstention_correct = []
        latencies = []

        query_map = {q["id"]: q for q in benchmark_suite}

        for res in results:
            qid = res.get("query_id") or res.get("id")
            q_spec = query_map.get(qid)
            if not q_spec:
                continue

            category = q_spec["category"]
            ans_text = res.get("answer", "")
            latencies.append(res.get("latency_ms", 0.0))

            # 1. Retrieval Metrics (Evaluated on queries with ground-truth relevant records)
            rel_indices = q_spec.get("relevant_row_indices", [])
            if rel_indices:
                ret_indices = res.get("retrieved_indices", [])
                r_metrics = compute_retrieval_metrics(ret_indices, rel_indices, k=k)
                retrieval_precisions.append(r_metrics["precision_at_k"])
                retrieval_bounded_precisions.append(r_metrics["bounded_precision_at_k"])
                retrieval_r_precisions.append(r_metrics["r_precision"])
                retrieval_recalls.append(r_metrics["recall_at_k"])
                retrieval_mrrs.append(r_metrics["mrr"])

            # 2. Final Answer Quality: Analytical Correctness (Exact dataset calculation)
            if category == "structured_analytical":
                exp_val = q_spec.get("expected_value")
                is_num_correct = evaluate_numerical_correctness(ans_text, exp_val)
                analytical_correct.append(1 if is_num_correct else 0)

            # 3. Retrieval / Evidence Grounding Quality (Evaluated on actual retrieved document chunks)
            elif category == "semantic_retrieval":
                exp_ents = q_spec.get("expected_entities", [])
                ret_texts = res.get("retrieved_texts", [])
                # If retrieved_texts is empty, fall back to checking ans_text only if it is derived from retrieved records
                is_grounded = evaluate_evidence_grounding(ret_texts if ret_texts else [ans_text], exp_ents)
                evidence_grounded.append(1 if is_grounded else 0)

            # 4. Final Answer Quality: Abstention Accuracy (Unsupported queries)
            elif category == "unsupported_abstention":
                is_abstained = evaluate_abstention(ans_text, True)
                abstention_correct.append(1 if is_abstained else 0)

        mean_p = (sum(retrieval_precisions) / len(retrieval_precisions)) if retrieval_precisions else 0.0
        mean_bounded_p = (sum(retrieval_bounded_precisions) / len(retrieval_bounded_precisions)) if retrieval_bounded_precisions else 0.0
        mean_r_prec = (sum(retrieval_r_precisions) / len(retrieval_r_precisions)) if retrieval_r_precisions else 0.0
        mean_r = (sum(retrieval_recalls) / len(retrieval_recalls)) if retrieval_recalls else 0.0
        mean_mrr = (sum(retrieval_mrrs) / len(retrieval_mrrs)) if retrieval_mrrs else 0.0
        
        analytical_acc = (sum(analytical_correct) / len(analytical_correct)) if analytical_correct else 0.0
        evidence_grounding_acc = (sum(evidence_grounded) / len(evidence_grounded)) if evidence_grounded else 0.0
        abstention_acc = (sum(abstention_correct) / len(abstention_correct)) if abstention_correct else 0.0
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        return {
            f"precision_at_{k}": round(mean_p, 4),
            f"bounded_precision_at_{k}": round(mean_bounded_p, 4),
            "r_precision": round(mean_r_prec, 4),
            f"recall_at_{k}": round(mean_r, 4),
            "mrr": round(mean_mrr, 4),
            "numerical_correctness": round(analytical_acc, 4),
            "evidence_grounding": round(evidence_grounding_acc, 4),
            "abstention_accuracy": round(abstention_acc, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_queries_evaluated": len(results)
        }
