"""
Pytest integration suite for isolated RAG evaluation benchmark.
"""
import pytest
import os
import sys
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.evaluation.document_indexer import DatasetCorpus
from tests.evaluation.bm25 import BM25Okapi
from tests.evaluation.rag_baseline import ProductionRAGBaselineAdapter
from tests.evaluation.retrieval_adapters import (
    BM25RetrieverAdapter,
    DenseFAISSRetrieverAdapter,
    HybridRRFAdapter,
    GroundedPipelineRunner
)
from tests.evaluation.metrics_calculator import (
    compute_retrieval_metrics,
    evaluate_numerical_correctness,
    evaluate_evidence_grounding,
    evaluate_abstention
)


def test_dataset_corpus_loading_and_isolation():
    corpus = DatasetCorpus()
    assert "sample_business_dataset" in corpus.datasets
    assert "enterprise_sales" in corpus.datasets
    assert "student_data" in corpus.datasets

    # Verify real row counts
    assert len(corpus.get_documents("sample_business_dataset")) == 10
    assert len(corpus.get_documents("enterprise_sales")) == 5
    assert len(corpus.get_documents("student_data")) == 50

    # Verify dataset isolation enforcement
    with pytest.raises(KeyError):
        corpus.get_documents("nonexistent_dataset")

    with pytest.raises(KeyError):
        corpus.get_documents("")


def test_bm25_pure_python():
    docs = [
        "TechMart purchased a printer in the south region with high profit",
        "ABC Corp purchased laptops and monitors",
        "OfficeHub ordered furniture desk"
    ]
    bm25 = BM25Okapi(docs)
    results = bm25.query("printer TechMart", top_k=2)
    assert len(results) == 2
    # Document 0 should be top ranked
    assert results[0][0] == 0
    assert results[0][1] > 0.0


def test_metrics_calculator_edge_cases():
    # Perfect retrieval where |Relevant| = 1 and k = 5
    m = compute_retrieval_metrics(retrieved_indices=[3, 1, 2], relevant_indices=[3], k=5)
    assert m["precision_at_k"] == 1.0 / 5.0
    assert m["bounded_precision_at_k"] == 1.0
    assert m["r_precision"] == 1.0
    assert m["recall_at_k"] == 1.0
    assert m["mrr"] == 1.0

    # Case where no relevant items are retrieved
    m_miss = compute_retrieval_metrics(retrieved_indices=[0, 1, 2], relevant_indices=[3], k=5)
    assert m_miss["precision_at_k"] == 0.0
    assert m_miss["bounded_precision_at_k"] == 0.0
    assert m_miss["r_precision"] == 0.0
    assert m_miss["recall_at_k"] == 0.0
    assert m_miss["mrr"] == 0.0

    # Numerical correctness within tolerance
    assert evaluate_numerical_correctness("Total revenue reaches $67,300.00", 67300.0) is True
    assert evaluate_numerical_correctness("Total revenue reaches $50,000.00", 67300.0) is False

    # Evidence grounding against retrieved chunks
    retrieved_chunks = [
        "Record #5 | customer: TechMart | product: Printer | profit: 900",
        "Record #1 | customer: XYZ Ltd | product: Phone"
    ]
    assert evaluate_evidence_grounding(retrieved_chunks, ["TechMart", "Printer", "900"]) is True
    assert evaluate_evidence_grounding(retrieved_chunks, ["Global Inc", "Laptop", "15000"]) is False

    # Abstention verification
    assert evaluate_abstention("The uploaded dataset does not contain student marks/GPA information.", True) is True
    assert evaluate_abstention("The student marks are 95%.", True) is False


def test_baseline_dataset_isolation():
    corpus = DatasetCorpus()
    baseline = ProductionRAGBaselineAdapter(corpus)
    with pytest.raises(ValueError, match="Dataset key is missing"):
        baseline.query("What is revenue?", "")
