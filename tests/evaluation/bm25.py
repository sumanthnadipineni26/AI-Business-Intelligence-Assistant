"""
Pure-Python BM25Okapi implementation.
Zero external dependencies required.
"""
import math
import re
from typing import List, Dict, Any, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Okapi:
    """
    Standard BM25Okapi ranking algorithm.
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = (
            sum(self.doc_lengths) / self.corpus_size
            if self.corpus_size > 0 else 0.0
        )
        
        # Calculate document frequencies
        self.df: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        for doc in self.tokenized_corpus:
            tf: Dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            for token in tf.keys():
                self.df[token] = self.df.get(token, 0) + 1

        # Calculate Robertson-Spärck Jones IDF
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: str) -> List[float]:
        """Compute BM25 scores for all documents in the corpus for a given query."""
        query_tokens = tokenize(query)
        scores = [0.0] * self.corpus_size
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, tf_dict in enumerate(self.doc_term_freqs):
                freq = tf_dict.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0))
                scores[i] += idf_val * (numerator / denominator)

        return scores

    def query(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return top_k document indices and their BM25 scores sorted descending."""
        scores = self.get_scores(query)
        scored_indices = [(i, score) for i, score in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        return scored_indices[:top_k]
