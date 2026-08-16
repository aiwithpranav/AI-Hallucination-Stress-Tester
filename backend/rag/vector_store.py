"""
RAG Vector Store Module

Builds a temporary FAISS index per request for similarity ranking
of evidence passages against the LLM-generated answer.

Uses FAISS IndexFlatIP (inner product) on L2-normalized vectors,
which is equivalent to cosine similarity.

The index is in-memory and per-request — no persistence layer required.

Usage:
    store = VectorStore()
    top_k = store.search(answer_vector, passage_vectors, passages, k=3)
    # Returns list of (passage_dict, similarity_score) tuples
"""

from typing import List, Tuple, Dict, Any

import numpy as np


class VectorStore:
    """
    In-memory FAISS similarity search over evidence passages.
    A new instance is created per verification request.
    """

    def __init__(self):
        self._faiss = None
        self._ensure_faiss()

    def _ensure_faiss(self):
        """Import FAISS (raises ImportError with helpful message if missing)."""
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed. "
                "Run: pip install faiss-cpu"
            )

    # ─── Public Interface ─────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        passage_vectors: np.ndarray,
        passages: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find the top-k most similar passages to the query vector.

        Args:
            query_vector:   Embedding of the LLM answer (1D float32 array).
            passage_vectors: Embeddings of evidence passages (2D float32 array).
            passages:       Original passage dicts (same order as passage_vectors).
            k:              Number of top results to return.

        Returns:
            List of (passage_dict, similarity_score) tuples, sorted by score desc.
            similarity_score is in [0, 1] (cosine similarity, normalized).
        """
        if passage_vectors is None or len(passage_vectors) == 0:
            return []

        n_passages = len(passage_vectors)
        if n_passages == 0:
            return []

        k = min(k, n_passages)  # Can't return more than we have

        embedding_dim = passage_vectors.shape[1]

        # Build FAISS index (inner product on normalized vectors = cosine sim)
        index = self._faiss.IndexFlatIP(embedding_dim)
        index.add(passage_vectors)

        # Query: reshape to 2D (1, dim) as FAISS expects batch input
        query_2d = query_vector.reshape(1, -1)

        # Search
        scores, indices = index.search(query_2d, k)

        # Build result list
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(passages):
                # FAISS may return -1 for invalid results
                continue
            # Clamp score to [0, 1] — inner product on normalized vecs is in [-1, 1]
            clamped_score = float(max(0.0, min(1.0, score)))
            results.append((passages[idx], clamped_score))

        return results

    def compute_average_similarity(
        self,
        query_vector: np.ndarray,
        passage_vectors: np.ndarray,
        top_k: int = 3,
    ) -> float:
        """
        Compute the average cosine similarity between the query vector
        and the top-k most similar passage vectors.

        Args:
            query_vector:    Embedding of the answer.
            passage_vectors: Embeddings of all evidence passages.
            top_k:           Number of top passages to average over.

        Returns:
            Float in [0, 1] — average similarity score.
        """
        if passage_vectors is None or len(passage_vectors) == 0:
            return 0.0

        n = len(passage_vectors)
        if n == 0:
            return 0.0

        top_k = min(top_k, n)
        embedding_dim = passage_vectors.shape[1]

        index = self._faiss.IndexFlatIP(embedding_dim)
        index.add(passage_vectors)

        query_2d = query_vector.reshape(1, -1)
        scores, _ = index.search(query_2d, top_k)

        valid_scores = [
            max(0.0, min(1.0, float(s)))
            for s in scores[0]
            if s >= 0  # FAISS returns -1 for invalid
        ]

        return float(np.mean(valid_scores)) if valid_scores else 0.0
