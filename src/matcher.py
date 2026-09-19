"""
Face Matcher Module.

Performs Similarity-Based Matching using Cosine Similarity between L2-normalized 512D vectors
and enforces an "Unknown" Rejection Mechanism based on a tunable operating threshold.

Matching Modes:
1. `max_similarity` (Default):
   Calculates dot product similarity between query vector `q` and ALL stored vectors `{e_1, e_2, ...}`
   for identity P, returning max(q . e_i). Robust to posture/lighting variations across enrolled images.

2. `mean_prototype`:
   Calculates average normalized prototype vector `u_P = normalize(mean(e_1, e_2, ...))`
   for identity P, returning dot product (q . u_P). Compact representation.

Design Rationale & Interview Notes:
- L2 Normalization Guarantee: Since both query and database vectors have ||v||_2 = 1.0,
  cosine_similarity(A, B) = A . B (dot product), avoiding expensive division operations.
- Unknown Rejection: If the maximum candidate similarity score is below `threshold`,
  the subject is classified as "unknown" to prevent False Acceptances of un-enrolled intruders.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import config


class FaceMatcher:
    """
    Performs cosine similarity matching of query embeddings against database records.
    """
    def __init__(
        self,
        threshold: float = config.MATCH_THRESHOLD,
        mode: str = config.MATCH_MODE
    ):
        self.threshold = threshold
        self.mode = mode

    def compute_similarity(self, query_emb: np.ndarray, target_embs: np.ndarray) -> float:
        """
        Compute cosine similarity score between a single query embedding (512,)
        and target embeddings array (N, 512).

        Returns float similarity score in range [-1.0, 1.0].
        """
        # Ensure query is 1D vector of shape (512,)
        if query_emb.ndim == 2:
            query_emb = query_emb.squeeze(0)

        # Ensure L2 unit norm for query vector
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0 and not np.isclose(q_norm, 1.0):
            query_emb = query_emb / q_norm

        if target_embs.ndim == 1:
            target_embs = target_embs.reshape(1, -1)

        # Compute dot products with all target vectors for this person
        sims = np.dot(target_embs, query_emb)  # Shape (N,)

        if self.mode == "mean_prototype":
            # Compute average embedding vector and re-normalize to unit length
            mean_vec = np.mean(target_embs, axis=0)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm > 0:
                mean_vec = mean_vec / mean_norm
            return float(np.dot(mean_vec, query_emb))
        else: # "max_similarity"
            return float(np.max(sims))

    def match(
        self,
        query_emb: np.ndarray,
        db_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
        threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Identify the closest match for query_emb across all identities in db_embeddings.

        Args:
            query_emb: L2-normalized 512D query embedding vector.
            db_embeddings: Dict mapping person_name -> np.ndarray of shape (N, 512).
            top_k: Number of top candidate matches to return.
            threshold_override: Optional threshold to override self.threshold.

        Returns:
            Dict containing:
            {
                "name": str,                  # Matched person name or "unknown"
                "score": float,               # Highest similarity score
                "best_match_person": str,     # Nearest neighbor person name
                "is_known": bool,             # True if score >= threshold
                "top_candidates": [(name, score), ...]
            }
        """
        threshold = threshold_override if threshold_override is not None else self.threshold

        if not db_embeddings:
            return {
                "name": "unknown",
                "score": 0.0,
                "best_match_person": "None",
                "is_known": False,
                "top_candidates": []
            }

        scores: List[Tuple[str, float]] = []

        for person_name, target_embs in db_embeddings.items():
            if target_embs is None or len(target_embs) == 0:
                continue
            sim_score = self.compute_similarity(query_emb, target_embs)
            scores.append((person_name, sim_score))

        if not scores:
            return {
                "name": "unknown",
                "score": 0.0,
                "best_match_person": "None",
                "is_known": False,
                "top_candidates": []
            }

        # Sort candidates by similarity score descending
        scores.sort(key=lambda item: item[1], reverse=True)

        best_person, best_score = scores[0]
        top_candidates = scores[:top_k]

        # Apply unknown rejection threshold decision rule
        if best_score >= threshold:
            decision_name = best_person
            is_known = True
        else:
            decision_name = "unknown"
            is_known = False

        return {
            "name": decision_name,
            "score": best_score,
            "best_match_person": best_person,
            "is_known": is_known,
            "top_candidates": top_candidates
        }
