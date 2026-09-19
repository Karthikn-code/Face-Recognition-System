"""
Tests for Face Matcher Module (src/matcher.py).
Verifies cosine similarity, threshold rejection, and candidate ranking.
"""

import numpy as np
import pytest
from src.matcher import FaceMatcher


def create_normalized_vector():
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_perfect_match_similarity():
    """Matching identical unit vector should give similarity score = 1.0."""
    matcher = FaceMatcher(threshold=0.40)
    emb = create_normalized_vector()
    sim = matcher.compute_similarity(emb, emb.reshape(1, -1))
    assert np.isclose(sim, 1.0, atol=1e-4)


def test_known_person_match():
    """Matching vector against target containing identical vector should return identity."""
    matcher = FaceMatcher(threshold=0.40)
    alice_emb = create_normalized_vector()
    bob_emb = create_normalized_vector()

    db_embs = {
        "Alice": alice_emb.reshape(1, -1),
        "Bob": bob_emb.reshape(1, -1)
    }

    res = matcher.match(alice_emb, db_embs)
    assert res["name"] == "Alice"
    assert res["is_known"] is True
    assert np.isclose(res["score"], 1.0, atol=1e-4)


def test_unknown_rejection():
    """Query vector far from database embeddings should be rejected as 'unknown'."""
    matcher = FaceMatcher(threshold=0.80)  # High threshold
    v1 = create_normalized_vector()
    # Create orthogonal vector
    v2 = create_normalized_vector()

    db_embs = {"Alice": v1.reshape(1, -1)}
    res = matcher.match(v2, db_embs)

    assert res["name"] == "unknown"
    assert res["is_known"] is False
    assert res["best_match_person"] == "Alice"
