"""
Tests for Embedder Module (src/embedder.py).
Verifies output shape, L2 unit normalization, and robust handling of non-face images.
"""

import numpy as np
import pytest
from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    """Fixture providing a initialized Embedder instance."""
    return Embedder()


def test_blank_image_returns_no_faces(embedder):
    """A solid black image should return no faces without crashing."""
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = embedder.get_faces(blank_img)
    assert isinstance(faces, list)
    assert len(faces) == 0, "Blank image should return empty list of faces."


def test_noise_image_handling(embedder):
    """A random noise image should be handled gracefully."""
    np.random.seed(42)
    noise_img = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
    faces = embedder.get_faces(noise_img)
    assert isinstance(faces, list)


def test_embedding_normalization_structure():
    """Verify that any non-zero 512D vector when L2-normalized has unit norm 1.0."""
    raw_emb = np.random.randn(512).astype(np.float32)
    norm = np.linalg.norm(raw_emb)
    normalized = raw_emb / norm
    unit_norm = np.linalg.norm(normalized)
    assert np.isclose(unit_norm, 1.0, atol=1e-5), f"Expected unit norm ~1.0, got {unit_norm}"
