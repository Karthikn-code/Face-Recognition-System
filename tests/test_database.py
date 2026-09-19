"""
Tests for Face Database Module (src/database.py).
Verifies save, load, list, remove, and clear operations.
"""

from pathlib import Path
import numpy as np
import pytest
from src.database import FaceDatabase


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary database instance backed by tmp_path."""
    db_file = tmp_path / "test_db.pkl"
    db = FaceDatabase(db_path=db_file)
    yield db
    db.clear()


def test_db_initialization(temp_db):
    """New database should be empty."""
    assert len(temp_db.records) == 0
    assert len(temp_db.list_people()) == 0


def test_db_save_and_load(tmp_path):
    """Test serialization and deserialization of records."""
    db_file = tmp_path / "test_db.pkl"
    db = FaceDatabase(db_path=db_file)

    # Manually insert dummy normalized embedding
    dummy_emb = np.random.randn(512).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    db.records["Alice"] = {
        "embeddings": [dummy_emb],
        "sources": ["dummy/path/0001.jpg"]
    }
    db.save()

    # Load into fresh instance
    loaded_db = FaceDatabase(db_path=db_file)
    assert "Alice" in loaded_db.records
    assert len(loaded_db.records["Alice"]["embeddings"]) == 1
    assert loaded_db.records["Alice"]["sources"][0] == "dummy/path/0001.jpg"


def test_db_remove(temp_db):
    """Test removing an identity from database."""
    temp_db.records["Bob"] = {
        "embeddings": [np.random.randn(512)],
        "sources": ["dummy/bob.jpg"]
    }
    temp_db.save()

    assert "Bob" in temp_db.records
    success = temp_db.remove("Bob")
    assert success is True
    assert "Bob" not in temp_db.records
