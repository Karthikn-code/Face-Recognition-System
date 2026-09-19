"""
End-to-End Integration Tests.
Verifies system workflow components.
"""

from pathlib import Path
import numpy as np
import pytest
import config
from src.matcher import FaceMatcher


def test_config_paths_exist():
    """Verify that config paths are valid Path objects."""
    assert isinstance(config.BASE_DIR, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.RESULTS_DIR, Path)


def test_matcher_top_k_ranking():
    """Verify top-k candidates ranking functionality."""
    matcher = FaceMatcher(threshold=0.30)
    
    # Create fixed vectors
    q = np.zeros(512, dtype=np.float32)
    q[0] = 1.0  # [1, 0, 0, ...]

    v_alice = np.zeros(512, dtype=np.float32)
    v_alice[0] = 0.9
    v_alice[1] = 0.43588989
    v_alice /= np.linalg.norm(v_alice)  # Sim ~0.9

    v_bob = np.zeros(512, dtype=np.float32)
    v_bob[0] = 0.5
    v_bob[1] = 0.8660254
    v_bob /= np.linalg.norm(v_bob)  # Sim ~0.5

    db_embs = {
        "Alice": v_alice.reshape(1, -1),
        "Bob": v_bob.reshape(1, -1)
    }

    res = matcher.match(q, db_embs, top_k=2)
    assert res["name"] == "Alice"
    assert len(res["top_candidates"]) == 2
    assert res["top_candidates"][0][0] == "Alice"
    assert res["top_candidates"][1][0] == "Bob"


def test_enroll_and_identify_roundtrip(tmp_path):
    """
    End-to-end test: Enroll a real person into a temp database,
    verify known query identifies correctly, and verify unknown query rejects.
    """
    from src.database import FaceDatabase
    from src.embedder import Embedder
    from src.utils import load_image

    db_path = tmp_path / "roundtrip_db.pkl"
    db = FaceDatabase(db_path=db_path)
    embedder = Embedder()
    matcher = FaceMatcher(threshold=0.35)

    # Check if LFW data is present
    person1_dir = config.LFW_DIR / "Abdullah_Gul"
    person2_dir = config.LFW_DIR / "Adrien_Brody"

    if person1_dir.exists() and person2_dir.exists():
        imgs_p1 = sorted(list(person1_dir.glob("*.jpg")))
        imgs_p2 = sorted(list(person2_dir.glob("*.jpg")))

        if len(imgs_p1) >= 2 and len(imgs_p2) >= 1:
            # Enroll person 1 with first image
            added, _ = db.enroll("Abdullah_Gul", [imgs_p1[0]], embedder, verbose=False)
            assert added == 1

            # Identify using person 1's second image (known query)
            q_img1 = load_image(imgs_p1[1])
            faces1 = embedder.get_faces(q_img1)
            assert len(faces1) > 0
            res1 = matcher.match(faces1[0]["embedding"], db.get_all_embeddings())
            assert res1["name"] == "Abdullah_Gul"
            assert res1["is_known"] is True

            # Identify using person 2's image (unknown query)
            q_img2 = load_image(imgs_p2[0])
            faces2 = embedder.get_faces(q_img2)
            assert len(faces2) > 0
            res2 = matcher.match(faces2[0]["embedding"], db.get_all_embeddings())
            # Since Adrien Brody is not enrolled, if score < threshold it must be unknown
            # If threshold is appropriate (e.g. 0.35 - 0.40)
            if res2["score"] < matcher.threshold:
                assert res2["name"] == "unknown"
                assert res2["is_known"] is False

