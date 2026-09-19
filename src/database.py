"""
Face Database Module.

Manages persistent storage of enrolled identities, face embeddings (512D L2-normalized vectors),
and associated metadata (image sources, image counts).

Storage format: Pickled dictionary saved to data/embeddings_db.pkl.
Structure:
{
    "Person_Name": {
        "embeddings": [np.ndarray (512,), ...],
        "sources": ["data/lfw/Alice/0001.jpg", ...]
    }, ...
}
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import config
from src.utils import load_image


class FaceDatabase:
    """
    Manages enrollment, persistence, retrieval, and deletion of identity embeddings.
    """
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        # Internal state: dict[person_name] = {"embeddings": List[np.ndarray], "sources": List[str]}
        self.records: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load records from pickle database file if it exists."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    self.records = pickle.load(f)
                print(f"[+] Loaded database from {self.db_path} ({len(self.records)} identities).")
            except Exception as e:
                print(f"[!] Warning: Failed to load database from {self.db_path}: {e}")
                self.records = {}
        else:
            self.records = {}

    def save(self) -> None:
        """Save records to pickle database file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.records, f)

    def enroll(
        self,
        name: str,
        image_paths: List[Union[str, Path]],
        embedder: Any,
        verbose: bool = True
    ) -> Tuple[int, int]:
        """
        Enroll a person by processing image paths and storing face embeddings.

        Args:
            name: Identity name string (e.g. "Alice_Smith")
            image_paths: List of file paths to enrollment images.
            embedder: Embedder instance (provides get_faces(image)).
            verbose: If True, prints warnings and progress messages.

        Returns:
            Tuple (successfully_added_count, skipped_count)
        """
        if not image_paths:
            if verbose:
                print(f"[!] No image paths provided for {name}.")
            return (0, 0)

        person_entry = self.records.get(name, {"embeddings": [], "sources": []})
        added_count = 0
        skipped_count = 0

        for img_p in image_paths:
            path_str = str(img_p)
            try:
                img = load_image(path_str)
                faces = embedder.get_faces(img)

                if len(faces) == 0:
                    if verbose:
                        print(f"[!] Warning: No face detected in {path_str}. Skipping.")
                    skipped_count += 1
                    continue

                if len(faces) > 1 and verbose:
                    print(
                        f"[!] Warning: Multiple ({len(faces)}) faces detected in {path_str}. "
                        "Using the largest face bounding box for enrollment."
                    )

                # Select largest face (index 0 since faces are sorted by area descending)
                best_face = faces[0]
                person_entry["embeddings"].append(best_face["embedding"])
                person_entry["sources"].append(path_str)
                added_count += 1

            except Exception as e:
                if verbose:
                    print(f"[!] Error processing {path_str} during enrollment: {e}")
                skipped_count += 1

        if added_count > 0:
            self.records[name] = person_entry
            self.save()
            if verbose:
                print(
                    f"[+] Successfully enrolled '{name}': {added_count} embeddings added "
                    f"({skipped_count} skipped). Total images for {name}: {len(person_entry['embeddings'])}."
                )

        return (added_count, skipped_count)

    def add_more_images(
        self,
        name: str,
        image_paths: List[Union[str, Path]],
        embedder: Any,
        verbose: bool = True
    ) -> Tuple[int, int]:
        """Add more enrollment images to an existing identity record."""
        return self.enroll(name, image_paths, embedder, verbose=verbose)

    def remove(self, name: str) -> bool:
        """Remove a person identity record from the database."""
        if name in self.records:
            del self.records[name]
            self.save()
            print(f"[+] Removed '{name}' from database.")
            return True
        print(f"[!] Identity '{name}' not found in database.")
        return False

    def list_people(self) -> List[Dict[str, Any]]:
        """Return list of enrolled people with metadata."""
        summary = []
        for name, data in self.records.items():
            summary.append({
                "name": name,
                "num_images": len(data["embeddings"]),
                "sources": data["sources"]
            })
        return summary

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Return dict mapping person_name -> np.ndarray of shape (N, 512)
        where each row is an L2-normalized 512D embedding vector.
        """
        result = {}
        for name, data in self.records.items():
            if data["embeddings"]:
                result[name] = np.array(data["embeddings"], dtype=np.float32)
        return result

    def clear(self) -> None:
        """Clear all records from database and remove database file."""
        self.records = {}
        if self.db_path.exists():
            self.db_path.unlink()
