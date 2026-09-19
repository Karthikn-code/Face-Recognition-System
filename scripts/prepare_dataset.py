"""
Dataset Preparation and Splitting Script for LFW (Labeled Faces in the Wild).

Downloads LFW automatically using scikit-learn (or direct web fallbacks),
saves images in standard BGR uint8 format under data/lfw/<Person_Name>/<idx>.jpg,
and generates zero-leakage data splits saved to results/splits.json.

Design rationale:
- slice_=None in fetch_lfw_people retains original 250x250 images so face detectors function on un-cropped context.
- Images are converted from float32 RGB [0,1] to uint8 BGR [0,255] for standard OpenCV & InsightFace compatibility.
- Zero-leakage splitting: Enrolled images are strictly disjoint from validation and test sets.
  Unknown people are never present in enrollment or known evaluation sets.
"""

import os
import sys
import json
import random
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

LFW_TGZ_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
MIRROR_LFW_URL = "https://github.com/scikit-learn/scikit-learn-data/raw/main/lfw_home/lfw.tgz"


def ensure_directories() -> None:
    """Create data and results directories if they do not exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.LFW_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def download_lfw_sklearn() -> bool:
    """
    Primary download method using sklearn.datasets.fetch_lfw_people.
    Saves full 250x250 un-cropped BGR images to data/lfw/<Person_Name>/<idx>.jpg.
    """
    print("[+] Attempting primary download method via sklearn.datasets.fetch_lfw_people...")
    try:
        from sklearn.datasets import fetch_lfw_people
        lfw_data = fetch_lfw_people(
            min_faces_per_person=config.MIN_FACES_PER_PERSON,
            color=True,
            resize=1.0,
            slice_=None  # Keep full 250x250 uncropped images
        )

        target_names = lfw_data.target_names
        images = lfw_data.images  # Float images in range [0, 1], shape (N, H, W, 3) in RGB order
        targets = lfw_data.target

        print(f"[+] Downloaded {len(images)} images across {len(target_names)} people from sklearn.")
        
        # Track per-person image count for naming
        person_counts: Dict[str, int] = {}

        for i, img_float in enumerate(images):
            person_name = target_names[targets[i]].replace(" ", "_")
            
            # Convert float RGB [0.0, 1.0] to uint8 BGR [0, 255]
            img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            person_dir = config.LFW_DIR / person_name
            person_dir.mkdir(parents=True, exist_ok=True)

            idx = person_counts.get(person_name, 0) + 1
            person_counts[person_name] = idx

            img_path = person_dir / f"{idx:04d}.jpg"
            cv2.imwrite(str(img_path), img_bgr)

        print("[+] Primary download and conversion completed successfully.")
        return True

    except Exception as e:
        print(f"[!] Primary sklearn download method failed: {e}")
        return False


def download_lfw_archive(url: str, description: str) -> bool:
    """
    Fallback method: Download official LFW tar.gz archive directly and extract.
    """
    import urllib.request
    archive_path = config.DATA_DIR / "lfw.tgz"
    print(f"[+] Attempting fallback method ({description}) from {url}...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r    Downloading {description}: {percent}% [{count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, archive_path, reporthook=reporthook)
        print("\n[+] Extracting archive...")
        
        extract_dir = config.DATA_DIR / "extracted_lfw"
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # Locate extracted lfw directory
        raw_lfw = extract_dir / "lfw"
        if not raw_lfw.exists():
            # Search nested directories
            found = list(extract_dir.glob("**/lfw"))
            if found:
                raw_lfw = found[0]

        # Copy persons with min_faces_per_person images to data/lfw
        copied_count = 0
        for person_dir in raw_lfw.iterdir():
            if person_dir.is_dir():
                images = list(person_dir.glob("*.jpg"))
                if len(images) >= config.MIN_FACES_PER_PERSON:
                    dest_dir = config.LFW_DIR / person_dir.name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for img in images:
                        shutil.copy(img, dest_dir / img.name)
                        copied_count += 1

        print(f"[+] Fallback extraction finished. Retained {copied_count} images.")
        
        # Cleanup temporary archive & extracted folder
        if archive_path.exists():
            archive_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        return True

    except Exception as e:
        print(f"\n[!] Fallback download from {url} failed: {e}")
        return False


def load_dataset_structure() -> Dict[str, List[Path]]:
    """Scan data/lfw and return dict mapping Person_Name to list of image paths."""
    dataset: Dict[str, List[Path]] = {}
    if not config.LFW_DIR.exists():
        return dataset

    for person_dir in sorted(config.LFW_DIR.iterdir()):
        if person_dir.is_dir():
            images = sorted(list(person_dir.glob("*.jpg")))
            if len(images) >= config.MIN_FACES_PER_PERSON:
                dataset[person_dir.name] = images
    return dataset


def create_splits(dataset: Dict[str, List[Path]]) -> Dict:
    """
    Partition dataset into KNOWN (~40 people) and UNKNOWN (~40 people).
    - KNOWN split:
      - 2-3 images -> enrollment
      - 1-2 images -> val_known
      - remaining -> test_known
    - UNKNOWN split:
      - ~20 people -> val_unknown
      - ~20 people -> test_unknown

    Assert zero data leakage programmatically.
    """
    random.seed(config.RANDOM_SEED)
    people = sorted(list(dataset.keys()))

    if len(people) < config.NUM_KNOWN_PEOPLE + config.NUM_UNKNOWN_PEOPLE:
        raise ValueError(
            f"Not enough people with >= {config.MIN_FACES_PER_PERSON} images. "
            f"Found {len(people)}, need at least {config.NUM_KNOWN_PEOPLE + config.NUM_UNKNOWN_PEOPLE}."
        )

    # Shuffle people deterministically
    random.shuffle(people)

    known_people = people[:config.NUM_KNOWN_PEOPLE]
    unknown_people = people[config.NUM_KNOWN_PEOPLE : config.NUM_KNOWN_PEOPLE + config.NUM_UNKNOWN_PEOPLE]

    # Split UNKNOWN people between validation and test (~20 each)
    val_unknown_people = unknown_people[:20]
    test_unknown_people = unknown_people[20:]

    enrollment_split: List[Dict[str, str]] = []
    val_known_split: List[Dict[str, str]] = []
    test_known_split: List[Dict[str, str]] = []
    val_unknown_split: List[Dict[str, str]] = []
    test_unknown_split: List[Dict[str, str]] = []

    for person in known_people:
        imgs = dataset[person].copy()
        # Shuffle person's images deterministically
        random.shuffle(imgs)

        n = len(imgs)
        # Allocate: 2-3 to enrollment, 1-2 to val, rest to test
        n_enroll = 3 if n >= 6 else 2
        n_val = 2 if (n - n_enroll) >= 3 else 1

        enroll_imgs = imgs[:n_enroll]
        val_imgs = imgs[n_enroll : n_enroll + n_val]
        test_imgs = imgs[n_enroll + n_val :]

        for img in enroll_imgs:
            enrollment_split.append({"person": person, "path": str(img.relative_to(config.BASE_DIR))})
        for img in val_imgs:
            val_known_split.append({"person": person, "path": str(img.relative_to(config.BASE_DIR))})
        for img in test_imgs:
            test_known_split.append({"person": person, "path": str(img.relative_to(config.BASE_DIR))})

    # Sample up to 4 images per unknown person for balanced representation (avoids 1 person like Bush dominating 70% of evaluations)
    MAX_UNKNOWN_IMAGES_PER_PERSON = 4
    for person in val_unknown_people:
        p_imgs = dataset[person].copy()
        random.shuffle(p_imgs)
        for img in p_imgs[:MAX_UNKNOWN_IMAGES_PER_PERSON]:
            val_unknown_split.append({"person": person, "path": str(img.relative_to(config.BASE_DIR))})

    for person in test_unknown_people:
        p_imgs = dataset[person].copy()
        random.shuffle(p_imgs)
        for img in p_imgs[:MAX_UNKNOWN_IMAGES_PER_PERSON]:
            test_unknown_split.append({"person": person, "path": str(img.relative_to(config.BASE_DIR))})

    splits_data = {
        "seed": config.RANDOM_SEED,
        "num_known_people": len(known_people),
        "num_unknown_people": len(unknown_people),
        "enrollment": enrollment_split,
        "val_known": val_known_split,
        "val_unknown": val_unknown_split,
        "test_known": test_known_split,
        "test_unknown": test_unknown_split,
    }

    # Programmatic zero-leakage assertions
    enroll_paths = {item["path"] for item in enrollment_split}
    val_paths = {item["path"] for item in val_known_split} | {item["path"] for item in val_unknown_split}
    test_paths = {item["path"] for item in test_known_split} | {item["path"] for item in test_unknown_split}

    assert len(enroll_paths & val_paths) == 0, "DATA LEAKAGE ERROR: Enrollment image found in Validation set!"
    assert len(enroll_paths & test_paths) == 0, "DATA LEAKAGE ERROR: Enrollment image found in Test set!"
    assert len(val_paths & test_paths) == 0, "DATA LEAKAGE ERROR: Validation image found in Test set!"

    known_set = set(known_people)
    unknown_set = set(unknown_people)
    assert len(known_set & unknown_set) == 0, "DATA LEAKAGE ERROR: Overlap between Known and Unknown people!"

    print("[+] Zero data leakage assertions passed successfully.")
    return splits_data


def main():
    print("=" * 70)
    print("      LFW DATASET PREPARATION & ZERO-LEAKAGE SPLITTING")
    print("=" * 70)

    ensure_directories()

    # Check if dataset is already populated
    existing_dataset = load_dataset_structure()
    if len(existing_dataset) >= config.NUM_KNOWN_PEOPLE + config.NUM_UNKNOWN_PEOPLE:
        print(f"[+] Dataset already present: found {len(existing_dataset)} people with >= {config.MIN_FACES_PER_PERSON} images.")
    else:
        # Download pipeline
        success = download_lfw_sklearn()
        if not success:
            success = download_lfw_archive(LFW_TGZ_URL, "Official LFW Archive")
        if not success:
            success = download_lfw_archive(MIRROR_LFW_URL, "Mirror LFW Archive")

        if not success:
            print("\n" + "!" * 70)
            print("CRITICAL ERROR: All automated download attempts failed.")
            print("Failed URLs:")
            print(f"  1. sklearn.datasets.fetch_lfw_people")
            print(f"  2. {LFW_TGZ_URL}")
            print(f"  3. {MIRROR_LFW_URL}")
            print("\nManual Instructions:")
            print("  Please download 'lfw.tgz' manually from http://vis-www.cs.umass.edu/lfw/lfw.tgz")
            print("  and place it at 'data/lfw.tgz', then re-run this script.")
            print("!" * 70)
            sys.exit(1)

        existing_dataset = load_dataset_structure()

    # Create zero-leakage splits
    print("[+] Creating zero-leakage dataset splits...")
    splits_data = create_splits(existing_dataset)

    with open(config.SPLITS_PATH, "w") as f:
        json.dump(splits_data, f, indent=2)

    print(f"[+] Data splits successfully saved to {config.SPLITS_PATH}")
    print("\n" + "-" * 70)
    print("                    DATASET SUMMARY TABLE")
    print("-" * 70)
    print(f"Total People (>= {config.MIN_FACES_PER_PERSON} images) : {len(existing_dataset)}")
    print(f"Known People (Enrolled)     : {splits_data['num_known_people']}")
    print(f"Unknown People (Unenrolled) : {splits_data['num_unknown_people']}")
    print(f"  - Validation Unknowns     : 20 people")
    print(f"  - Test Unknowns           : 20 people")
    print(f"Enrollment Images           : {len(splits_data['enrollment'])}")
    print(f"Validation Queries (Known)  : {len(splits_data['val_known'])}")
    print(f"Validation Queries (Unknown): {len(splits_data['val_unknown'])}")
    print(f"Test Queries (Known)        : {len(splits_data['test_known'])}")
    print(f"Test Queries (Unknown)      : {len(splits_data['test_unknown'])}")
    print("-" * 70)


if __name__ == "__main__":
    main()
