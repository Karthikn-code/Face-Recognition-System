"""
Configuration file for Face Recognition Identification System.
Centralizes all file paths, model parameters, matching thresholds, and random seeds.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LFW_DIR = DATA_DIR / "lfw"
RESULTS_DIR = BASE_DIR / "results"
FAILURES_DIR = RESULTS_DIR / "failures"
PLOTS_DIR = RESULTS_DIR / "plots"

# File Paths
DB_PATH = DATA_DIR / "embeddings_db.pkl"
SPLITS_PATH = RESULTS_DIR / "splits.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FAILURE_SUMMARY_PATH = RESULTS_DIR / "failure_summary.md"

# Dataset & Splitting Parameters
RANDOM_SEED = 42
MIN_FACES_PER_PERSON = 5
NUM_KNOWN_PEOPLE = 40
NUM_UNKNOWN_PEOPLE = 40

# Face Detection & Embedding Parameters
MIN_DET_SCORE = 0.50     # Minimum confidence score for a detected face
MIN_FACE_SIZE = 30       # Minimum width/height in pixels for a valid face bbox
EMBEDDING_DIM = 512      # Dimensionality of L2-normalized ArcFace/FaceNet embeddings

# Matcher & Decision Parameters
# Calibrated optimal operating threshold tuned programmatically on the validation split
MATCH_THRESHOLD = 0.60
MATCH_MODE = "max_similarity"  # Options: "max_similarity" or "mean_prototype"

# Fallback Backends
# Primary: insightface (buffalo_l model pack)
# Fallback: facenet-pytorch (MTCNN + InceptionResnetV1)
PREFERRED_BACKEND = "insightface"
FALLBACK_BACKEND = "facenet_pytorch"
