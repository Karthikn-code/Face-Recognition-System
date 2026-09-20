"""
Face Recognition System - Web Application Server.

A lightweight, zero-dependency REST API server and static file host built on Python's
built-in http.server. Supports live face detection, ArcFace embedding extraction,
similarity matching, dynamic threshold rejection, identity gallery management,
and evaluation plot inspection.
"""

import http.server
import socketserver
import json
import urllib.parse
import mimetypes
import base64
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.embedder import Embedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher
from src.utils import load_image, draw_annotations

PORT = 8000
STATIC_DIR = config.BASE_DIR / "static"
RESULTS_DIR = config.RESULTS_DIR

# Global shared singletons
print("[+] Initializing system for web application...")
EMBEDDER = Embedder()
DATABASE = FaceDatabase()
MATCHER = FaceMatcher(threshold=config.MATCH_THRESHOLD)


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 encoded image string to an OpenCV BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image data.")
    return img


class FaceRecognitionHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with REST endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(config.BASE_DIR), **kwargs)

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Helper to send JSON response."""
        resp_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        """Helper to send error JSON response."""
        self._send_json({"error": message, "success": False}, status=status)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle HTTP GET requests."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Static root
        if path == "/" or path == "/index.html":
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(index_path.stat().st_size))
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "index.html not found")
                return

        # Serve static assets (/static/...)
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.send_response(200)
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Serve results plots (/results/plots/...)
        if path.startswith("/results/plots/"):
            rel_path = path[len("/results/plots/"):]
            file_path = config.PLOTS_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.send_response(200)
                self.send_header("Content-Type", mime or "image/png")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Serve failure cases (/results/failures/...)
        if path.startswith("/results/failures/"):
            rel_path = path[len("/results/failures/"):]
            file_path = config.FAILURES_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.send_response(200)
                self.send_header("Content-Type", mime or "image/jpeg")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Serve dataset samples (/data/...)
        norm_path = path.replace("\\", "/")
        if norm_path.startswith("/data/"):
            rel_path = norm_path[len("/data/"):]
            file_path = config.DATA_DIR / Path(rel_path)
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.send_response(200)
                self.send_header("Content-Type", mime or "image/jpeg")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # API: Status & Configuration
        if path == "/api/status":
            DATABASE.load()
            people = DATABASE.list_people()
            self._send_json({
                "success": True,
                "backend": EMBEDDER.backend_name.upper(),
                "threshold": config.MATCH_THRESHOLD,
                "min_face_size": config.MIN_FACE_SIZE,
                "min_det_score": config.MIN_DET_SCORE,
                "enrolled_count": len(people),
                "total_templates": sum(p["num_images"] for p in people)
            })
            return

        # API: List enrolled identities
        if path == "/api/identities":
            DATABASE.load()
            people = DATABASE.list_people()
            self._send_json({
                "success": True,
                "identities": people
            })
            return

        # API: Sample test images from LFW
        if path == "/api/samples":
            samples = []
            # Check splits if present
            if config.SPLITS_PATH.exists():
                try:
                    with open(config.SPLITS_PATH) as f:
                        splits = json.load(f)
                    
                    # Add known samples
                    known_added = set()
                    for item in splits.get("test_known", []):
                        p = item["person"]
                        if p not in known_added and len(known_added) < 4:
                            known_added.add(p)
                            clean_p = str(item["path"]).replace("\\", "/")
                            samples.append({
                                "category": "Known Identity (Enrolled/Test)",
                                "person": p,
                                "path": clean_p
                            })

                    # Add unknown samples
                    unknown_added = set()
                    for item in splits.get("test_unknown", []):
                        p = item["person"]
                        if p not in unknown_added and len(unknown_added) < 4:
                            unknown_added.add(p)
                            clean_p = str(item["path"]).replace("\\", "/")
                            samples.append({
                                "category": "Unknown Intruder (Unenrolled)",
                                "person": p,
                                "path": clean_p
                            })
                except Exception:
                    pass

            self._send_json({
                "success": True,
                "samples": samples
            })
            return

        # API: Evaluation Metrics & Failure Summary
        if path == "/api/metrics":
            metrics_data = {}
            if config.METRICS_PATH.exists():
                with open(config.METRICS_PATH) as f:
                    metrics_data = json.load(f)

            # Failure summary text
            failure_text = ""
            if config.FAILURE_SUMMARY_PATH.exists():
                with open(config.FAILURE_SUMMARY_PATH) as f:
                    failure_text = f.read()

            # Failure image list
            failure_images = []
            if config.FAILURES_DIR.exists():
                for img_p in sorted(config.FAILURES_DIR.glob("*.jpg")):
                    failure_images.append({
                        "filename": img_p.name,
                        "url": f"/results/failures/{img_p.name}"
                    })

            self._send_json({
                "success": True,
                "metrics": metrics_data,
                "failure_summary": failure_text,
                "failure_images": failure_images,
                "plots": [
                    {"title": "FAR & FRR vs Threshold", "url": "/results/plots/far_frr_vs_threshold.png"},
                    {"title": "Receiver Operating Characteristic (ROC)", "url": "/results/plots/roc_det_curve.png"},
                    {"title": "Score Distribution (Genuine vs Impostor)", "url": "/results/plots/score_distribution.png"},
                    {"title": "Per-Person Accuracy", "url": "/results/plots/confusion_per_person.png"}
                ]
            })
            return

        # Fallback 404
        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        """Handle HTTP POST requests."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            payload = json.loads(body) if body else {}
        except Exception:
            self._send_error_json("Invalid JSON payload.")
            return

        # API: Identify
        if path == "/api/identify":
            image_data = payload.get("image")
            image_path = payload.get("image_path")
            threshold = float(payload.get("threshold", config.MATCH_THRESHOLD))

            img = None
            if image_data:
                try:
                    img = decode_base64_image(image_data)
                except Exception as e:
                    self._send_error_json(f"Failed to decode base64 image: {e}")
                    return
            elif image_path:
                full_p = config.BASE_DIR / image_path
                if not full_p.exists():
                    self._send_error_json(f"Image path not found: {image_path}")
                    return
                try:
                    img = load_image(full_p)
                except Exception as e:
                    self._send_error_json(f"Failed to load image: {e}")
                    return
            else:
                self._send_error_json("Either 'image' (base64) or 'image_path' must be provided.")
                return

            # Extract faces
            faces = EMBEDDER.get_faces(img)
            if not faces:
                self._send_json({
                    "success": True,
                    "faces": [],
                    "message": "No face detected in image."
                })
                return

            DATABASE.load()
            db_embs = DATABASE.get_all_embeddings()
            results = []

            for face in faces:
                q_emb = face["embedding"]
                match_res = MATCHER.match(q_emb, db_embs, top_k=5, threshold_override=threshold)

                # Format candidates for UI
                formatted_candidates = []
                for cand_name, cand_score in match_res["top_candidates"]:
                    formatted_candidates.append({
                        "name": cand_name,
                        "score": round(float(cand_score), 4),
                        "percentage": round(max(0.0, float(cand_score)) * 100, 1)
                    })

                results.append({
                    "bbox": face["bbox"],
                    "det_score": round(float(face["score"]), 3),
                    "name": match_res["name"],
                    "score": round(float(match_res["score"]), 4),
                    "best_match_person": match_res["best_match_person"],
                    "is_known": match_res["is_known"],
                    "top_candidates": formatted_candidates
                })

            self._send_json({
                "success": True,
                "threshold": threshold,
                "faces_count": len(results),
                "faces": results
            })
            return

        # API: Enroll Identity
        if path == "/api/enroll":
            name = payload.get("name", "").strip().replace(" ", "_")
            images_b64 = payload.get("images", [])

            if not name:
                self._send_error_json("Name is required.")
                return
            if not images_b64:
                self._send_error_json("At least one image is required.")
                return

            person_dir = config.LFW_DIR / name
            person_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []

            for idx, b64_str in enumerate(images_b64, start=1):
                try:
                    img = decode_base64_image(b64_str)
                    out_p = person_dir / f"enrolled_{idx:03d}.jpg"
                    cv2.imwrite(str(out_p), img)
                    saved_paths.append(out_p)
                except Exception as e:
                    pass

            if not saved_paths:
                self._send_error_json("Failed to decode and save enrollment images.")
                return

            added, skipped = DATABASE.enroll(name, saved_paths, EMBEDDER, verbose=True)
            self._send_json({
                "success": True,
                "name": name,
                "added_faces": added,
                "skipped_faces": skipped,
                "total_faces": len(DATABASE.records.get(name, {}).get("embeddings", []))
            })
            return

        # API: Remove Identity
        if path == "/api/remove":
            name = payload.get("name", "").strip()
            if not name:
                self._send_error_json("Name is required.")
                return

            removed = DATABASE.remove(name)
            self._send_json({
                "success": removed,
                "name": name,
                "message": f"Successfully removed '{name}'" if removed else f"'{name}' not found."
            })
            return

        # Fallback 404
        self.send_error(404, "Endpoint not found")


def start_server(port: int = PORT):
    """Start the Face Recognition Web Server."""
    # Ensure static directory exists
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    
    server_address = ("", port)
    with socketserver.TCPServer(server_address, FaceRecognitionHandler) as httpd:
        print("\n" + "=" * 70)
        print(f"       FACE RECOGNITION WEB APPLICATION RUNNING")
        print(f"       URL: http://127.0.0.1:{port}")
        print("=" * 70)
        print("[+] Backend Model : INSIGHTFACE (buffalo_l on CPU)")
        print(f"[+] Operational Threshold: {config.MATCH_THRESHOLD}")
        print("[+] Press Ctrl+C to stop the server.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Shutting down server gracefully...")
            httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face Recognition Web App Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    start_server(port=args.port)
