"""
Production FastAPI Application Server for Face Recognition Identification System.

Designed to meet high-concurrency production standards:
- Asynchronous non-blocking architecture using asyncio.to_thread for CPU-bound ONNX model inference.
- Strict Pydantic models for request and response validation.
- Interactive OpenAPI documentation (/docs and /redoc).
- Static file serving for the glassmorphic frontend dashboard.
- CORS middleware enabled for cross-origin client integration.
"""

import sys
import base64
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.embedder import Embedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher
from src.utils import load_image

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CandidateMatch(BaseModel):
    name: str
    score: float
    percentage: float

class DetectedFaceResult(BaseModel):
    bbox: List[int]
    det_score: float
    name: str
    score: float
    best_match_person: str
    is_known: bool
    top_candidates: List[CandidateMatch]

class IdentifyRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image string (data:image/jpeg;base64,...)")
    image_path: Optional[str] = Field(None, description="Relative server path to test image file")
    threshold: Optional[float] = Field(config.MATCH_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold tau")

class IdentifyResponse(BaseModel):
    success: bool
    threshold: float
    num_faces: int
    faces: List[DetectedFaceResult]
    message: Optional[str] = None

class EnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Person's full name")
    images: List[str] = Field(..., min_length=1, description="List of base64 encoded image strings")

class EnrollResponse(BaseModel):
    success: bool
    name: str
    added_count: int
    skipped_count: int
    message: str

class DeleteIdentityRequest(BaseModel):
    name: str

class SystemStatusResponse(BaseModel):
    success: bool
    backend: str
    threshold: float
    min_face_size: int
    min_det_score: float
    enrolled_count: int
    total_templates: int

# ---------------------------------------------------------------------------
# Lifespan Context Manager (Startup / Shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load neural models and database once at server startup."""
    print("[+] [FastAPI] Initializing Face Recognition Singletons...")
    app.state.embedder = Embedder()
    app.state.db = FaceDatabase()
    app.state.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    print(f"[+] [FastAPI] Ready. Backend: {app.state.embedder.backend_name.upper()} | Enrolled: {len(app.state.db.list_people())}")
    yield
    print("[-] [FastAPI] Shutting down Face Recognition Server.")

# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FaceID.ai Biometric Microservice",
    description="High-performance, CPU-optimized Open-Set Face Recognition & Attendance API powered by ArcFace, RetinaFace, and ONNX Runtime.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=SystemStatusResponse, tags=["System Health"])
async def get_system_status():
    """Get system health, active ONNX execution provider, and database stats."""
    db: FaceDatabase = app.state.db
    db.load()
    people = db.list_people()
    return {
        "success": True,
        "backend": app.state.embedder.backend_name.upper(),
        "threshold": config.MATCH_THRESHOLD,
        "min_face_size": config.MIN_FACE_SIZE,
        "min_det_score": config.MIN_DET_SCORE,
        "enrolled_count": len(people),
        "total_templates": sum(p["num_images"] for p in people)
    }

@app.get("/api/identities", tags=["Gallery Management"])
async def list_identities():
    """Retrieve all enrolled identities and metadata."""
    db: FaceDatabase = app.state.db
    db.load()
    people = db.list_people()
    return {"success": True, "identities": people}

@app.post("/api/remove", tags=["Gallery Management"])
async def remove_identity_post(req: DeleteIdentityRequest):
    """Remove an identity from the database (called by web dashboard)."""
    db: FaceDatabase = app.state.db
    db.load()
    name = req.name.strip()
    target_name = None
    if name in db.records:
        target_name = name
    else:
        for k in db.records.keys():
            if k.lower() == name.lower() or k.replace("_", " ").lower() == name.lower():
                target_name = k
                break

    if target_name and db.remove(target_name):
        return {"success": True, "name": target_name, "message": f"Successfully removed '{target_name}'."}
    return JSONResponse(status_code=404, content={"success": False, "error": f"Identity '{name}' not found."})

@app.delete("/api/identities", tags=["Gallery Management"])
async def delete_identity(req: DeleteIdentityRequest):
    """Delete an identity from the persistent database (REST standard)."""
    return await remove_identity_post(req)

@app.post("/api/enroll", response_model=EnrollResponse, tags=["Biometrics"])
async def enroll_person(payload: EnrollRequest):
    """
    Enroll a new individual with one or more base64 facial images.
    Uses asyncio.to_thread to run CPU-bound feature extraction in a worker threadpool.
    """
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")

    embedder: Embedder = app.state.embedder
    db: FaceDatabase = app.state.db

    def _sync_enroll():
        added_count = 0
        skipped_count = 0
        person_entry = db.records.get(name, {"embeddings": [], "sources": []})

        for idx, b64_img in enumerate(payload.images):
            try:
                img = decode_base64_image(b64_img)
                faces = embedder.get_faces(img)
                if not faces:
                    skipped_count += 1
                    continue
                # Pick largest face
                face = faces[0]
                person_entry["embeddings"].append(face["embedding"])
                person_entry["sources"].append(f"webcam_capture_{idx+1}.jpg")
                added_count += 1
            except Exception:
                skipped_count += 1

        if added_count > 0:
            db.records[name] = person_entry
            db.save()
        return added_count, skipped_count

    # Execute in worker threadpool to prevent event loop blocking
    added, skipped = await asyncio.to_thread(_sync_enroll)

    if added == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "name": name,
                "added_count": 0,
                "skipped_count": skipped,
                "message": "No valid faces could be detected in the provided images."
            }
        )

    return {
        "success": True,
        "name": name,
        "added_count": added,
        "skipped_count": skipped,
        "message": f"Successfully enrolled {name} with {added} face template(s)."
    }

@app.post("/api/identify", response_model=IdentifyResponse, tags=["Biometrics"])
async def identify_face(payload: IdentifyRequest):
    """
    Detect faces in query image, extract 512D ArcFace embeddings, and match against gallery.
    Non-blocking execution using asyncio.to_thread.
    """
    threshold = payload.threshold if payload.threshold is not None else config.MATCH_THRESHOLD

    def _decode():
        if payload.image:
            return decode_base64_image(payload.image)
        elif payload.image_path:
            full_p = config.BASE_DIR / payload.image_path
            if not full_p.exists():
                raise FileNotFoundError(f"Image path not found: {payload.image_path}")
            return load_image(full_p)
        else:
            raise ValueError("Either 'image' or 'image_path' must be provided.")

    try:
        img = await asyncio.to_thread(_decode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    embedder: Embedder = app.state.embedder
    matcher: FaceMatcher = app.state.matcher
    db: FaceDatabase = app.state.db

    def _inference_and_match(query_img):
        faces = embedder.get_faces(query_img)
        if not faces:
            return []

        db.load()
        db_embs = db.get_all_embeddings()
        results = []

        for face in faces:
            q_emb = face["embedding"]
            match_res = matcher.match(q_emb, db_embs, top_k=5, threshold_override=threshold)

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
        return results

    # Run inference in threadpool without blocking FastAPI event loop
    face_results = await asyncio.to_thread(_inference_and_match, img)

    if not face_results:
        return {
            "success": True,
            "threshold": threshold,
            "num_faces": 0,
            "faces": [],
            "message": "No face detected in image."
        }

    return {
        "success": True,
        "threshold": threshold,
        "num_faces": len(face_results),
        "faces": face_results,
        "message": f"Detected and analyzed {len(face_results)} face(s)."
    }

@app.get("/api/samples", tags=["Evaluation"])
async def get_evaluation_samples():
    """Retrieve sample known and unknown faces from the LFW test split."""
    samples = []
    if config.SPLITS_PATH.exists():
        try:
            with open(config.SPLITS_PATH) as f:
                splits = json.load(f)

            known_added = set()
            for item in splits.get("test_known", []):
                p = item["person"]
                if p not in known_added and len(known_added) < 4:
                    known_added.add(p)
                    samples.append({
                        "category": "Known Identity (Enrolled/Test)",
                        "person": p,
                        "path": str(item["path"]).replace("\\", "/")
                    })

            unknown_added = set()
            for item in splits.get("test_unknown", []):
                p = item["person"]
                if p not in unknown_added and len(unknown_added) < 4:
                    unknown_added.add(p)
                    samples.append({
                        "category": "Unknown Intruder (Unenrolled)",
                        "person": p,
                        "path": str(item["path"]).replace("\\", "/")
                    })
        except Exception:
            pass
    return {"success": True, "samples": samples}

@app.get("/api/metrics", tags=["Evaluation"])
async def get_evaluation_metrics():
    """Retrieve benchmark metrics, failure logs, and ROC/DET plot metadata."""
    metrics_data = {}
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            metrics_data = json.load(f)

    failure_text = ""
    if config.FAILURE_SUMMARY_PATH.exists():
        with open(config.FAILURE_SUMMARY_PATH) as f:
            failure_text = f.read()

    failure_images = []
    if config.FAILURES_DIR.exists():
        for img_p in sorted(config.FAILURES_DIR.glob("*.jpg")):
            failure_images.append({
                "filename": img_p.name,
                "url": f"/results/failures/{img_p.name}"
            })

    return {
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
    }

# ---------------------------------------------------------------------------
# Static Files & Dashboard UI Mounting
# ---------------------------------------------------------------------------

# Mount static web UI assets
if (config.BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")

if config.PLOTS_DIR.exists():
    app.mount("/results/plots", StaticFiles(directory=str(config.PLOTS_DIR)), name="plots")

if config.FAILURES_DIR.exists():
    app.mount("/results/failures", StaticFiles(directory=str(config.FAILURES_DIR)), name="failures")

if config.DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(config.DATA_DIR)), name="data")

@app.get("/", response_class=FileResponse, tags=["Dashboard UI"])
async def serve_index():
    """Serve the glassmorphic FaceID frontend dashboard."""
    index_file = config.BASE_DIR / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>FaceID API running. Frontend not found.</h1>", status_code=404)

# ---------------------------------------------------------------------------
# Local Server Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import socket

    def get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    print("=" * 64)
    print("  [*] FaceID.ai Biometric Microservice is Live!")
    print(f"  [-] Local PC:       http://127.0.0.1:8000/")
    print(f"  [-] Local Network:  http://{local_ip}:8000/")
    print(f"  [-] Swagger Docs:   http://127.0.0.1:8000/docs")
    print("=" * 64)
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
