"""
Unified Face Embedder Module.

Abstracts Face Detection, Facial Landmark Alignment, and Feature Vector Extraction
behind a single clean class interface (`Embedder`).

Backends:
1. Primary: InsightFace (`buffalo_l` model pack with RetinaFace detector & ArcFace 512D embedder via ONNXRuntime CPU).
2. Fallback: Facenet-PyTorch (MTCNN detector & InceptionResnetV1 512D embedder pretrained on VGGFace2).

Design Rationale & Interview Notes:
- L2 Normalization: Normalizing output embedding vectors to unit length (||e||_2 = 1.0)
  simplifies Cosine Similarity to a simple Vector Dot Product:
  cos_sim(A, B) = (A . B) / (||A||_2 * ||B||_2) = A . B
- ArcFace Loss: ArcFace applies an Additive Angular Margin penalty to the target angle,
  forcing hyper-spherical feature embeddings to have tight intra-class distance and large
  inter-class margin.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import config

class BaseEmbedderBackend:
    """Abstract interface for face embedding backends."""
    def get_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InsightFaceBackend(BaseEmbedderBackend):
    """
    Primary Embedder Backend using InsightFace (buffalo_l).
    Uses RetinaFace / SCRFD for detection and ArcFace ONNX model for 512D embedding.
    """
    def __init__(self, min_det_score: float = config.MIN_DET_SCORE):
        import insightface
        from insightface.app import FaceAnalysis

        print("[+] Initializing InsightFace (buffalo_l model pack on CPU)...")
        # Load buffalo_l with CPU execution provider (det_size=(320, 320) optimal for LFW 250x250 inputs)
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=min_det_score)
        self.min_det_score = min_det_score

    def get_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces and compute L2-normalized 512D ArcFace embeddings.
        InsightFace expects 3-channel BGR numpy array.
        """
        # Run InsightFace analysis
        faces = self.app.get(image)
        results = []

        for face in faces:
            det_score = float(face.det_score)
            if det_score < self.min_det_score:
                continue

            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # Filter faces smaller than minimum face size threshold
            if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                continue

            raw_embedding = face.embedding.astype(np.float32)
            
            # Ensure 512-dimensional vector
            if raw_embedding.shape[0] != config.EMBEDDING_DIM:
                continue

            # Compute L2 norm and normalize vector to unit length
            norm = np.linalg.norm(raw_embedding)
            if norm > 0:
                normalized_embedding = raw_embedding / norm
            else:
                normalized_embedding = raw_embedding

            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": det_score,
                "embedding": normalized_embedding,
                "area": w * h
            })

        # Sort detected faces by bounding box area descending (largest face first)
        results.sort(key=lambda x: x["area"], reverse=True)
        return results


class FacenetPyTorchBackend(BaseEmbedderBackend):
    """
    Fallback Embedder Backend using Facenet-PyTorch.
    Uses MTCNN for face detection/alignment and InceptionResnetV1 (pretrained on VGGFace2) for 512D embedding.
    """
    def __init__(self, min_det_score: float = config.MIN_DET_SCORE):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        print("[+] Initializing Facenet-PyTorch fallback (MTCNN + InceptionResnetV1)...")
        self.device = torch.device("cpu")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=config.MIN_FACE_SIZE,
            select_largest=False,
            post_process=True,
            keep_all=True,
            device=self.device
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.min_det_score = min_det_score

    def get_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using MTCNN and compute L2-normalized 512D embeddings via InceptionResNetV1.
        Facenet-PyTorch expects PIL image or RGB numpy array.
        """
        import torch
        from PIL import Image

        # Convert BGR OpenCV image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)

        # Detect bounding boxes and probabilities
        boxes, probs = self.mtcnn.detect(pil_img)
        if boxes is None or probs is None:
            return []

        # Extract aligned face crops tensor
        aligned_faces = self.mtcnn(pil_img)
        if aligned_faces is None:
            return []

        if len(aligned_faces.shape) == 3:
            aligned_faces = aligned_faces.unsqueeze(0)

        results = []
        with torch.no_grad():
            embeddings = self.model(aligned_faces).cpu().numpy()

        for i, (bbox, prob) in enumerate(zip(boxes, probs)):
            if i >= len(embeddings):
                break
            det_score = float(prob) if prob is not None else 0.0
            if det_score < self.min_det_score:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            w, h = max(0, x2 - x1), max(0, y2 - y1)

            if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                continue

            raw_embedding = embeddings[i].astype(np.float32)
            norm = np.linalg.norm(raw_embedding)
            if norm > 0:
                normalized_embedding = raw_embedding / norm
            else:
                normalized_embedding = raw_embedding

            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": det_score,
                "embedding": normalized_embedding,
                "area": w * h
            })

        results.sort(key=lambda x: x["area"], reverse=True)
        return results


class Embedder:
    """
    Unified Embedder wrapper class used across the application.
    Automatically initializes the primary InsightFace backend, falling back to Facenet-PyTorch
    if InsightFace fails to load.
    """
    def __init__(self, backend_override: Optional[str] = None):
        self.backend_name = backend_override or config.PREFERRED_BACKEND
        self.backend = None

        if self.backend_name == "insightface":
            try:
                self.backend = InsightFaceBackend()
            except Exception as e:
                print(f"[!] Primary InsightFace backend failed to initialize: {e}")
                print("[!] Falling back to facenet-pytorch...")
                self.backend_name = "facenet_pytorch"

        if self.backend is None and self.backend_name == "facenet_pytorch":
            try:
                self.backend = FacenetPyTorchBackend()
            except Exception as e:
                print(f"[!] Facenet-PyTorch fallback failed: {e}")
                raise RuntimeError(
                    "All embedding backends failed to initialize. "
                    "Ensure onnxruntime, insightface, or facenet-pytorch are correctly installed."
                ) from e

        print(f"[+] Active Embedder Backend: {self.backend_name.upper()}")

    def get_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image and return list of face dicts:
        [
          {
            "bbox": [x1, y1, x2, y2],
            "score": float,        # Face detection confidence
            "embedding": np.ndarray (512,), L2-normalized float32
            "area": int
          }, ...
        ]
        Sorted largest face first. Returns empty list [] if no valid faces detected.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []
        
        try:
            return self.backend.get_faces(image)
        except Exception as e:
            print(f"[!] Exception during face extraction: {e}")
            return []
