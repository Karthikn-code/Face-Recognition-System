# Face Recognition System

A lightweight, open-set face identification and verification service built with Python, InsightFace (ArcFace), OpenCV, and FastAPI.

The system performs facial detection, landmark alignment, 512-dimensional feature embedding extraction, and cosine similarity matching against an enrolled identity gallery with open-set unknown rejection.

---

## Architecture Overview

```
Input Image ──► Face Detection & Alignment (RetinaFace / 5 Landmarks)
                     │
                     ▼
             Feature Extraction (ArcFace ResNet-50 ──► 512-D L2-Normalized Vector)
                     │
                     ▼
             Cosine Similarity Matching (Dot Product against Gallery)
                     │
                     ▼
             Decision Logic [Score >= Threshold (τ)?]
               ├── Yes ──► Identified Person
               └── No  ──► Unknown Rejection
```

### Key Components

- **Detection & Alignment**: Uses RetinaFace / SCRFD (`buffalo_l`) to detect faces and align 5 facial landmarks (eyes, nose, mouth corners) via affine transformation to a standardized $112 \times 112$ crop.
- **Feature Extractor**: ArcFace with ResNet-50 backbone producing $L_2$-normalized 512-dimensional embeddings ($\|\mathbf{e}\|_2 = 1.0$).
- **Matcher**: Open-set 1:N cosine similarity matching. Because vectors are unit-normalized, cosine similarity reduces to a fast dot product $\mathbf{q} \cdot \mathbf{e}_i$.
- **Unknown Rejection**: Queries with similarity below the decision threshold ($\tau = 0.34$) are safely rejected as `unknown`.
- **API & UI**: FastAPI backend with async endpoints and a lightweight web interface for camera enrollment and verification.

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/Karthikn-code/Face-Recognition-System.git
cd Face-Recognition-System

python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download & Prepare Benchmark Data

Downloads a subset of the standard LFW (Labeled Faces in the Wild) dataset partitioned into known and unknown identities for evaluation:

```bash
python scripts/prepare_dataset.py
```

### 3. Run Unit Tests

```bash
pytest
```

### 4. Launch the Web Application

```bash
# Run FastAPI server (includes interactive docs at http://127.0.0.1:8000/docs)
python fastapi_app.py
```

Open **http://127.0.0.1:8000** in your browser to access the webcam enrollment and verification interface.

---

## CLI Usage

You can also interact with the system via the command-line interface:

```bash
# Enroll images for an identity
python main.py enroll --name "Alice" --images path/to/photo1.jpg path/to/photo2.jpg

# Enroll from a directory
python main.py enroll --name "Bob" --folder path/to/bob_photos/

# Identify faces in a query photo
python main.py identify --image path/to/query.jpg --save-annotated results/output.jpg

# List enrolled people
python main.py list

# Remove an enrolled person
python main.py remove --name "Alice"
```

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/status` | `GET` | Health check, active model, and gallery template counts |
| `/api/identities` | `GET` | List all enrolled identities and metadata |
| `/api/enroll` | `POST` | Enroll a person with name and base64-encoded images |
| `/api/identify` | `POST` | Run face detection, alignment, and 1:N matching on an image |
| `/api/identities` | `DELETE` | Remove an enrolled identity from the gallery |

Interactive OpenAPI documentation is available at `/docs` when running `fastapi_app.py`.

---

## Repository Structure

```
Face-Recognition-System/
├── fastapi_app.py         # FastAPI REST service & static web mount
├── app.py                 # Fallback standalone HTTP server
├── main.py                # Command-line interface (CLI)
├── config.py              # System configuration and hyperparameters
├── requirements.txt       # Project dependencies
├── src/
│   ├── database.py        # Face template storage and persistence
│   ├── embedder.py        # RetinaFace detection + ArcFace embedding interface
│   ├── matcher.py         # Cosine similarity calculation & threshold decision
│   ├── system.py          # Unified end-to-end recognition pipeline
│   ├── evaluate.py        # Evaluation pipeline (FAR, FRR, ROC metrics)
│   └── utils.py           # Image I/O and visualization helpers
├── static/                # Frontend web dashboard (HTML, CSS, JS)
├── scripts/
│   ├── prepare_dataset.py # LFW dataset partitioner
│   └── generate_failure_cases.py # Boundary stress test evaluation
└── tests/                 # Unit test suite
```

---

## Evaluation & Operational Threshold

The decision threshold was calibrated on a held-out validation split of LFW:

- **Operating Threshold ($\tau = 0.34$)**: Chosen to minimize False Accept Rate ($\text{FAR} \le 1.0\%$) while maintaining high true identification rates.
- **Equal Error Rate (EER)**: Observed near $\tau = 0.31$ where $\text{FAR} \approx \text{FRR}$.

Run the evaluation script to reproduce metrics and generate ROC curves:

```bash
python -m src.evaluate
```

> **Production Note**: While performance on the 40-identity LFW benchmark shows clean separation between genuine and impostor pairs, production deployments with large galleries ($N > 10,000$) should incorporate approximate nearest neighbor search (e.g., FAISS / HNSW) and presentation attack detection (anti-spoofing).

---

## Tech Stack

- **Deep Learning / CV**: InsightFace (`buffalo_l`), ONNX Runtime, OpenCV
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Vanilla JavaScript, HTML5 Canvas / MediaDevices API, CSS
- **Testing**: Pytest
