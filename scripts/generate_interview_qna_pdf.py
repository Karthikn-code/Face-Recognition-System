"""
Interview Preparation Guide PDF Generator for Code Nimbus Solutions AI/ML Intern Round 2.
Generates an executive, publication-grade study guide covering the Face Recognition System,
FastAPI, Gemini AI SDK, SentenceTransformers & CrossEncoder RAG, Multimodal Vision, and Observability.
"""

import os
import sys
import warnings
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

warnings.filterwarnings("ignore")

FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITALIC = "C:/Windows/Fonts/ariali.ttf"


class InterviewPrepPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(14, 15, 14)
        self.set_auto_page_break(auto=True, margin=15)
        
        # Load Arial Fonts
        self.add_font("ArialCustom", "", FONT_REGULAR)
        self.add_font("ArialCustom", "B", FONT_BOLD)
        self.add_font("ArialCustom", "I", FONT_ITALIC)
        
        # Color Palette
        self.c_navy = (15, 23, 42)        # Slate 900
        self.c_indigo = (79, 70, 229)     # Indigo 600
        self.c_sky = (14, 165, 233)       # Sky 500
        self.c_emerald = (16, 185, 129)   # Emerald 500
        self.c_amber = (245, 158, 11)     # Amber 500
        self.c_rose = (225, 29, 72)       # Rose 600
        self.c_bg_light = (248, 250, 252) # Slate 50
        self.c_code_bg = (241, 245, 249)  # Slate 100
        self.c_border = (226, 232, 240)   # Slate 200
        self.c_text = (30, 41, 59)        # Slate 800
        self.c_muted = (100, 116, 139)    # Slate 500

    def header(self):
        if self.page_no() > 1:
            self.set_xy(14, 7)
            self.set_font("ArialCustom", "B", 7.5)
            self.set_text_color(*self.c_indigo)
            self.cell(100, 4, "CODE NIMBUS SOLUTIONS | AI/ML INTERN ROUND 2 TECHNICAL INTERVIEW PREP", 0, 0, "L")
            self.set_font("ArialCustom", "", 7.5)
            self.set_text_color(*self.c_muted)
            self.cell(82, 4, "CAMPUS F2F MASTER STUDY GUIDE", 0, 1, "R")
            self.set_draw_color(*self.c_border)
            self.set_line_width(0.25)
            self.line(14, 12, 196, 12)
            self.set_y(15)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.25)
        self.line(14, 285, 196, 285)
        self.set_xy(14, 286.5)
        self.set_font("ArialCustom", "", 7.5)
        self.set_text_color(*self.c_muted)
        self.cell(110, 4, "Face Recognition & Biometrics | FastAPI | Gemini SDK | RAG | ONNX & Vision", 0, 0, "L")
        self.cell(72, 4, f"Page {self.page_no()}", 0, 0, "R")

    def chapter_banner(self, pillar_num, title, subtitle):
        if self.get_y() > 240:
            self.add_page()
        self.ln(3)
        x = self.get_x()
        y = self.get_y()
        w = 182
        
        # Banner container
        self.set_fill_color(15, 23, 42)
        self.rect(x, y, w, 14, "F")
        
        # Accent left bar
        self.set_fill_color(*self.c_indigo)
        self.rect(x, y, 4, 14, "F")
        
        # Title text
        self.set_xy(x + 7, y + 2)
        self.set_font("ArialCustom", "B", 10)
        self.set_text_color(255, 255, 255)
        self.cell(w - 10, 5, f"PILLAR {pillar_num}: {title.upper()}", 0, 1, "L")
        
        # Subtitle
        self.set_xy(x + 7, y + 7.5)
        self.set_font("ArialCustom", "I", 7.5)
        self.set_text_color(148, 163, 184)
        self.cell(w - 10, 4.5, subtitle, 0, 1, "L")
        self.set_y(y + 16)

    def question_block(self, q_num, question, importance="HIGH FREQUENCY"):
        if self.get_y() > 245:
            self.add_page()
            
        self.ln(2)
        x = self.get_x()
        y = self.get_y()
        w = 182
        
        if importance == "CRITICAL (MUST-KNOW)":
            badge_bg = (254, 226, 226)
            badge_fg = (220, 38, 38)
        elif importance == "DIRECT JD MATCH":
            badge_bg = (224, 231, 255)
            badge_fg = (67, 56, 202)
        else:
            badge_bg = (236, 253, 245)
            badge_fg = (4, 120, 87)
            
        # Question Header Box
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(*self.c_border)
        self.rect(x, y, w, 8.5, "DF")
        
        # Left tag
        self.set_fill_color(*self.c_indigo)
        self.rect(x, y, 2.5, 8.5, "F")
        
        # Question text
        self.set_xy(x + 5, y + 1.8)
        self.set_font("ArialCustom", "B", 8.5)
        self.set_text_color(*self.c_navy)
        
        q_label = f"Q{q_num}: {question}"
        if len(q_label) > 95:
            q_label = q_label[:92] + "..."
        self.cell(135, 5, q_label, 0, 0, "L")
        
        # Importance Badge
        badge_w = 38
        badge_x = x + w - badge_w - 2
        self.set_fill_color(*badge_bg)
        self.rect(badge_x, y + 1.6, badge_w, 5.2, "F")
        self.set_xy(badge_x, y + 1.6)
        self.set_font("ArialCustom", "B", 6.5)
        self.set_text_color(*badge_fg)
        self.cell(badge_w, 5.2, importance, 0, 1, "C")
        
        self.set_y(y + 10.5)

    def subhead(self, text, icon=">>"):
        self.ln(1)
        self.set_font("ArialCustom", "B", 8)
        self.set_text_color(*self.c_indigo)
        self.cell(0, 4.2, f"{icon} {text}", 0, 1, "L")
        self.set_text_color(*self.c_text)

    def body_text(self, text, bold_prefix="", indent=0):
        self.set_x(14 + indent)
        w = 182 - indent
        self.set_font("ArialCustom", "", 7.8)
        self.set_text_color(*self.c_text)
        
        if bold_prefix:
            self.set_font("ArialCustom", "B", 7.8)
            prefix_w = self.get_string_width(bold_prefix)
            self.cell(prefix_w + 1, 4.2, bold_prefix, 0, 0, "L")
            self.set_font("ArialCustom", "", 7.8)
            self.multi_cell(w - prefix_w - 1, 4.2, text)
        else:
            self.multi_cell(w, 4.2, text)

    def bullet_item(self, bold_title, description, indent=3):
        if self.get_y() > 265:
            self.add_page()
        self.set_x(14 + indent)
        w = 182 - indent
        
        # Small accent rectangle as bullet point
        bx = 14 + indent
        by = self.get_y() + 1.4
        self.set_fill_color(*self.c_indigo)
        self.rect(bx, by, 1.4, 1.4, "F")
        
        self.set_x(bx + 3)
        self.set_font("ArialCustom", "B", 7.8)
        self.set_text_color(*self.c_navy)
        title_str = f"{bold_title}: " if bold_title else ""
        title_w = self.get_string_width(title_str)
        self.cell(title_w, 4.2, title_str, 0, 0, "L")
        
        self.set_font("ArialCustom", "", 7.8)
        self.set_text_color(*self.c_text)
        rem_w = w - 3 - title_w
        self.multi_cell(rem_w, 4.2, description)

    def code_box(self, code_lines):
        if self.get_y() + len(code_lines) * 3.8 > 270:
            self.add_page()
            
        self.ln(1)
        x = 16
        y = self.get_y()
        w = 178
        h = len(code_lines) * 3.8 + 3
        
        self.set_fill_color(*self.c_code_bg)
        self.set_draw_color(*self.c_border)
        self.rect(x, y, w, h, "DF")
        
        # Accent left line
        self.set_fill_color(*self.c_sky)
        self.rect(x, y, 1.8, h, "F")
        
        self.set_xy(x + 4, y + 1.5)
        self.set_font("Courier", "", 7.2)
        self.set_text_color(15, 23, 42)
        for line in code_lines:
            self.set_x(x + 4)
            self.cell(w - 6, 3.8, line, 0, 1, "L")
        self.set_y(y + h + 1.5)

    def pro_tip_box(self, title, tip_text):
        if self.get_y() > 255:
            self.add_page()
            
        self.ln(1)
        x = 15
        y = self.get_y()
        w = 180
        
        self.set_font("ArialCustom", "", 7.5)
        lines = self.multi_cell(w - 10, 3.8, tip_text, dry_run=True, output="LINES")
        h = len(lines) * 3.8 + 7
        
        self.set_fill_color(255, 251, 235)  # Amber 50
        self.set_draw_color(251, 191, 36)   # Amber 400
        self.rect(x, y, w, h, "DF")
        
        self.set_fill_color(245, 158, 11)   # Amber 500
        self.rect(x, y, 2.5, h, "F")
        
        self.set_xy(x + 5, y + 1.5)
        self.set_font("ArialCustom", "B", 7.8)
        self.set_text_color(180, 83, 9)
        self.cell(w - 10, 4, f"INTERVIEW DEFENSE TIP: {title.upper()}", 0, 1, "L")
        
        self.set_xy(x + 5, y + 5.5)
        self.set_font("ArialCustom", "", 7.5)
        self.set_text_color(120, 53, 15)
        self.multi_cell(w - 10, 3.8, tip_text)
        self.set_y(y + h + 1.5)


def build_prep_pdf(output_path: Path):
    pdf = InterviewPrepPDF()
    pdf.add_page()
    
    # -------------------------------------------------------------
    # COVER / HEADER
    # -------------------------------------------------------------
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(14, 15, 182, 38, "F")
    
    # Accent top border
    pdf.set_fill_color(79, 70, 229)
    pdf.rect(14, 15, 182, 3, "F")
    
    pdf.set_xy(20, 21)
    pdf.set_font("ArialCustom", "B", 14.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(170, 7, "ROUND 2 TECHNICAL INTERVIEW: COMPREHENSIVE STUDY GUIDE", 0, 1, "L")
    
    pdf.set_xy(20, 28)
    pdf.set_font("ArialCustom", "B", 9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(170, 5, "Candidate Master Defense: Face Recognition System + Code Nimbus AI/ML Intern JD", 0, 1, "L")
    
    pdf.set_xy(20, 34)
    pdf.set_font("ArialCustom", "", 7.8)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(170, 5, "Company: Code Nimbus Solutions  |  Role: AI/ML Intern  |  Target Batch: 2027  |  Format: On-Campus F2F", 0, 1, "L")
    
    pdf.set_y(57)
    
    # Brief Overview Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(14, 55, 182, 22, "DF")
    pdf.set_fill_color(16, 185, 129)
    pdf.rect(14, 55, 3, 22, "F")
    
    pdf.set_xy(19, 57)
    pdf.set_font("ArialCustom", "B", 8.2)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(170, 4.5, "HOW TO USE THIS MASTER DEFENSE GUIDE IN YOUR CAMPUS INTERVIEW", 0, 1, "L")
    
    pdf.set_xy(19, 62)
    pdf.set_font("ArialCustom", "", 7.5)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(172, 3.8, 
        "This guide maps your Face Recognition project directly onto the JD requirements: FastAPI backend scaling, "
        "Gemini AI ecosystem (google-genai SDK), RAG with SentenceTransformers & CrossEncoder reranking, "
        "multimodal vision runtimes (CLIP, MediaPipe, OpenCV, ONNX), and observability (Prometheus/OpenTelemetry). "
        "Study the bold mathematical terms and code snippets to give confident, production-grade answers."
    )
    
    pdf.set_y(80)

    # -------------------------------------------------------------
    # PILLAR 1: FACE RECOGNITION ARCHITECTURE & CORE PIPELINE
    # -------------------------------------------------------------
    pdf.chapter_banner(1, "Face Recognition Project Architecture & Pipeline", "End-to-end design, detection, alignment, embeddings, and unknown rejection")
    
    pdf.question_block(1, "Can you give a 90-second elevator pitch of your project?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("The Pitch", 
        "I built an end-to-end, CPU-optimized Open-Set Face Recognition and Attendance System in Python with $0 cloud cost. "
        "It solves the 1:N identification problem where an incoming person could either be one of N enrolled users or an un-enrolled stranger. "
        "The architecture combines RetinaFace for face detection and 5-point landmark affine alignment, ArcFace (ResNet-50) executed via ONNX Runtime "
        "to produce 512-dimensional L2-normalized embeddings, and cosine similarity matching against a persistent template gallery. "
        "It features an empirically tuned operating threshold (tau = 0.34) achieving zero False Acceptance on LFW with strict zero data leakage, "
        "and includes a full-stack glassmorphic dashboard with live HTML5 webcam attendance and dynamic threshold control."
    )
    pdf.pro_tip_box("First Impression", "Lead with the engineering problem: 'Open-set recognition with unknown rejection'. Mentioning zero data leakage and ONNX CPU optimization immediately separates you from generic tutorial projects.")

    pdf.question_block(2, "Walk me through the exact pipeline from a camera frame to an ID decision.", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("Step 1 (Ingestion & Quality Filter)", "Raw BGR frame is captured via OpenCV. Bounding boxes smaller than 30px or detection confidence < 0.50 are filtered out.")
    pdf.bullet_item("Step 2 (RetinaFace Detection & Alignment)", "Detects 5 facial landmarks (two eyes, nose tip, two mouth corners). Uses an Affine Transformation to warp and normalize the face to a standardized 112x112 frontal canonical view.")
    pdf.bullet_item("Step 3 (ArcFace Feature Extraction)", "The 112x112 crop passes through an ArcFace ResNet-50 ONNX model, outputting a continuous 512-D latent vector.")
    pdf.bullet_item("Step 4 (L2 Normalization)", "The raw 512-D vector is normalized to unit length: e = v / ||v||_2. This guarantees ||e||_2 = 1.0.")
    pdf.bullet_item("Step 5 (Cosine Similarity Search)", "Computes dot product q · e_i against enrolled gallery templates. Since both are unit vectors, Cosine Similarity simplifies directly to Dot Product.")
    pdf.bullet_item("Step 6 (Unknown Rejection)", "If max similarity score s >= 0.34, assign person identity. If s < 0.34, safely reject as 'UNKNOWN'.")

    pdf.question_block(3, "Detection vs Verification vs Closed-Set vs Open-Set Identification?", "HIGH FREQUENCY")
    pdf.bullet_item("Face Detection", "Localizing face bounding boxes and facial landmarks in an image [x, y, w, h]. No identity classification.")
    pdf.bullet_item("Face Verification (1:1)", "One-to-one claim verification: 'Is this user Alice?' Compares query against Alice's template. Used in smartphone unlock.")
    pdf.bullet_item("Closed-Set Identification (1:N)", "Assumes the query person is ALWAYS enrolled in the database. Simply returns argmax(sim). Flawed for security because an intruder is always misclassified as the nearest enrolled person.")
    pdf.bullet_item("Open-Set Identification (1:N + Rejection)", "Our system's task. Solves both: 'Is this person enrolled?' (Thresholding) AND 'If yes, who are they?' (Argmax). Essential for real-world attendance and surveillance.")

    pdf.question_block(4, "How do you handle edge cases (no face, multiple faces, blur)?", "HIGH FREQUENCY")
    pdf.bullet_item("No Face Detected", "RetinaFace returns empty detection list; system gracefully yields 'NO_FACE_DETECTED' without crashing, skipping heavy neural inference.")
    pdf.bullet_item("Multiple Faces in Frame", "During Enrollment, the system automatically isolates the largest face (by bounding box area w * h) and logs a warning. During Identification, the system iterates over ALL detected faces independently and renders green boxes for known users and red boxes for unknown visitors.")
    pdf.bullet_item("Motion Blur & Low Light", "Mitigated by calculating the Laplacian Variance of the face crop (cv2.Laplacian(img, cv2.CV_64F).var()). Frames with variance < 100 are rejected as blurry before running ArcFace.")

    # -------------------------------------------------------------
    # PILLAR 2: MATHEMATICS OF EMBEDDINGS & ARCFACE
    # -------------------------------------------------------------
    pdf.chapter_banner(2, "Mathematics of Embeddings & ArcFace Loss", "Angular margin loss, hyperspherical geometry, and metric properties")

    pdf.question_block(5, "Why ArcFace? How does Additive Angular Margin Loss work mathematically?", "CRITICAL (MUST-KNOW)")
    pdf.body_text("Traditional Softmax Loss separates classes using hyperplanes in Euclidean space, but does not enforce compact intra-class clustering. Triplet Loss minimizes anchor-positive distance while maximizing anchor-negative distance, but suffers from combinatorial explosion and unstable semi-hard negative mining.")
    pdf.subhead("The ArcFace Formulation")
    pdf.body_text(
        "ArcFace reformulates Softmax on a normalized hypersphere. Feature vectors x_i and weight vectors W_j are normalized (||x_i||=1, ||W_j||=1), "
        "making W_j^T x_i = cos(theta_j). It then adds an Additive Angular Margin m (m = 0.5 rad) directly to the target ground-truth angle theta_y_i:"
    )
    pdf.code_box([
        "L_ArcFace = -log( exp(s * cos(theta_yi + m)) / ",
        "            (exp(s * cos(theta_yi + m)) + sum_{j != yi} exp(s * cos(theta_j))) )",
        "where s = feature scale parameter (typically 64.0), m = angular margin (0.5)"
    ])
    pdf.bullet_item("Intuition", "Because cosine is strictly monotonically decreasing on [0, pi], adding margin m penalizes the target angle, forcing the network during backpropagation to compress embeddings of the same identity into an extremely tight geodesic cone on the unit sphere.")

    pdf.question_block(6, "Why Cosine Similarity over Euclidean Distance? Why Dot Product?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("Direction vs Magnitude", "ArcFace maps identity to angular direction on the hypersphere. Vector magnitude often correlates with image brightness or contrast rather than identity. Cosine similarity evaluates pure angular direction: cos(theta) = (A · B) / (||A|| * ||B||).")
    pdf.bullet_item("Mathematical Equivalence", "Because all vectors in our pipeline are strictly L2-normalized (||q||_2 = 1.0, ||e||_2 = 1.0):")
    pdf.code_box([
        "Cosine_Similarity(q, e) = (q · e) / (1.0 * 1.0) = q · e = sum_{k=1}^{512} q_k * e_k"
    ])
    pdf.bullet_item("Production Speedup", "Computing dot products avoids square roots and divisions, enabling 1:N search to run as a single BLAS-optimized matrix multiplication (q @ E^T) across thousands of templates in <1ms.")

    pdf.question_block(7, "What is max_similarity vs mean_prototype matching?", "HIGH FREQUENCY")
    pdf.bullet_item("max_similarity (Default)", "Compares query vector q against ALL enrolled photo vectors {e_1, e_2, ...} for person P and takes max(q · e_i). Preserves multi-modal facial variation (e.g. smiling vs neutral, slight angle changes).")
    pdf.bullet_item("mean_prototype", "Calculates a single centroid vector u_P = normalize(mean(e_1, e_2, ...)). Faster storage, but can wash out extreme variations.")

    # -------------------------------------------------------------
    # PILLAR 3: METRICS, TUNING & STATISTICAL REALITY
    # -------------------------------------------------------------
    pdf.chapter_banner(3, "Evaluation Metrics, Threshold Tuning & Scale", "FAR, FRR, EER, zero-leakage protocol, and the 1:N Extreme Value problem")

    pdf.question_block(8, "Define FAR, FRR, and EER. How did you pick tau = 0.34?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("False Acceptance Rate (FAR)", "FAR = False Accepts / Total Impostor Queries. The rate at which an unauthorized stranger is mistakenly accepted as an enrolled user. (Security Risk).")
    pdf.bullet_item("False Rejection Rate (FRR)", "FRR = False Rejects / Total Genuine Queries. The rate at which an authorized enrolled user is rejected as unknown. (Convenience Issue).")
    pdf.bullet_item("Equal Error Rate (EER)", "The threshold where FAR equals FRR. Represents the intrinsic discriminative capacity of the model.")
    pdf.bullet_item("Zero-Leakage Tuning", "Threshold was evaluated across 81 increments (0.10 to 0.90) STRICTLY on the validation split (80 known, 80 unknown). We selected tau = 0.34 by optimizing: Maximize Known Top-1 Accuracy subject to FAR <= 1.0%. On validation, tau = 0.34 gave FAR = 0.0% and 100% accuracy. EER occurred at tau = 0.31.")

    pdf.question_block(9, "Your test set got 100% accuracy. Is that realistic in production? Why does it degrade?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("Honest Engineering Answer", "No, 100% test accuracy is an artifact of a small benchmark gallery (N = 40 people) and ArcFace's huge angular margin on LFW (genuine pairs score 0.55-0.85, while impostor pairs stay < 0.18).")
    pdf.bullet_item("The 1:N Extreme Value Problem", "In a real enterprise with N = 100,000 enrolled identities, an unknown visitor is compared against 100,000 random vectors. By Extreme Value Theory, the distribution of max_{i=1..N} (q · e_i) shifts significantly to the right. The probability of an accidental random alignment >= 0.34 increases drastically.")
    pdf.bullet_item("Production Mitigations", "1. Elevate threshold (tau >= 0.50). 2. Score Normalization (T-norm / Z-norm). 3. Multi-image verification and active liveness detection.")

    # -------------------------------------------------------------
    # PILLAR 4: FASTAPI & PRODUCTION BACKEND SCALING (JD MUST-HAVE)
    # -------------------------------------------------------------
    pdf.chapter_banner(4, "FastAPI Backend Architecture & Scaling", "Direct JD Requirement: High-performance ML APIs, async vs CPU threadpools, Pydantic")

    pdf.question_block(10, "How would you architect this Face Recognition System as a FastAPI service?", "DIRECT JD MATCH")
    pdf.bullet_item("Modular Router Structure", "Separate endpoints into api/v1/enroll.py, api/v1/identify.py, and api/v1/attendance.py using FastAPI APIRouter.")
    pdf.bullet_item("Lifespan Context Manager", "Load heavy ONNX models once during server startup using @asynccontextmanager, storing the embedder and vector index in app.state.")
    pdf.bullet_item("Pydantic Validation", "Define strict request/response schemas for payload validation and automatic OpenAPI Swagger documentation.")
    pdf.code_box([
        "from fastapi import FastAPI, UploadFile, File, Depends, HTTPException",
        "from pydantic import BaseModel, Field",
        "from typing import List, Optional",
        "",
        "class IdentifyResponse(BaseModel):",
        "    name: str",
        "    score: float = Field(..., ge=0.0, le=1.0)",
        "    is_known: bool",
        "    bbox: List[int]",
        "    timestamp: str"
    ])

    pdf.question_block(11, "In FastAPI, should /identify be 'async def' or 'def'? Explain Event Loop blocking.", "CRITICAL (MUST-KNOW)")
    pdf.body_text("THIS IS THE #1 FASTAPI INTERVIEW TRAP. Interviewers test if you understand the Python GIL and AsyncIO Event Loop.")
    pdf.bullet_item("The Trap", "If you write 'async def identify(file: UploadFile):' and run CPU-heavy operations inside (OpenCV decode, RetinaFace ONNX inference, matrix multiplication), the synchronous CPU work BLOCKS the single-threaded asyncio event loop. No other request can be accepted or processed while that face is running!")
    pdf.bullet_item("The Solution (Option A - Standard def)", "Define the route with standard 'def identify(...)'. FastAPI automatically executes standard 'def' endpoints in an external threadpool (anyio worker threads), keeping the event loop 100% non-blocking.")
    pdf.bullet_item("The Solution (Option B - Explicit Worker Pool)", "If using 'async def', offload the CPU-bound inference explicitly to a worker threadpool using asyncio.to_thread:")
    pdf.code_box([
        "@app.post('/api/v1/identify', response_model=IdentifyResponse)",
        "async def identify_face(file: UploadFile = File(...)):",
        "    contents = await file.read() # Asynchronous I/O read",
        "    # Offload CPU-heavy inference to threadpool without blocking event loop:",
        "    result = await asyncio.to_thread(embedder.process_and_match, contents)",
        "    return result"
    ])

    pdf.question_block(12, "Base64 JSON vs Multipart/form-data (UploadFile) in ML APIs?", "HIGH FREQUENCY")
    pdf.bullet_item("Base64 JSON", "Pros: Easy to transmit over WebSockets or single JSON payloads from frontend HTML5 canvases. Cons: Base64 encoding inflates payload size by ~33% and consumes CPU to decode strings back to byte buffers.")
    pdf.bullet_item("UploadFile (Multipart)", "Pros: Streams raw binary bytes directly into SpooledTemporaryFile with zero encoding overhead; highly efficient for multi-megapixel images and batch enrollments.")
    pdf.bullet_item("Best Practice", "Use Base64 for live webcam streaming frames (<100KB); use UploadFile for batch enrollment photos.")

    # -------------------------------------------------------------
    # PILLAR 5: GEMINI AI ECOSYSTEM & MULTIMODAL GENAI (JD REQUIREMENT)
    # -------------------------------------------------------------
    pdf.chapter_banner(5, "Gemini AI Ecosystem & Multimodal GenAI", "Direct JD Requirement: google-genai SDK, multimodal reasoning, and structured outputs")

    pdf.question_block(13, "What is the new google-genai SDK, and how does it differ from older libraries?", "DIRECT JD MATCH")
    pdf.bullet_item("google-genai SDK", "Google's modern, official unified Python SDK for Gemini 1.5 Flash, 1.5 Pro, and 2.0 models. It replaces legacy packages with a cleaner client-centric architecture.")
    pdf.bullet_item("Key Enhancements", "1. Single unified Client (client = genai.Client()). 2. Native Pydantic structured output validation (response_schema). 3. Native multimodal support (images, video, audio, PDFs). 4. System instructions, thinking models, and built-in function calling.")

    pdf.question_block(14, "How would you integrate Gemini into your Face Recognition & Attendance System?", "DIRECT JD MATCH")
    pdf.bullet_item("Use Case 1 (Multimodal ID Card KYC Verification)", "During employee/student enrollment, capture a webcam selfie and a photo of their physical ID card. Feed both images to Gemini 1.5 Flash to verify facial match, extract text (Name, Roll No, Department), and detect fake/tampered cards.")
    pdf.bullet_item("Use Case 2 (Intruder Incident Auto-Reporting)", "When an 'UNKNOWN' intruder is detected during off-hours, pass the video clip/frames to Gemini to generate an automated incident narrative: 'Unknown male subject in dark hoodie entered through corridor B at 11:42 PM; facial features matched no registered personnel.'")
    pdf.bullet_item("Use Case 3 (Visual Demographics & Emotion Analytics)", "Use Gemini's multimodal reasoning to extract auxiliary non-biometric attributes (glasses, mask, approximate mood/fatigue) for workplace wellness analytics.")

    pdf.question_block(15, "Write a Python snippet using google-genai SDK for multimodal analysis with Pydantic.", "DIRECT JD MATCH")
    pdf.code_box([
        "from google import genai",
        "from google.genai import types",
        "from pydantic import BaseModel",
        "from PIL import Image",
        "",
        "class IDVerificationResult(BaseModel):",
        "    id_card_name: str",
        "    faces_match: bool",
        "    confidence_score: float",
        "    tampering_detected: bool",
        "    notes: str",
        "",
        "client = genai.Client(api_key='YOUR_GEMINI_API_KEY')",
        "response = client.models.generate_content(",
        "    model='gemini-2.0-flash',",
        "    contents=[Image.open('selfie.jpg'), Image.open('student_id.jpg'),",
        "              'Verify if the person in the selfie matches the student ID card.'],",
        "    config=types.GenerateContentConfig(",
        "        response_mime_type='application/json',",
        "        response_schema=IDVerificationResult,",
        "        system_instruction='You are an enterprise biometric security compliance auditor.'",
        "    )",
        ")",
        "result: IDVerificationResult = response.parsed"
    ])

    # -------------------------------------------------------------
    # PILLAR 6: SEMANTIC SEARCH & RAG PIPELINES (JD REQUIREMENT)
    # -------------------------------------------------------------
    pdf.chapter_banner(6, "Semantic Search & RAG Pipelines", "Direct JD Requirement: SentenceTransformers, CrossEncoder reranking, and vector search")

    pdf.question_block(16, "How do Face Embeddings connect to Text Embeddings in Semantic Search?", "DIRECT JD MATCH")
    pdf.bullet_item("Conceptual Equivalence", "Both map unstructured raw inputs into a dense vector space (hypersphere) where semantic meaning or biological identity equals geometric distance.")
    pdf.bullet_item("Face Recognition", "Raw face pixels -> ArcFace (ResNet-50) -> 512-D vector -> Cosine similarity.")
    pdf.bullet_item("Semantic Search", "Raw text sentence -> SentenceTransformer (BERT/RoBERTa) -> 384-D / 768-D vector -> Cosine similarity.")
    pdf.bullet_item("Indexing", "Both systems perform 1:N nearest neighbor retrieval against stored vector indices (FAISS, Chroma, Qdrant).")

    pdf.question_block(17, "Bi-Encoder (SentenceTransformers) vs Cross-Encoder (Reranker)? Why use both?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("Bi-Encoder (SentenceTransformers)", "Encodes Query q and Document d independently into vectors u and v. Similarity is simply u · v. Documents can be pre-embedded offline and stored in a vector index. Extremely fast: retrieves Top-50 candidates from millions of docs in <10ms. Drawback: No cross-token attention between query and document words.")
    pdf.bullet_item("Cross-Encoder (CrossEncoder Reranker)", "Passes Query and Document TOGETHER into full self-attention layers: [CLS] Query [SEP] Document. Captures rich token-level interaction, yielding vastly higher ranking accuracy. Drawback: Cannot pre-compute embeddings; requires running heavy transformer forward passes at query time. Too slow for full-database search.")
    pdf.bullet_item("The Two-Stage Pipeline (Industry Standard RAG)", 
        "Stage 1 (Retrieval): Bi-Encoder + Vector DB quickly retrieves top 50 candidates in <10ms. "
        "Stage 2 (Reranking): CrossEncoder scores the top 50 to select the top 5 most relevant chunks in <40ms. "
        "This achieves the speed of a Bi-Encoder with the precision of a Cross-Encoder."
    )
    pdf.code_box([
        "from sentence_transformers import SentenceTransformer, CrossEncoder",
        "",
        "# Stage 1: Fast Bi-Encoder Retrieval (ANN Vector Search)",
        "bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')",
        "query_vector = bi_encoder.encode('What is the attendance policy?')",
        "top_50_docs = vector_db.search(query_vector, top_k=50)",
        "",
        "# Stage 2: High-Precision Cross-Encoder Reranking Filter",
        "reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')",
        "pairs = [[query, doc.text] for doc in top_50_docs]",
        "scores = reranker.predict(pairs)",
        "top_5_docs = [doc for _, doc in sorted(zip(scores, top_50_docs), reverse=True)[:5]]"
    ])

    pdf.question_block(18, "How would you build an Enterprise RAG pipeline from scratch?", "HIGH FREQUENCY")
    pdf.bullet_item("1. Document Ingestion & Chunking", "Parse documents (PDF, Markdown). Apply recursive token-aware chunking (chunk size: 512 tokens, overlap: 50 tokens) to preserve semantic context across chunk boundaries.")
    pdf.bullet_item("2. Dense Vector Indexing", "Compute embeddings with SentenceTransformers ('all-mpnet-base-v2') and index into a vector DB with HNSW indexing.")
    pdf.bullet_item("3. Hybrid Search", "Combine BM25 keyword search (sparse) with dense vector search to capture both exact domain terms and semantic intent via Reciprocal Rank Fusion (RRF).")
    pdf.bullet_item("4. Cross-Encoder Reranking", "Filter candidate pool down to top 3-5 high-relevance chunks using a CrossEncoder.")
    pdf.bullet_item("5. Context Assembly & LLM Generation", "Inject reranked chunks into Gemini 1.5 Flash with strict grounding prompt: 'Answer using only the provided context. If unknown, state unverified.'")

    # -------------------------------------------------------------
    # PILLAR 7: VISION RUNTIMES, CLIP & OBSERVABILITY
    # -------------------------------------------------------------
    pdf.chapter_banner(7, "Vision Frameworks, Edge Runtimes & Observability", "CLIP, MediaPipe, OpenCV, ONNX Runtime, Prometheus, and OpenTelemetry")

    pdf.question_block(19, "What is CLIP and how does Multimodal Contrastive Learning work?", "DIRECT JD MATCH")
    pdf.bullet_item("CLIP (Contrastive Language-Image Pretraining)", "OpenAI model featuring two parallel encoders: a Vision Transformer (ViT) for images and a Text Transformer for text captions.")
    pdf.bullet_item("Shared Embedding Space", "Trained using InfoNCE loss on 400M (image, text) pairs. It projects both image vectors and text vectors into the SAME shared metric hypersphere.")
    pdf.bullet_item("Multimodal Search Application", "Enables zero-shot text-to-image queries: To find 'a person wearing a medical mask', embed the text with CLIP text encoder, compute cosine similarity against image embeddings, and retrieve matching frames.")

    pdf.question_block(20, "Why ONNX Runtime? How does it optimize CPU inference in your project?", "DIRECT JD MATCH")
    pdf.bullet_item("Graph Optimizations", "Merges redundant layers (Operator Fusion: Conv + BatchNorm + ReLU merged into a single optimized fused kernel). Eliminates dead computation nodes.")
    pdf.bullet_item("Multi-Threaded Parallelism", "ONNX Runtime leverages OpenMP and intra-op thread allocation to saturate all CPU cores during ArcFace inference, dropping latency to ~150ms per face on commodity laptops.")
    pdf.bullet_item("INT8 Quantization", "Converts 32-bit floating point weights to 8-bit integers (Post-Training Quantization). Reduces model memory footprint by 75% (from 250MB to ~60MB) and doubles CPU throughput with <1% accuracy loss.")

    pdf.question_block(21, "MediaPipe vs OpenCV vs RetinaFace: When do you use each?", "DIRECT JD MATCH")
    pdf.bullet_item("OpenCV", "General-purpose computer vision library: image I/O, resizing, colorspace conversion (BGR to RGB), geometric affine transformations, matrix drawing.")
    pdf.bullet_item("MediaPipe", "Ultra-lightweight edge framework by Google. Ideal for real-time mobile/browser tasks (BlazeFace runs at 100+ FPS, 468-point 3D Face Mesh, hand landmark tracking). Best for client-side liveness detection.")
    pdf.bullet_item("RetinaFace", "Heavy deep learning detector with Feature Pyramid Networks (FPN). Slower than MediaPipe BlazeFace, but significantly higher accuracy on tiny faces, extreme occlusion, and poor lighting.")

    pdf.question_block(22, "How would you monitor this API with Prometheus, Grafana, and OpenTelemetry?", "DIRECT JD MATCH")
    pdf.bullet_item("Prometheus (Metrics)", "Instrument FastAPI with prometheus-fastapi-instrumentator to expose a /metrics endpoint. Track: 1. Histogram: 'http_request_duration_seconds' (monitor p95/p99 latency). 2. Counter: 'face_identification_total{status=\"matched|unknown|no_face\"}'. 3. Gauge: 'enrolled_gallery_size'.")
    pdf.bullet_item("Grafana (Visualization)", "Build dashboards displaying real-time FPS throughput, inference latency percentiles, error rate spikes, and CPU/RAM memory utilization.")
    pdf.bullet_item("OpenTelemetry (Distributed Tracing)", "Add trace spans to decompose end-to-end request latency: Image Decode Span (5ms) -> RetinaFace Detection Span (90ms) -> ArcFace Embedding Span (120ms) -> Vector Search Span (2ms). If p95 latency spikes, trace spans pinpoint the exact bottleneck.")

    # -------------------------------------------------------------
    # PILLAR 8: SCALING TO 100,000 IDENTITIES & SECURITY
    # -------------------------------------------------------------
    pdf.chapter_banner(8, "Production Scaling to 100,000+ Identities & Anti-Spoofing", "Vector databases (FAISS, Milvus), HNSW, and presentation attack mitigation")

    pdf.question_block(23, "How would you scale your template database from 40 to 100,000 people?", "CRITICAL (MUST-KNOW)")
    pdf.bullet_item("The Bottleneck", "Linear scan (NumPy matrix dot product) takes O(N * D). For N = 100,000 templates, computing 100k dot products per incoming camera frame causes high CPU latency and stalls.")
    pdf.bullet_item("Solution 1 (FAISS Vector Indexing)", "Replace linear search with FAISS (Facebook AI Similarity Search) using HNSW (Hierarchical Navigable Small World) graphs or IVF-PQ (Inverted File with Product Quantization). Search complexity drops from O(N) to O(log N), returning nearest neighbors in <3ms.")
    pdf.bullet_item("Solution 2 (Vector Database)", "Deploy Milvus, Qdrant, or Pgvector. Decouples vector indexing and filtering from the web server with horizontal scalability.")
    pdf.bullet_item("Solution 3 (Microservices Architecture)", "Decouple into an asynchronous ingestion queue (RabbitMQ/Kafka) -> GPU Face Worker Pods (for RetinaFace + ArcFace) -> Vector Search Service (gRPC).")

    pdf.question_block(24, "How do you protect the system against Presentation Attacks (Spoofing)?", "HIGH FREQUENCY")
    pdf.bullet_item("The Attack", "An adversary holds up a printed color photograph, tablet video, or 3D silicone mask of an enrolled employee to spoof attendance.")
    pdf.bullet_item("Passive Anti-Spoofing", "Deploy a lightweight CNN (e.g. MiniFASNet) to analyze high-frequency Fourier spectrum, specular screen reflections, and moire interference patterns typical of digital screens.")
    pdf.bullet_item("Active Liveness", "Prompt the user with dynamic real-time challenges: 'Blink twice', 'Turn head 30 degrees right', or track eye pupil dilation against a random screen color flash.")
    pdf.bullet_item("Multi-Spectral Hardware", "Integrate Near-Infrared (NIR) or 3D Time-of-Flight (ToF) depth cameras; printed photos and phone screens appear completely black or flat under infrared illumination.")

    # -------------------------------------------------------------
    # PILLAR 9: CAMPUS INTERVIEW CHEAT SHEET & BEHAVIORAL
    # -------------------------------------------------------------
    pdf.chapter_banner(9, "Round 2 Campus Interview Strategy & Behavioral Script", "The 60-second intro, answering trade-offs, and high-impact questions to ask")

    pdf.question_block(25, "The 60-Second Self-Introduction & Project Pitch", "CRITICAL (MUST-KNOW)")
    pdf.body_text(
        "\"Hi, I'm Karthik, a 2027 graduate passionate about production AI engineering and backend systems. "
        "In my recent project, I built a production-grade Face Recognition and Attendance platform optimized for zero cloud cost and CPU execution. "
        "I tackled core machine learning challenges like open-set recognition with unknown rejection using ArcFace 512-dimensional embeddings and RetinaFace alignment. "
        "I ensured zero data leakage with a rigorous train/val/test protocol on LFW and tuned the decision threshold empirically. "
        "Beyond Computer Vision, I'm deeply interested in high-performance ML backends with FastAPI, modern GenAI workflows using the google-genai SDK, "
        "and RAG architectures combining SentenceTransformers with CrossEncoder reranking. "
        "I'm really excited about the AI/ML Intern role at Code Nimbus Solutions because it aligns directly with what I love building: "
        "deploying scalable, production-ready AI pipelines that deliver real business impact.\""
    )

    pdf.question_block(26, "Top 3 Thoughtful Questions to Ask the Interviewer", "HIGH FREQUENCY")
    pdf.bullet_item("Question 1", "\"In Code Nimbus's production RAG workflows, what reranking strategy or CrossEncoder models have you found strike the best balance between latency and retrieval accuracy?\"")
    pdf.bullet_item("Question 2", "\"How does the team currently handle observability and trace latency across multimodal models—do you use OpenTelemetry and Prometheus, and what are your p95 latency targets?\"")
    pdf.bullet_item("Question 3", "\"For an intern joining the AI/ML team, what does a successful first month look like in terms of shipping models or FastAPI microservices to production?\"")

    # Output to disk
    pdf.output(str(output_path))
    print(f"[+] Successfully generated Interview Preparation Guide PDF at: {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "Face_Recognition_Interview_Prep_Guide.pdf"
    build_prep_pdf(out_file)
