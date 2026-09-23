"""
Quick Technical Cheat Sheet PDF Generator from INTERVIEW_NOTES.md
Generates a clean, professional, publication-quality PDF reference for interview defense.
"""

import os
import sys
import warnings
from fpdf import FPDF

warnings.filterwarnings("ignore")

FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_ITALIC = "C:/Windows/Fonts/ariali.ttf"

class CheatSheetPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(12, 12, 12)
        self.set_auto_page_break(auto=True, margin=14)
        
        self.add_font("ArialCustom", "", FONT_REGULAR)
        self.add_font("ArialCustom", "B", FONT_BOLD)
        self.add_font("ArialCustom", "I", FONT_ITALIC)
        
        # Colors
        self.c_navy = (15, 23, 42)        # Slate 900
        self.c_indigo = (79, 70, 229)     # Indigo 600
        self.c_blue = (37, 99, 235)       # Blue 600
        self.c_card_bg = (248, 250, 252)  # Slate 50
        self.c_border = (226, 232, 240)   # Slate 200
        self.c_text = (30, 41, 59)        # Slate 800
        self.c_muted = (100, 116, 139)    # Slate 500

    def header(self):
        if self.page_no() > 1:
            self.set_xy(12, 6)
            self.set_font("ArialCustom", "B", 7.5)
            self.set_text_color(*self.c_indigo)
            self.cell(110, 4, "FACE RECOGNITION SYSTEM | ENGINEERING INTERVIEW CHEAT SHEET", 0, 0, "L")
            self.set_font("ArialCustom", "", 7.5)
            self.set_text_color(*self.c_muted)
            self.cell(76, 4, "HIGH-YIELD DEFENSE NOTES", 0, 1, "R")
            self.set_draw_color(*self.c_border)
            self.set_line_width(0.2)
            self.line(12, 11, 198, 11)
            self.set_y(14)

    def footer(self):
        self.set_y(-10)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.2)
        self.line(12, 287, 198, 287)
        self.set_font("ArialCustom", "", 7.5)
        self.set_text_color(*self.c_muted)
        self.cell(110, 4, "Candidate Quick Study Guide | Code Nimbus Solutions AI/ML Round 2", 0, 0, "L")
        self.cell(76, 4, f"Page {self.page_no()}", 0, 0, "R")

    def section_header(self, q_num, question):
        if self.get_y() > 250:
            self.add_page()
        self.ln(2)
        y = self.get_y()
        self.set_fill_color(238, 242, 255) # Light Indigo
        self.rect(12, y, 186, 7.5, "F")
        self.set_fill_color(*self.c_indigo)
        self.rect(12, y, 2.5, 7.5, "F") # Accent strip
        
        self.set_xy(16, y)
        self.set_font("ArialCustom", "B", 8.5)
        self.set_text_color(*self.c_indigo)
        self.cell(180, 7.5, f"Q{q_num}: {question}", 0, 1, "L")
        self.ln(1)

    def bullet(self, title, content):
        if self.get_y() > 265:
            self.add_page()
        self.set_font("ArialCustom", "B", 8)
        self.set_text_color(*self.c_navy)
        bullet_char = "-"
        self.cell(4, 4.2, bullet_char, 0, 0, "L")
        self.cell(self.get_string_width(f"{title}: ") + 1, 4.2, f"{title}: ", 0, 0, "L")
        
        self.set_font("ArialCustom", "", 8)
        self.set_text_color(*self.c_text)
        self.multi_cell(0, 4.2, content)
        self.ln(0.6)

    def math_box(self, formula_text, explanation=""):
        if self.get_y() > 260:
            self.add_page()
        self.set_fill_color(*self.c_card_bg)
        self.set_draw_color(*self.c_border)
        self.set_line_width(0.2)
        y = self.get_y()
        h = 7.5 if not explanation else 11.5
        self.rect(16, y, 178, h, "FD")
        self.set_xy(18, y + 1.2)
        self.set_font("ArialCustom", "B", 8)
        self.set_text_color(*self.c_blue)
        self.cell(0, 4, formula_text, 0, 1, "L")
        if explanation:
            self.set_x(18)
            self.set_font("ArialCustom", "I", 7.5)
            self.set_text_color(*self.c_muted)
            self.cell(0, 4, explanation, 0, 1, "L")
        self.set_y(y + h + 1.5)


def build_pdf():
    pdf = CheatSheetPDF()
    pdf.add_page()

    # Title Banner
    pdf.set_fill_color(15, 23, 42) # Slate 900
    pdf.rect(12, 12, 186, 17, "F")
    
    pdf.set_xy(16, 14.5)
    pdf.set_font("ArialCustom", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(100, 6, "Face Recognition System - Interview Defense Cheat Sheet", 0, 1, "L")
    
    pdf.set_xy(16, 21)
    pdf.set_font("ArialCustom", "", 8)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(178, 4, "High-Yield Technical Explanations, Mathematical Formulations, Architecture Trade-Offs & Scaling", 0, 1, "L")
    pdf.set_y(31)

    # Q1
    pdf.section_header(1, "Why do we use Cosine Similarity instead of Euclidean Distance?")
    pdf.bullet("Direction vs. Magnitude", "Face embeddings are trained via angular margin loss (ArcFace) mapping identities to directional positions on a 512D hypersphere. Angle reflects identity, whereas vector magnitude reflects image brightness, lighting intensity, or sensor artifacts.")
    pdf.bullet("L2-Norm Efficiency", "All embeddings are L2-normalized to unit length (||e|| = 1.0). Cosine similarity simplifies directly into the vector dot product:")
    pdf.math_box("CosineSim(q, e) = (q . e) / (||q|| * ||e||)  ==>  q . e", "Avoids computing square roots and divisions during 1:N real-time matrix matching (q . E^T).")
    pdf.bullet("Mathematical Relation", "Euclidean distance squared on unit vectors is proportional: d^2 = 2 - 2*(q . e). Cosine similarity provides an intuitive, bounded confidence scale [-1.0, 1.0].")

    # Q2
    pdf.section_header(2, "What exactly is a Face Embedding, and how does ArcFace work?")
    pdf.bullet("512D Semantic Vector", "A compact continuous 512-dimensional vector extracted by ResNet-50 where facial geometry is mapped so identical people have small angular distance and different people have large angular distance.")
    pdf.bullet("ArcFace Innovation", "Traditional Softmax separates classes using linear hyperplanes without enforcing tight intra-class clustering. ArcFace injects an Additive Angular Margin m = 0.5 directly to target ground-truth angle theta_y:")
    pdf.math_box("cos(theta_y + m)   with scale s = 64", "Because cosine is monotonically decreasing on [0, pi], adding margin m penalizes angle, forcing embeddings into tight geodesic cones.")

    # Q3
    pdf.section_header(3, "How did you select the operating threshold (tau = 0.60 / 0.34), and what is the FAR/FRR trade-off?")
    pdf.bullet("Zero-Leakage Tuning", "Tuned strictly on validation split (80 known queries, 80 unknown queries) across 81 threshold increments (0.10 to 0.90).")
    pdf.bullet("Optimal Metrics", "At tau = 0.34: FAR = 0.0%, Top-1 Accuracy = 100.0%. Equal Error Rate (EER) where FAR == FRR occurred at tau = 0.31.")
    pdf.bullet("Production Choice (tau = 0.60)", "Higher threshold (0.60) prioritizes security in biometric access control. Intruders are strictly blocked (FAR -> 0.0), even if genuine users occasionally face false rejections under bad lighting (FRR).")
    pdf.math_box("FAR = False Accepts / Total Impostor Attempts  |  FRR = False Rejects / Total Genuine Attempts", "High Security (tau >= 0.60): Low FAR, higher FRR  <===>  High Convenience (tau <= 0.20): Low FRR, higher FAR")

    # Q4
    pdf.section_header(4, "Why did your test set achieve 100% accuracy, and is that realistic for production?")
    pdf.bullet("Engineering Honesty", "100% accuracy was achieved due to: (1) Small gallery size (N = 40), and (2) ArcFace's huge angular margin (genuine pairs: 0.55 - 0.85; impostor pairs: < 0.18). Threshold 0.34/0.60 easily separates them.")
    pdf.bullet("Degradation at Scale", "In open-set matching with N = 100,000 enrolled people, extreme value theory dictates that the maximum impostor similarity among random dot products increases significantly: max(q . e_i).")
    pdf.bullet("Production Scaling", "At scale, multi-factor verification, dynamic score calibration, and strict thresholds (tau >= 0.60) are required.")

    # Q5
    pdf.section_header(5, "How are edge cases (no face, multiple faces, corrupt inputs) handled?")
    pdf.bullet("No Face Detected", "If RetinaFace confidence < 0.50 or box < 30px, system gracefully returns empty list without throwing exceptions. Matcher marks candidate as 'no_face' and safely rejects.")
    pdf.bullet("Multiple Faces", "Enrollment: Selects largest bounding box by pixel area (w * h) as the intended subject. Identification: Detects, aligns, and matches all faces simultaneously in a vectorized loop.")
    pdf.bullet("Corrupt Images", "Handled via try-except in cv2 decoding; raises clear 400 Bad Request error instead of silent server death.")

    # Q6
    pdf.section_header(6, "How would you scale this system from 40 to 100,000 enrolled identities?")
    pdf.bullet("FAISS Sub-Linear Search", "Brute-force scan takes O(N * d). For 100,000 vectors, linear search creates latency bottlenecks. We replace it with FAISS HNSW (Hierarchical Navigable Small World) or IVF-PQ graphs, dropping retrieval to O(log N) (<5ms).")
    pdf.bullet("Vector Databases", "Decouple storage into dedicated vector engines (Milvus, Qdrant, or Pinecone) with gRPC microservice separation.")
    pdf.bullet("Score Calibration", "Apply T-norm / Z-norm score normalization to adjust threshold dynamically per individual, preventing dense cluster bias.")

    # Q7
    pdf.section_header(7, "What are the primary production failure modes and mitigations?")
    pdf.bullet("Spoofing / Presentation Attacks", "Printed photos, phone screens, 3D masks fool 2D RGB. Mitigation: Passive liveness detection (texture analysis, reflection detection) or IR/Depth sensors.")
    pdf.bullet("Severe Pose / Yaw (>45 deg)", "Affine landmark alignment fails when eye landmarks are occluded. Mitigation: Multi-pose enrollment (frontal, +30 deg, -30 deg) and 3D face mesh normalization.")
    pdf.bullet("Motion Blur / Poor Light", "Loss of high-frequency spatial features. Mitigation: Pre-capture Laplacian variance blur detector (rejecting frames with var < 100) before neural inference.")

    # Code Nimbus JD Quick Alignment Table
    pdf.ln(2.5)
    y_card = pdf.get_y()
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(12, y_card, 186, 6.5, "F")
    pdf.set_xy(16, y_card)
    pdf.set_font("ArialCustom", "B", 8.5)
    pdf.set_text_color(*pdf.c_navy)
    pdf.cell(180, 6.5, "Code Nimbus Solutions AI/ML Intern - Core JD Competency Checklist", 0, 1, "L")
    pdf.ln(1)

    topics = [
        ("FastAPI Microservices", "Async endpoints, Pydantic schemas, threadpool execution (asyncio.to_thread), OpenAPI /docs."),
        ("Vector Embeddings & RAG", "512D unit-norm vectors, Cosine similarity = dot product, FAISS HNSW sub-linear indexing."),
        ("Multimodal Vision Models", "RetinaFace (localization + 5-point landmarks) + ArcFace (512D deep facial semantics)."),
        ("Generative AI & LLMs", "Gemini 2.5 Flash SDK (google-genai) for automated visual audit reporting and analysis."),
        ("Production Observability", "Prometheus latency histograms, OpenTelemetry distributed tracing, Grafana dashboards.")
    ]

    for title, desc in topics:
        pdf.set_font("ArialCustom", "B", 7.5)
        pdf.set_text_color(*pdf.c_indigo)
        pdf.cell(42, 4.2, f"[x] {title}:", 0, 0, "L")
        pdf.set_font("ArialCustom", "", 7.5)
        pdf.set_text_color(*pdf.c_text)
        pdf.multi_cell(0, 4.2, desc)
        pdf.ln(0.5)

    out_path = "Face_Recognition_Quick_Cheat_Sheet.pdf"
    pdf.output(out_path)
    print(f"[+] Successfully generated: {out_path}")

if __name__ == "__main__":
    build_pdf()
