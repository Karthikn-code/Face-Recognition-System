"""
Utility functions for image I/O, format conversions, visualization, and logging.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Tuple
import cv2
import numpy as np


def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path or return it if it is already a numpy ndarray.
    Returns standard 3-channel uint8 BGR image format.
    Raises ValueError if image file cannot be read.
    """
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        if len(image_input.shape) == 2:  # Grayscale to BGR
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input

    path_str = str(image_input)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Image path does not exist: {path_str}")

    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image from path: {path_str}")

    return image


def draw_annotations(
    image: np.ndarray,
    results: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    unknown_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Draw bounding boxes, labels, matching scores, and detection confidence on image copy.

    Args:
        image: Original BGR image ndarray.
        results: List of dicts containing face identification results:
                 [{"bbox": [x1, y1, x2, y2], "det_score": float, "name": str, "score": float}, ...]
    Returns:
        Annotated BGR numpy image.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    # Calculate proportional font scale and thickness based on image dimensions
    font_scale = max(0.5, min(width, height) / 600.0)
    thickness = max(1, int(min(width, height) / 300.0))

    for item in results:
        bbox = item.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = item.get("name", "Unknown")
        match_score = item.get("score", 0.0)
        det_score = item.get("det_score", 0.0)

        color = unknown_color if name.lower() == "unknown" else box_color

        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text: "Alice (0.78)" or "Unknown (0.24)"
        label = f"{name} ({match_score:.2f})"
        if det_score > 0:
            label += f" [det:{det_score:.2f}]"

        # Label background pill
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_y = max(y1 - 5, h + 5)
        cv2.rectangle(
            annotated,
            (x1, text_y - h - baseline - 2),
            (x1 + w + 4, text_y + baseline),
            color,
            -1  # Filled rectangle background
        )
        
        # Draw white text over color pill
        cv2.putText(
            annotated,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save BGR image to output_path creating parent directories if needed."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), image)
