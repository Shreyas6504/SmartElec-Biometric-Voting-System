"""
SmartElec Face Detection Module
Provides real-time face detection, grayscale normalization, and feature extraction
using OpenCV Haar Cascade Classifiers.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

# OpenCV will be imported dynamically or at module level
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from .config import FACE_IMG_SIZE, DATA_DIR


class FaceDetector:
    """
    Encapsulates OpenCV Haar Cascade face detection and preprocessing pipeline.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        self.cascade_path = cascade_path
        self.classifier = None
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """Finds and loads the Haar Cascade XML classifier."""
        if not OPENCV_AVAILABLE:
            return

        candidates = []
        if self.cascade_path and os.path.exists(self.cascade_path):
            candidates.append(self.cascade_path)

        # OpenCV builtin cascade path
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

        # Local fallback directory
        local_cascade = DATA_DIR / "haarcascade_frontalface_default.xml"
        candidates.append(str(local_cascade))

        for path in candidates:
            if os.path.exists(path):
                clf = cv2.CascadeClassifier(path)
                if not clf.empty():
                    self.classifier = clf
                    self.cascade_path = path
                    return

        # If not found yet, create classifier with default name (OpenCV searches internal paths)
        clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not clf.empty():
            self.classifier = clf
            self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def detect_faces(self, frame: np.ndarray, scale_factor: float = 1.2, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detects bounding boxes for all faces in a BGR frame.
        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        if self.classifier is None:
            raise RuntimeError("Haar Cascade classifier is not initialized or OpenCV is missing.")

        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to handle lighting variations
        gray = cv2.equalizeHist(gray)

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert to standard Python tuple list
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_face_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = FACE_IMG_SIZE) -> np.ndarray:
        """
        Crops, resizes to target_size (e.g. 50x50), converts to grayscale,
        and returns a flattened 1D feature vector of shape (2500,).
        """
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # Clamp bounding box coordinates within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return np.zeros(target_size[0] * target_size[1], dtype=np.float32)

        if len(crop.shape) == 3 and crop.shape[2] == 3:
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop_gray = crop

        resized = cv2.resize(crop_gray, target_size, interpolation=cv2.INTER_AREA)
        # Flatten into 1D vector (50 * 50 = 2500 dimensions)
        feature_vector = resized.flatten().astype(np.float32)
        return feature_vector

    def draw_face_hud(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, status: str = "ELIGIBLE") -> np.ndarray:
        """
        Draws high-tech bounding box and HUD overlays on the video frame.
        status colors:
            - ELIGIBLE: Bright Green (0, 255, 0)
            - ALREADY_VOTED: Amber/Orange (0, 165, 255)
            - UNREGISTERED: Vibrant Red (0, 0, 255)
            - SCANNING: Cyan (255, 255, 0)
        """
        color_map = {
            "ELIGIBLE": (0, 255, 0),        # Green
            "ALREADY_VOTED": (0, 165, 255),  # Orange
            "UNREGISTERED": (0, 0, 255),     # Red
            "SCANNING": (255, 255, 0),       # Cyan
            "INFO": (200, 200, 200)          # Gray
        }
        color = color_map.get(status, (0, 255, 0))

        x, y, w, h = bbox
        corner_len = int(min(w, h) * 0.25)
        thickness = 2

        # Draw tech-style corner accents
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        # Semi-transparent thin bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        # Label background pill
        label_text = f" {label} "
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

        label_y = max(y - 10, th + 10)
        cv2.rectangle(frame, (x, label_y - th - 5), (x + tw, label_y + 5), color, cv2.FILLED)
        cv2.putText(frame, label_text, (x, label_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return frame
