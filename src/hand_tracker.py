"""
Hand Tracking Module using OpenCV and MediaPipe.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HandLandmarkResult:
    hand_detected: bool = False
    landmarks_3d: Optional[np.ndarray] = None  # Shape (21, 3) float32
    landmarks_pixel: Optional[List[Tuple[int, int]]] = None # Shape (21, 2) int (x,y) in frame space
    handedness: str = "Unknown"
    confidence: float = 0.0
    mp_landmarks: Any = None


class HandTracker:
    """
    Wrapper around MediaPipe Hands for 21-landmark detection and frame rendering.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe hands solution safely."""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            logger.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise RuntimeError(f"MediaPipe initialization error: {e}")

    def process_frame(self, frame_bgr: np.ndarray) -> HandLandmarkResult:
        """
        Process a single BGR frame and extract 21 hand landmarks.

        Args:
            frame_bgr: BGR image numpy array from OpenCV

        Returns:
            HandLandmarkResult object containing landmark arrays and metadata.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return HandLandmarkResult(hand_detected=False)

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            results = self.hands.process(frame_rgb)
        except Exception as e:
            logger.warning(f"Error during MediaPipe process: {e}")
            return HandLandmarkResult(hand_detected=False)

        if not results.multi_hand_landmarks:
            return HandLandmarkResult(hand_detected=False)

        # Get primary (first/highest confidence) hand
        primary_hand_landmarks = results.multi_hand_landmarks[0]

        handedness = "Right"
        confidence = 1.0
        if results.multi_handedness and len(results.multi_handedness) > 0:
            hand_info = results.multi_handedness[0].classification[0]
            handedness = hand_info.label
            confidence = float(hand_info.score)

        # Extract 21 x, y, z normalized 3D landmarks
        landmarks_3d = np.zeros((21, 3), dtype=np.float32)
        landmarks_pixel = []

        for idx, lm in enumerate(primary_hand_landmarks.landmark):
            landmarks_3d[idx] = [lm.x, lm.y, lm.z]
            px, py = int(lm.x * w), int(lm.y * h)
            landmarks_pixel.append((px, py))

        return HandLandmarkResult(
            hand_detected=True,
            landmarks_3d=landmarks_3d,
            landmarks_pixel=landmarks_pixel,
            handedness=handedness,
            confidence=confidence,
            mp_landmarks=primary_hand_landmarks
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, result: HandLandmarkResult) -> np.ndarray:
        """
        Draw hand landmarks and skeleton onto the frame.
        """
        output_frame = frame_bgr.copy()
        if result.hand_detected and result.mp_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                output_frame,
                result.mp_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return output_frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands:
            self.hands.close()
            logger.info("MediaPipe Hands closed.")
