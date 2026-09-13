"""
Real-Time Gesture Predictor with Confidence Thresholding and Temporal Smoothing.
"""

import os
import sys
import time
import joblib
import logging
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import FeatureExtractor

logger = logging.getLogger("GesturePredictor")


class GesturePredictor:
    """
    Predicts hand gesture from raw landmarks using the trained model,
    enforcing confidence thresholds and temporal sliding-window smoothing.
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
        confidence_threshold: float = 0.65,
        smoothing_window: int = 5
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_window_size = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.feature_extractor = FeatureExtractor()

        self._load_model_and_preprocessor(model_path, preprocessor_path)

    def _load_model_and_preprocessor(self, model_path: str, preprocessor_path: str):
        """Safely load trained model and preprocessor artifacts."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first using 'python train.py'.")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}.")

        try:
            self.model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            self.scaler = preprocessor["scaler"]
            self.label_encoder = preprocessor["label_encoder"]
            self.feature_names = preprocessor["feature_names"]

            logger.info("Successfully loaded gesture model and preprocessor pipeline.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError(f"Failed to load model artifacts: {e}")

    def predict(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        """
        Predict gesture for a single frame 3D landmark array (21, 3) or (63,).

        Returns:
            Dict containing:
                - predicted_gesture: smoothed final gesture label (str)
                - raw_prediction: instantaneous raw prediction label (str)
                - confidence: float confidence score [0.0, 1.0]
                - latency_ms: inference latency in milliseconds
                - is_fallback: bool whether confidence fell below threshold
        """
        if landmarks_3d is None or landmarks_3d.size != 63:
            return {
                "predicted_gesture": "NEUTRAL",
                "raw_prediction": "NEUTRAL",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "is_fallback": True
            }

        start_time = time.time()

        # 1. Feature Engineering
        features_1d = self.feature_extractor.extract_features(landmarks_3d)

        # 2. Scaling (using trained StandardScaler)
        features_scaled = self.scaler.transform(features_1d.reshape(1, -1))

        # 3. Model Inference
        raw_pred_idx = self.model.predict(features_scaled)[0]
        raw_gesture = self.label_encoder.inverse_transform([raw_pred_idx])[0]

        # 4. Confidence Estimation
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 1.0  # Fallback if classifier does not output probabilities

        latency_ms = (time.time() - start_time) * 1000

        # Confidence Thresholding
        is_fallback = False
        effective_gesture = raw_gesture
        if confidence < self.confidence_threshold:
            effective_gesture = "NEUTRAL"
            is_fallback = True

        # 5. Temporal Smoothing (Sliding Window Majority Voting)
        self.prediction_history.append(effective_gesture)
        smoothed_gesture = self._majority_vote()

        return {
            "predicted_gesture": smoothed_gesture,
            "raw_prediction": raw_gesture,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 3),
            "is_fallback": is_fallback
        }

    def _majority_vote(self) -> str:
        """Computes majority vote over prediction history deque."""
        if not self.prediction_history:
            return "NEUTRAL"

        counts = {}
        for g in self.prediction_history:
            counts[g] = counts.get(g, 0) + 1

        # Return gesture with max votes
        best_gesture = max(counts.keys(), key=lambda k: counts[k])
        return best_gesture

    def reset_smoothing(self):
        """Reset temporal prediction queue."""
        self.prediction_history.clear()
