"""
Feature Engineering Module for Hand Landmark Processing.
Extracts normalized spatial, distance, joint angle, and orientation features.
"""

import numpy as np
import logging
from typing import List, Tuple, Union, Optional
from src.preprocessing import LandmarkNormalizer

logger = logging.getLogger(__name__)

# Key landmark indices (MediaPipe 21 Hand Landmarks)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class FeatureExtractor:
    """
    Engineers geometric features from 21 3D hand landmarks.
    Features include normalized coordinates, inter-fingertip distances,
    fingertip-to-wrist distances, joint bending angles, and palm tilt orientation.
    """

    def __init__(self, use_normalization: bool = True):
        self.use_normalization = use_normalization
        self.normalizer = LandmarkNormalizer()
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive feature column names."""
        names = []
        # 1. 63 Normalized 3D Coordinates
        for i in range(21):
            names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])

        # 2. Inter-fingertip distances
        names.extend([
            "dist_thumb_index",
            "dist_index_middle",
            "dist_middle_ring",
            "dist_ring_pinky",
            "dist_thumb_pinky"
        ])

        # 3. Fingertip-to-wrist distances
        names.extend([
            "dist_wrist_thumb",
            "dist_wrist_index",
            "dist_wrist_middle",
            "dist_wrist_ring",
            "dist_wrist_pinky"
        ])

        # 4. Fingertip-to-MCP extension distances
        names.extend([
            "ext_thumb",
            "ext_index",
            "ext_middle",
            "ext_ring",
            "ext_pinky"
        ])

        # 5. Joint angles (degrees)
        names.extend([
            "angle_thumb_mcp",
            "angle_index_pip",
            "angle_index_dip",
            "angle_middle_pip",
            "angle_middle_dip",
            "angle_ring_pip",
            "angle_pinky_pip"
        ])

        # 6. Palm Orientation & Tilt
        names.extend([
            "palm_normal_x",
            "palm_normal_y",
            "palm_normal_z",
            "palm_tilt_roll",
            "palm_tilt_pitch"
        ])

        return names

    @staticmethod
    def _calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by vectors (p1-p2) and (p3-p2) in degrees.
        """
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def extract_features(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Extract complete feature vector from 21x3 or 63-element landmark array.

        Args:
            landmarks_3d: shape (21, 3) or (63,)

        Returns:
            np.ndarray: 1D feature vector of shape (N_features,)
        """
        if landmarks_3d is None:
            raise ValueError("Input landmarks cannot be None")

        if landmarks_3d.size != 63:
            raise ValueError(f"Expected 63 landmark coordinates, got {landmarks_3d.size}")

        if landmarks_3d.ndim == 1:
            coords = landmarks_3d.reshape((21, 3))
        else:
            coords = landmarks_3d.copy()

        # Step 1: Preprocessing / Normalization
        if self.use_normalization:
            coords = self.normalizer.normalize_landmarks(coords)

        features = []

        # 1. 63 Normalized Relative 3D Coordinates
        features.extend(coords.flatten())

        # 2. Inter-fingertip distances
        features.append(np.linalg.norm(coords[THUMB_TIP] - coords[INDEX_TIP]))
        features.append(np.linalg.norm(coords[INDEX_TIP] - coords[MIDDLE_TIP]))
        features.append(np.linalg.norm(coords[MIDDLE_TIP] - coords[RING_TIP]))
        features.append(np.linalg.norm(coords[RING_TIP] - coords[PINKY_TIP]))
        features.append(np.linalg.norm(coords[THUMB_TIP] - coords[PINKY_TIP]))

        # 3. Fingertip-to-wrist distances
        features.append(np.linalg.norm(coords[THUMB_TIP] - coords[WRIST]))
        features.append(np.linalg.norm(coords[INDEX_TIP] - coords[WRIST]))
        features.append(np.linalg.norm(coords[MIDDLE_TIP] - coords[WRIST]))
        features.append(np.linalg.norm(coords[RING_TIP] - coords[WRIST]))
        features.append(np.linalg.norm(coords[PINKY_TIP] - coords[WRIST]))

        # 4. Fingertip-to-MCP extension distances
        features.append(np.linalg.norm(coords[THUMB_TIP] - coords[THUMB_MCP]))
        features.append(np.linalg.norm(coords[INDEX_TIP] - coords[INDEX_MCP]))
        features.append(np.linalg.norm(coords[MIDDLE_TIP] - coords[MIDDLE_MCP]))
        features.append(np.linalg.norm(coords[RING_TIP] - coords[RING_MCP]))
        features.append(np.linalg.norm(coords[PINKY_TIP] - coords[PINKY_MCP]))

        # 5. Finger Joint Angles
        features.append(self._calculate_angle(coords[THUMB_CMC], coords[THUMB_MCP], coords[THUMB_TIP]))
        features.append(self._calculate_angle(coords[INDEX_MCP], coords[INDEX_PIP], coords[INDEX_TIP]))
        features.append(self._calculate_angle(coords[INDEX_PIP], coords[INDEX_DIP], coords[INDEX_TIP]))
        features.append(self._calculate_angle(coords[MIDDLE_MCP], coords[MIDDLE_PIP], coords[MIDDLE_TIP]))
        features.append(self._calculate_angle(coords[MIDDLE_PIP], coords[MIDDLE_DIP], coords[MIDDLE_TIP]))
        features.append(self._calculate_angle(coords[RING_MCP], coords[RING_PIP], coords[RING_TIP]))
        features.append(self._calculate_angle(coords[PINKY_MCP], coords[PINKY_PIP], coords[PINKY_TIP]))

        # 6. Palm Orientation & Tilt
        v_index = coords[INDEX_MCP] - coords[WRIST]
        v_pinky = coords[PINKY_MCP] - coords[WRIST]
        palm_normal = np.cross(v_index, v_pinky)
        norm_val = np.linalg.norm(palm_normal)
        if norm_val > 1e-6:
            palm_normal = palm_normal / norm_val
        else:
            palm_normal = np.array([0.0, 0.0, 1.0])

        features.extend(palm_normal)

        # Roll and Pitch angle from palm normal vector
        roll = float(np.degrees(np.arctan2(palm_normal[1], palm_normal[0])))
        pitch = float(np.degrees(np.arctan2(palm_normal[2], np.sqrt(palm_normal[0]**2 + palm_normal[1]**2))))
        features.extend([roll, pitch])

        return np.array(features, dtype=np.float32)

    def extract_batch(self, landmarks_batch: np.ndarray) -> np.ndarray:
        """
        Extract features for a 2D array of landmarks (N, 63).
        """
        if landmarks_batch.ndim != 2 or landmarks_batch.shape[1] != 63:
            raise ValueError(f"Expected batch shape (N, 63), got {landmarks_batch.shape}")

        features_list = [self.extract_features(row) for row in landmarks_batch]
        return np.array(features_list, dtype=np.float32)
