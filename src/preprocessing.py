"""
Landmark Preprocessing and Normalization Module.
Ensures translation and scale invariance for hand landmarks across training and real-time inference.
"""

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9  # Used as reference distance for palm scale normalization


class LandmarkNormalizer:
    """
    Normalizes 3D hand landmark coordinates (21, 3) or (63,) arrays.
    Subtracts wrist position (translation invariance) and scales by palm distance (scale invariance).
    """

    def __init__(self, reference_pair: tuple = (WRIST_INDEX, MIDDLE_MCP_INDEX)):
        self.ref_idx_1, self.ref_idx_2 = reference_pair

    def normalize_landmarks(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Normalize 21 x 3 3D landmark array.

        Args:
            landmarks_3d: np.ndarray of shape (21, 3) or (63,)

        Returns:
            np.ndarray of same shape normalized for translation and scale.
        """
        if landmarks_3d is None:
            raise ValueError("Input landmarks_3d cannot be None")

        is_flattened = landmarks_3d.ndim == 1 or (landmarks_3d.ndim == 2 and landmarks_3d.shape[1] != 3)

        if is_flattened:
            if landmarks_3d.size != 63:
                raise ValueError(f"Expected 63 values for flattened landmarks, got {landmarks_3d.size}")
            coords = landmarks_3d.reshape((21, 3))
        else:
            coords = landmarks_3d.copy()

        # Step 1: Translation Invariance - relative to wrist (landmark 0)
        wrist = coords[self.ref_idx_1]
        translated = coords - wrist

        # Step 2: Scale Invariance - normalize by wrist-to-middle-finger-base distance
        ref_pt = translated[self.ref_idx_2]
        scale = np.linalg.norm(ref_pt)

        # Fallback if scale is too small (e.g. invalid points)
        if scale < 1e-6:
            # Use max landmark distance from wrist as fallback scale
            scale = np.max(np.linalg.norm(translated, axis=1))
            if scale < 1e-6:
                scale = 1.0

        normalized = translated / scale

        if is_flattened:
            return normalized.flatten()
        return normalized

    def batch_normalize(self, landmarks_batch: np.ndarray) -> np.ndarray:
        """
        Normalize a 2D array of flattened landmarks (N, 63).
        """
        if landmarks_batch.ndim != 2 or landmarks_batch.shape[1] != 63:
            raise ValueError(f"Expected batch shape (N, 63), got {landmarks_batch.shape}")

        normalized_batch = np.zeros_like(landmarks_batch)
        for i in range(landmarks_batch.shape[0]):
            normalized_batch[i] = self.normalize_landmarks(landmarks_batch[i])

        return normalized_batch
