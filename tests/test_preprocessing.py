"""
Unit tests for Landmark Normalization.
"""

import pytest
import numpy as np
from src.preprocessing import LandmarkNormalizer


def test_landmark_normalizer_shape():
    normalizer = LandmarkNormalizer()
    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)

    norm_3d = normalizer.normalize_landmarks(dummy_landmarks)
    assert norm_3d.shape == (21, 3)

    norm_flat = normalizer.normalize_landmarks(dummy_landmarks.flatten())
    assert norm_flat.shape == (63,)


def test_landmark_normalizer_translation_invariance():
    normalizer = LandmarkNormalizer()
    base_landmarks = np.random.rand(21, 3).astype(np.float32)

    norm1 = normalizer.normalize_landmarks(base_landmarks)

    # Shift all landmarks by (10.0, -5.0, 2.5)
    shifted_landmarks = base_landmarks + np.array([10.0, -5.0, 2.5])
    norm2 = normalizer.normalize_landmarks(shifted_landmarks)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)


def test_landmark_normalizer_scale_invariance():
    normalizer = LandmarkNormalizer()
    base_landmarks = np.random.rand(21, 3).astype(np.float32)

    norm1 = normalizer.normalize_landmarks(base_landmarks)

    # Scale relative to wrist
    scaled_landmarks = (base_landmarks - base_landmarks[0]) * 3.5 + base_landmarks[0]
    norm2 = normalizer.normalize_landmarks(scaled_landmarks)

    np.testing.assert_allclose(norm1, norm2, atol=1e-5)


def test_invalid_input():
    normalizer = LandmarkNormalizer()
    with pytest.raises(ValueError):
        normalizer.normalize_landmarks(np.zeros(10))
